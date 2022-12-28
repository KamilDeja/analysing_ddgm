import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
from guided_diffusion.two_parts_model import TwoPartsUNetModel
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import numpy.matlib
from matplotlib.colors import LogNorm
import matplotlib as mpl

from . import logger

if os.uname().nodename == "titan4":
    from guided_diffusion import dist_util_titan as dist_util
elif os.uname().nodename == "node7001.grid4cern.if.pw.edu.pl":
    from guided_diffusion import dist_util_dwarf as dist_util
else:
    from guided_diffusion import dist_util
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler, TaskAwareSampler
from .gaussian_diffusion import _extract_into_tensor
from .nn import mean_flat
from .losses import normal_kl
from .resample import LossAwareSampler, UniformSampler, TaskAwareSampler, DAEOnlySampler

import wandb

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
# INITIAL_LOG_LOSS_SCALE = 20.0
from .unet import UNetModel


class TrainLoop:
    def __init__(
            self,
            *,
            params,
            model,
            prev_model,
            diffusion,
            data,
            batch_size,
            microbatch,
            lr,
            ema_rate,
            log_interval,
            skip_save,
            save_interval,
            plot_interval,
            resume_checkpoint,
            task_id,
            use_fp16=False,
            fp16_scale_growth=1e-3,
            schedule_sampler=None,
            weight_decay=0.0,
            lr_anneal_steps=0,
            scheduler_rate=1,
            scheduler_step=1000,
            num_steps=10000,
            image_size=32,
            in_channels=3,
            class_cond=False,
            max_class=None,
            generate_previous_examples_at_start_of_new_task=False,
            generate_previous_samples_continuously=False,
            validator=None,
            validation_interval=None,
            semi_supervised_training=False
    ):
        self.params = params
        self.task_id = task_id
        self.model = model
        self.prev_ddp_model = prev_model
        self.diffusion = diffusion
        self.data = data
        self.image_size = image_size
        self.in_channels = in_channels
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.class_cond = class_cond
        self.max_class = max_class
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.skip_save = skip_save
        self.plot_interval = plot_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.num_steps = num_steps

        self.step = 0
        self.resume_step = 0
        self.world_size = dist.get_world_size()
        self.global_batch = self.batch_size * self.world_size

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        self.scheduler = th.optim.lr_scheduler.ExponentialLR(self.opt, gamma=scheduler_rate)
        self.scheduler_step = scheduler_step
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]
        self.semi_supervised_training = semi_supervised_training
        if th.cuda.is_available():
            self.use_ddp = False
            find_unused_params = ((not isinstance(self.model, UNetModel)) and (
                not isinstance(self.schedule_sampler,
                               DAEOnlySampler))) or self.diffusion.skip_classifier_loss
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=find_unused_params,
            )
        else:
            if self.world_size > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model
        self.generate_previous_examples_at_start_of_new_task = generate_previous_examples_at_start_of_new_task
        self.generate_previous_samples_continuously = generate_previous_samples_continuously
        self.validator = validator
        if validator is None:
            self.validation_interval = self.num_steps + 1  # Skipping validation
        else:
            self.validation_interval = validation_interval

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)

            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        if isinstance(self.schedule_sampler, DAEOnlySampler):
            plot = self.plot_dae_only
        else:
            plot = self.plot
        while (
                (not self.lr_anneal_steps
                 or self.step + self.resume_step < self.lr_anneal_steps) and (self.step < self.num_steps)
        ):
            if self.step > 100:
                self.mp_trainer.skip_gradient_thr = self.params.skip_gradient_thr
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                if logger.get_rank_without_mpi_import() == 0:
                    wandb.log(logger.getkvs(), step=self.step)
                logger.dumpkvs()
            if (not self.skip_save) & (self.step % self.save_interval == 0) & (self.step != 0):
                self.save(self.task_id)
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            # make SNR plots
            if self.params.snr_log_interval > 0 and self.step % self.params.snr_log_interval == 0:
                self.snr_plots(batch, cond, self.task_id, self.step)
            if self.step > 0:
                if (self.step == 1000) or (self.step % self.plot_interval == 0):
                    plot(self.task_id, self.step)
                if self.step % self.scheduler_step == 0:
                    self.scheduler.step()
                if self.step % self.validation_interval == 0:
                    logger.log(f"Validation for step {self.step}")
                    if self.params.train_with_classifier:
                        preds, test_loss, test_accuracy = self.validator.calculate_accuracy_with_classifier(
                            model=self.model, task_id=self.task_id)
                        if logger.get_rank_without_mpi_import() == 0:
                            wandb.log({"preds": preds})
                            wandb.log({"test_classification_loss": test_loss})
                            wandb.log({"test_classification_accuracy": test_accuracy})
                            logger.log(f"Test classification loss: {test_loss}, Test acc: {test_accuracy}")
                    else:
                        if self.diffusion.dae_model:
                            dae_result = self.validate_dae()
                            logger.log(f"DAE test MAE: {dae_result:.3}")
                            if logger.get_rank_without_mpi_import() == 0:
                                wandb.log({"dae_test_MAE": dae_result})
                        if not isinstance(self.schedule_sampler, DAEOnlySampler):
                            fid_result, precision, recall = self.validator.calculate_results(train_loop=self,
                                                                                             task_id=self.task_id,
                                                                                             dataset=self.params.dataset,
                                                                                             n_generated_examples=self.params.n_examples_validation,
                                                                                             batch_size=self.params.microbatch if self.params.microbatch > 0 else self.params.batch_size)
                            if logger.get_rank_without_mpi_import() == 0:
                                wandb.log({"fid": fid_result})
                                wandb.log({"precision": precision})
                                wandb.log({"recall": recall})
                            logger.log(f"FID: {fid_result}, Prec: {precision}, Rec: {recall}")

            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if not self.skip_save:
            if (self.step - 1) % self.save_interval != 0:
                self.save(self.task_id)
        if (self.step - 1) % self.plot_interval != 0:
            plot(self.task_id, self.step)
        if self.params.snr_log_interval > 0:
            self.snr_plots(batch, cond, self.task_id, self.step)
            self.draw_final_snr_plot(self.step, self.task_id)

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i: i + self.microbatch].to(dist_util.dev())
            # micro_cond = cond[i: i + self.microbatch].to(dist_util.dev())  # {
            micro_cond = {
                k: v[i: i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            if isinstance(self.schedule_sampler, TaskAwareSampler):
                t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev(), micro_cond["y"],
                                                          self.task_id)
            else:
                t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            if self.generate_previous_samples_continuously and (self.task_id > 0):
                shape = [self.batch_size, self.in_channels, self.image_size, self.image_size]
                prev_loss = self.diffusion.calculate_loss_previous_task(current_model=self.ddp_model,
                                                                        prev_model=self.prev_ddp_model,
                                                                        # Frozen copy of the model
                                                                        schedule_sampler=self.schedule_sampler,
                                                                        task_id=self.task_id,
                                                                        n_examples_per_task=self.batch_size,
                                                                        shape=shape,
                                                                        batch_size=self.microbatch)
            else:
                prev_loss = 0

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
                dae_only=isinstance(self.schedule_sampler, DAEOnlySampler)
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean() + prev_loss
            losses["prev_kl"] = prev_loss
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self, task_id):
        def save_checkpoint(rate, state_dict, suffix=""):
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate} {suffix}...")
                if not rate:
                    filename = f"model{(self.step + self.resume_step):06d}_{task_id}{suffix}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step + self.resume_step):06d}_{task_id}{suffix}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        if self.params.model_name == "UNetModel":
            state_dict = self.mp_trainer.master_params_to_state_dict(self.mp_trainer.master_params)
            save_checkpoint(0, state_dict)
        else:
            state_dict_1, state_dict_2 = self.mp_trainer.master_params_to_state_dict_DAE(self.mp_trainer.master_params)
            save_checkpoint(0, state_dict_1, suffix="_part_1")
            if state_dict_2 is not None:
                save_checkpoint(0, state_dict_2, suffix="_part_2")

        # for rate, params in zip(self.ema_rate, self.ema_params):
        #     save_checkpoint(rate, params)

        # if dist.get_rank() == 0:
        #     with bf.BlobFile(
        #             bf.join(get_blob_logdir(), f"opt{(self.step + self.resume_step):06d}.pt"),
        #             "wb",
        #     ) as f:
        #         th.save(self.opt.state_dict(), f)

        dist.barrier()

    @th.no_grad()
    def generate_examples(self, task_id, n_examples_per_task, batch_size=-1, only_one_task=False):
        if not only_one_task:
            total_num_exapmles = n_examples_per_task * (task_id + 1)
        else:
            total_num_exapmles = n_examples_per_task
        if batch_size == -1:
            batch_size = total_num_exapmles
        model = self.mp_trainer.model
        model.eval()
        all_images = []
        model_kwargs = {}
        if self.class_cond:  ### @TODO add option for class conditioning not task conditioning
            if only_one_task:
                tasks = th.zeros(n_examples_per_task, device=dist_util.dev()) + task_id
            else:
                tasks = th.tensor((list(range(task_id + 1)) * (n_examples_per_task)), device=dist_util.dev()).sort()[0]
        else:
            tasks = None
        i = 0
        while len(all_images) < total_num_exapmles:

            num_examples_to_generate = min(batch_size, total_num_exapmles - len(all_images))
            if self.class_cond:
                model_kwargs["y"] = tasks[i * batch_size:i * batch_size + num_examples_to_generate]
            sample_fn = (
                self.diffusion.p_sample_loop  # if not self.use_ddim else diffusion.ddim_sample_loop
            )
            sample = sample_fn(
                model,
                (num_examples_to_generate, self.in_channels, self.image_size, self.image_size),
                clip_denoised=False, model_kwargs=model_kwargs,
            )
            all_images.extend(sample.cpu())
            print(f"generated: {len(all_images)}/{total_num_exapmles}")
            i += 1
        model.train()
        all_images = th.stack(all_images, 0)
        return all_images, tasks

    @th.no_grad()
    def plot(self, task_id, step, num_exammples=8):
        sample, _ = self.generate_examples(task_id, num_exammples)
        samples_grid = make_grid(sample.detach().cpu(), num_exammples, normalize=True).permute(1, 2, 0)
        sample_wandb = wandb.Image(samples_grid.permute(2, 0, 1), caption=f"sample_task_{task_id}")
        logs = {"sampled_images": sample_wandb}

        plt.imshow(samples_grid)
        plt.axis('off')
        if not os.path.exists(os.path.join(logger.get_dir(), f"samples/")):
            os.makedirs(os.path.join(logger.get_dir(), f"samples/"))
        out_plot = os.path.join(logger.get_dir(), f"samples/task_{task_id:02d}_step_{step:06d}")
        plt.savefig(out_plot)
        if logger.get_rank_without_mpi_import() == 0:
            wandb.log(logs, step=step)

    def snr_plots(self, batch, cond, task_id, step):
        logs = {}
        snr_fwd_fl = []
        snr_bwd_fl = []
        kl_fl = []
        num_examples = 50
        for task in range(task_id + 1):
            if self.class_cond:
                id_curr = th.where(cond['y'] == task)[0][:num_examples]
                batch = batch[id_curr]
            batch = batch.to(dist_util.dev())
            num_examples = batch.shape[0]
            task_tsr = th.tensor([task] * num_examples, device=dist_util.dev())

            snr_fwd, snr_bwd, kl, x_q, x_p = self.get_snr_encode(batch,
                                                                 task_tsr,
                                                                 save_x=self.params.num_points_plot)
            x_p_q = th.cat([
                th.cat([x_q[j::self.params.num_points_plot],
                        x_p[j::self.params.num_points_plot]
                        ]) for j in range(self.params.num_points_plot)])
            x_p_q_vis = make_grid(x_p_q.detach().cpu(),
                                  x_q.shape[0] // self.params.num_points_plot,
                                  normalize=True, scale_each=True)
            logs[f"plot/x_{task}"] = wandb.Image(x_p_q_vis)
            kl_fl.append(kl.cpu())
            snr_bwd_fl.append(snr_bwd.reshape(snr_bwd.shape[0], num_examples, -1).detach().cpu().mean(-1))
            snr_fwd_fl.append(snr_fwd.reshape(snr_fwd.shape[0], num_examples, -1).detach().cpu().mean(-1))
        logs["plot/snr_encode"] = self.draw_snr_plot(snr_bwd_fl, snr_fwd_fl, log_scale=True)
        logs["plot/snr_encode_linear"] = self.draw_snr_plot(snr_bwd_fl, snr_fwd_fl, log_scale=False)
        # save the averages
        th.save(th.stack([s.mean(1) for s in snr_bwd_fl], 0),
                os.path.join(wandb.run.dir, f'bwd_snr_step_{step}.npy'))

        fig, axes = plt.subplots(ncols=task_id + 1, nrows=1, figsize=(5 * (task_id + 1), 4),
                                 sharey=True, constrained_layout=True)
        if task_id == 0:
            axes = np.expand_dims(axes, 0)
        for task in range(task_id + 1):
            time_hist(axes[task], kl_fl[task])
            axes[task].set_xlabel('T')
            axes[task].set_title(f'KL (task {task})')
            axes[task].grid(True)
        logs["plot/kl"] = wandb.Image(fig)
        if logger.get_rank_without_mpi_import() == 0:
            wandb.log(logs, step=step)

    @th.no_grad()
    def draw_final_snr_plot(self, step, task_id):
        av_snrs = []
        save_steps = list(range(0, step + 1, self.params.snr_log_interval))
        if save_steps[-1] < step:
            save_steps.append(step)
        for s in save_steps:
            # N_tasks x T each
            av_snrs.append(th.load(os.path.join(wandb.run.dir, f'bwd_snr_step_{s}.npy')))

        # linear
        cmap = plt.get_cmap('RdYlGn', len(av_snrs))
        fig, axes = plt.subplots(ncols=task_id + 1, nrows=1, figsize=(5 * (task_id + 1), 4),
                                 sharey=True, constrained_layout=True)
        if task_id == 0:
            axes = np.expand_dims(axes, 0)
        for task in range(task_id + 1):
            snr_to_plot = [s[task] for s in av_snrs]
            for i in range(len(snr_to_plot)):
                axes[task].plot(snr_to_plot[i], c=cmap(i))
            axes[task].grid(True)
        # Normalizer
        norm = mpl.colors.Normalize(vmin=0, vmax=len(av_snrs) - 1)

        # creating ScalarMappable
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ticks=np.linspace(0, len(av_snrs) - 1, len(av_snrs)))
        cbar.ax.set_yticklabels(save_steps)
        logs = {"plot/final_snr_linear": wandb.Image(fig)}
        # log scale
        fig, axes = plt.subplots(ncols=task_id + 1, nrows=1, figsize=(5 * (task_id + 1), 4),
                                 sharey=True, constrained_layout=True)
        if task_id == 0:
            axes = np.expand_dims(axes, 0)
        for task in range(task_id + 1):
            snr_to_plot = [s[task] for s in av_snrs]
            for i in range(len(snr_to_plot)):
                axes[task].plot(th.log(snr_to_plot[i]), c=cmap(i))
            axes[task].grid(True)
        # Normalizer
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ticks=np.linspace(0, len(av_snrs) - 1, len(av_snrs)))
        cbar.ax.set_yticklabels(save_steps)
        logs["plot/final_snr_log"] = wandb.Image(fig)
        if logger.get_rank_without_mpi_import() == 0:
            wandb.log(logs, step=step)

    @th.no_grad()
    def draw_snr_plot(self, snr_bwd_fl, snr_fwd_fl, log_scale=True):
        n_task = len(snr_bwd_fl)
        fig, axes = plt.subplots(ncols=n_task, nrows=2, figsize=(5 * n_task, 8),
                                 sharey=True, sharex=True, constrained_layout=True)
        if n_task == 1:
            axes = np.expand_dims(axes, 0)
        for task in range(n_task):
            fwd = snr_fwd_fl[task]
            bwd = snr_bwd_fl[task]
            title = f'SNR (task {task})'
            if log_scale:
                title = 'log ' + title
                fwd = th.log(fwd)
                bwd = th.log(bwd)
            time_hist(axes[task, 0], fwd)
            axes[task, 0].plot(fwd.mean(1))
            time_hist(axes[task, 1], bwd)
            axes[task, 1].plot(bwd.mean(1))
            axes[task, 0].set_xlabel('T')
            axes[task, 1].set_xlabel('T')
            axes[task, 0].set_ylabel(title)
            axes[task, 0].grid(True)
            axes[task, 1].grid(True)

        axes[0, 0].set_title(f'Forward diffusion ({snr_fwd_fl[task].shape[1]} points)',
                             fontsize=20)
        axes[0, 1].set_title(f'Backward diffusion ({snr_fwd_fl[task].shape[1]} points)',
                             fontsize=20);
        return wandb.Image(fig)

    @th.no_grad()
    def get_snr_encode(self, x, task_id, save_x):
        model = self.mp_trainer.model
        model.eval()
        model_kwargs = {}
        if self.class_cond:
            model_kwargs = {
                "y": th.zeros(task_id.shape[0], device=dist_util.dev()) + task_id
            }
        indices_fwd = list(range(self.diffusion.num_timesteps))  # [0, 1, ....T]
        shape = x.shape
        x_curr = x.clone()
        x_q = []
        x_p = []
        snr_fwd = []
        snr_bwd = []
        kl = []
        for i in indices_fwd:
            t = th.tensor([i] * shape[0], device=dist_util.dev())
            # get q(x_{i} | x_{i-1})
            mu = _extract_into_tensor(np.sqrt(1.0 - self.diffusion.betas), t, shape) * x_curr
            var = _extract_into_tensor(self.diffusion.betas, t, shape)
            snr_fwd.append(mu ** 2 / var)
            # sample x_{i}
            noise = th.randn_like(x_curr)
            x_curr = mu + (var ** 0.5) * noise
            x_q.append(x_curr[:save_x])
            # get p(x_{i-1} | x_{i})
            p_out = self.diffusion.p_mean_variance(
                model, x_curr, t, clip_denoised=False, model_kwargs=model_kwargs
            )
            x_p.append(p_out['mean'][:save_x] + (p_out['variance'][:save_x] ** 0.5) * th.randn_like(x_curr[:save_x]))
            snr_bwd.append(p_out['mean'] ** 2 / p_out['variance'])
            if i > 0:
                # get q(x_{i-1} | x_{i}, x_0) - posterior
                true_mean, true_var, true_log_variance_clipped = self.diffusion.q_posterior_mean_variance(
                    x_start=x, x_t=x_curr, t=t
                )
                kl.append(mean_flat(normal_kl(
                    true_mean, true_log_variance_clipped, p_out["mean"], p_out["log_variance"]
                )))
        return th.stack(snr_fwd), th.stack(snr_bwd), th.stack(kl), th.cat(x_q), th.cat(x_p)

    @th.no_grad()
    def plot_dae_only(self, task_id, step, num_exammples=8):
        test_loader = self.validator.dataloaders[self.task_id]
        diffs = []
        i = 0
        t = th.tensor(0, device=dist_util.dev())
        self.model.eval()
        batch, cond = next(iter(test_loader))
        batch = batch.to(dist_util.dev())
        x_t = self.diffusion.q_sample(batch, t)
        t = th.tensor([0] * x_t.shape[0], device=x_t.device)
        with th.no_grad():
            out = self.diffusion.p_sample(
                self.model,
                x_t,
                t,
                clip_denoised=False,
            )
            img = out["sample"]
        self.model.train()
        to_plot = th.cat([batch[:num_exammples], x_t[:num_exammples], img[:num_exammples]])
        to_plot = th.clamp(to_plot, -1, 1)
        samples_grid = make_grid(to_plot.detach().cpu(), num_exammples, normalize=True).permute(1, 2, 0)
        sample_wandb = wandb.Image(samples_grid.permute(2, 0, 1), caption=f"sample_task_{task_id}")
        if logger.get_rank_without_mpi_import() == 0:
            wandb.log({"sampled_images": sample_wandb})

        plt.imshow(samples_grid)
        plt.axis('off')
        if not os.path.exists(os.path.join(logger.get_dir(), f"samples/")):
            os.makedirs(os.path.join(logger.get_dir(), f"samples/"))
        out_plot = os.path.join(logger.get_dir(), f"samples/task_{task_id:02d}_step_{step:06d}")
        plt.savefig(out_plot)

    def validate_dae(self):
        test_loader = self.validator.dataloaders[self.task_id]
        diffs = []
        i = 0
        t = th.tensor(0, device=dist_util.dev())
        self.model.eval()
        for batch, cond in test_loader:
            batch = batch.to(dist_util.dev())
            x_t = self.diffusion.q_sample(batch, t)
            t = th.tensor([0] * x_t.shape[0], device=x_t.device)
            with th.no_grad():
                out = self.diffusion.p_sample(
                    self.model,
                    x_t,
                    t,
                    clip_denoised=False,
                )
                img = out["sample"]
            diff = th.abs(batch - img).mean().item()
            diffs.append(diff)
            if i > 100:
                break
            i += 1
        self.model.train()
        return np.mean(diffs)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        if key != "prev_kl":
            # Log the quantiles (four quartiles, in particular).
            for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
                quartile = int(4 * sub_t / diffusion.num_timesteps)
                logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
                if sub_t == 0:
                    logger.logkv_mean(f"{key}_0_step", sub_loss)


def time_hist(ax, data):
    num_pt, num_ts = data.shape
    num_fine = num_pt * 10
    x_fine = np.linspace(0, num_pt, num_fine)
    y_fine = np.empty((num_ts, num_fine), dtype=float)
    for i in range(num_ts):
        y_fine[i, :] = np.interp(x_fine, range(num_pt), data[:, i])
    y_fine = y_fine.flatten()
    x_fine = np.matlib.repmat(x_fine, num_ts, 1).flatten()
    cmap = copy.copy(plt.cm.BuPu)
    cmap.set_bad(cmap(0))
    h, xedges, yedges = np.histogram2d(x_fine, y_fine, bins=[40, 1000])
    pcm = ax.pcolormesh(xedges, yedges, h.T, cmap=cmap, vmax=num_ts, rasterized=True)
    plt.colorbar(pcm, ax=ax, label="# points", pad=0);
