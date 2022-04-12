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

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler, TaskAwareSampler
from .gaussian_diffusion import _extract_into_tensor
from .nn import mean_flat
from .losses import normal_kl

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
            generate_previous_samples_continuously=False
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
        self.global_batch = self.batch_size * dist.get_world_size()

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

        if th.cuda.is_available():
            self.use_ddp = True
            find_unused_params = isinstance(self.model, TwoPartsUNetModel)
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=find_unused_params,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model
        self.generate_previous_examples_at_start_of_new_task = generate_previous_examples_at_start_of_new_task
        self.generate_previous_samples_continuously = generate_previous_samples_continuously

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
        while (
                (not self.lr_anneal_steps
                 or self.step + self.resume_step < self.lr_anneal_steps) and (self.step < self.num_steps)
        ):
            if self.step > 100:
                self.mp_trainer.skip_gradient_thr = self.params.skip_gradient_thr
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                wandb.log(logger.getkvs())
                logger.dumpkvs()
            if (not self.skip_save) & (self.step % self.save_interval == 0):
                self.save(self.task_id)
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            if self.step > 0:
                if self.step % self.plot_interval == 0:
                    self.plot(self.task_id, self.step)
                    if self.params.log_snr:
                        self.snr_plots(batch, cond, self.task_id, self.step)
                if self.step % self.scheduler_step == 0:
                    self.scheduler.step()
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if not self.skip_save:
            if (self.step - 1) % self.save_interval != 0:
                self.save(self.task_id)
        if (self.step - 1) % self.plot_interval != 0:
            self.plot(self.task_id, self.step)
            if self.params.log_snr:
                self.snr_plots(batch, cond, self.task_id, self.step)

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
                                                                        prev_model=self.prev_ddp_model, #Frozen copy of the model
                                                                        schedule_sampler= self.schedule_sampler,
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
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step + self.resume_step):06d}_{task_id}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step + self.resume_step):06d}_{task_id}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                    bf.join(get_blob_logdir(), f"opt{(self.step + self.resume_step):06d}.pt"),
                    "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

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
            i += 1
        model.train()
        all_images = th.stack(all_images, 0)
        return all_images, tasks

    @th.no_grad()
    def plot(self, task_id, step, num_examples=4):
        sample, _ = self.generate_examples(task_id, num_examples)
        samples_grid = make_grid(sample.detach().cpu(), num_examples, normalize=True).permute(1, 2, 0)
        sample_wandb = wandb.Image(samples_grid.permute(2, 0, 1), caption=f"sample_task_{task_id}")
        logs = {"sampled_images": sample_wandb}

        plt.imshow(samples_grid)
        plt.axis('off')
        if not os.path.exists(os.path.join(logger.get_dir(), f"samples/")):
            os.makedirs(os.path.join(logger.get_dir(), f"samples/"))
        out_plot = os.path.join(logger.get_dir(), f"samples/task_{task_id:02d}_step_{step:06d}")
        plt.savefig(out_plot)
        wandb.log(logs, step=step)

    def snr_plots(self, batch, cond, task_id, step):
        logs = {}
        snr_fwd_fl = []
        snr_bwd_fl = []
        kl_fl = []
        num_examples = 40
        for task in range(task_id+1):
            id_curr = th.where(cond['y'] == task)[0][:num_examples]
            x = batch[id_curr]
            x = x.to(dist_util.dev())
            task_tsr = th.tensor([task]*num_examples, device=dist_util.dev())
            N_pts = 2
            snr_fwd, snr_bwd, kl, x_q, x_p = self.get_snr_encode(x, task_tsr, save_x=N_pts)
            x_p_q = th.cat([th.cat([x_q[j::N_pts], x_p[j::N_pts]]) for j in range(N_pts)])
            x_p_q_vis = make_grid(x_p_q.detach().cpu(), x_q.shape[0]//N_pts,
                                  normalize=True, scale_each=True)
            logs[f"plot/x_{task}"] = wandb.Image(x_p_q_vis)
            kl_fl.append(kl.cpu())
            snr_bwd_fl.append(snr_bwd.reshape(snr_bwd.shape[0], num_examples, -1).detach().cpu().mean(-1))
            snr_fwd_fl.append(snr_fwd.reshape(snr_fwd.shape[0], num_examples, -1).detach().cpu().mean(-1))
        logs["plot/snr_encode"] = self.draw_snr_plot(snr_bwd_fl, snr_fwd_fl, log_scale=True)
        logs["plot/snr_encode_linear"] = self.draw_snr_plot(snr_bwd_fl, snr_fwd_fl, log_scale=False)
        fig, axes = plt.subplots(ncols=task_id+1, nrows=1, figsize=(5*(task_id+1), 4),
                                 sharey=True, constrained_layout=True)
        if task_id == 0:
            axes = np.expand_dims(axes, 0)
        for task in range(task_id+1):
            time_hist(axes[task], kl_fl[task])
            axes[task].set_xlabel('T')
            axes[task].set_title(f'KL (task {task})')
            axes[task].grid(True)
        logs["plot/kl"] = wandb.Image(fig)
        wandb.log(logs, step=step)

    @th.no_grad()
    def draw_snr_plot(self, snr_bwd_fl, snr_fwd_fl, log_scale=True):
        n_task = len(snr_bwd_fl)
        fig, axes = plt.subplots(ncols=n_task, nrows=2, figsize=(5*n_task, 8),
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
            snr_fwd.append(mu**2 / var)
            # sample x_{i}
            noise = th.randn_like(x_curr)
            x_curr = mu + (var ** 0.5) * noise
            x_q.append(x_curr[:save_x])
            # get p(x_{i-1} | x_{i})
            p_out = self.diffusion.p_mean_variance(
                model, x_curr, t, clip_denoised=False, model_kwargs=model_kwargs
            )
            x_p.append(p_out['mean'][:save_x] + (p_out['variance'][:save_x] ** 0.5) * th.randn_like(x_curr[:save_x]))
            snr_bwd.append(p_out['mean']**2 / p_out['variance'])
            if i > 0:
                # get q(x_{i-1} | x_{i}, x_0) - posterior
                true_mean, true_var, true_log_variance_clipped = self.diffusion.q_posterior_mean_variance(
                    x_start=x, x_t=x_curr, t=t
                )
                kl.append(mean_flat(normal_kl(
                    true_mean, true_log_variance_clipped, p_out["mean"], p_out["log_variance"]
                    )))
        return th.stack(snr_fwd), th.stack(snr_bwd), th.stack(kl), th.cat(x_q), th.cat(x_p)


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
        if key!="prev_kl":
            # Log the quantiles (four quartiles, in particular).
            for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
                quartile = int(4 * sub_t / diffusion.num_timesteps)
                logger.logkv_mean(f"{key}_q{quartile}", sub_loss)


def time_hist(ax, data):
    num_pt, num_ts = data.shape
    num_fine = num_pt*10
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