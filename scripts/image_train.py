"""
Train a diffusion model on images.
"""

import argparse
import copy
from collections import OrderedDict

from dataloaders import base
from dataloaders.datasetGen import *
from evaluations.validation import Validator, CERN_Validator

from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser, results_to_log,
)
from torch.utils import data
import torch.distributed as dist
from guided_diffusion.train_util import TrainLoop
from guided_diffusion import logger
import os

if os.uname().nodename == "titan4":
    from guided_diffusion import dist_util_titan as dist_util
elif os.uname().nodename == "node7001.grid4cern.if.pw.edu.pl":
    from guided_diffusion import dist_util_dwarf as dist_util
else:
    from guided_diffusion import dist_util

import torch


# os.environ["WANDB_MODE"] = "disabled"

def yielder(loader):
    while True:
        yield from loader


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist(args)

    if args.seed is not None:
        print("Using manual seed = {}".format(args.seed))
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print("WARNING: Not using manual seed - your experiments will not be reproducible")

    os.environ["OPENAI_LOGDIR"] = f"results/{args.experiment_name}"
    logger.configure()

    if logger.get_rank_without_mpi_import() == 0:
        import wandb
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
        wandb.init(project="ddgm_representations", name=args.experiment_name, config=args, entity="kdeja")

    if args.img_size:
        args.img_size = int(args.img_size)
        train_dataset, val_dataset, image_size, image_channels = base.__dict__[args.dataset](args.dataroot,
                                                                                             train_aug=args.train_aug,
                                                                                             resolution=args.img_size)
    else:
        train_dataset, val_dataset, image_size, image_channels = base.__dict__[args.dataset](args.dataroot,
                                                                                             train_aug=args.train_aug)

    print(f"Training with image size:{image_size}")

    args.image_size = image_size
    args.in_channels = image_channels
    if args.dataset.lower() in ["celeba"]:
        n_classes = 10
    else:
        n_classes = train_dataset.number_classes

    args.num_classes = n_classes #args.num_tasks
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()), dae_only=args.schedule_sampler == "only_dae"
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion, args)

    logger.log("creating data loader...")

    train_dataset_splits, val_dataset_splits, task_output_space = data_split(dataset=train_dataset,
                                                                             return_classes=args.class_cond,
                                                                             return_task_as_class=False,
                                                                             dataset_name=args.dataset.lower(),
                                                                             num_batches=args.num_tasks,
                                                                             num_classes=n_classes,
                                                                             random_split=args.random_split,
                                                                             limit_data=args.limit_data,
                                                                             dirichlet_split_alpha=args.dirichlet,
                                                                             reverse=args.reverse,
                                                                             limit_classes=args.limit_classes,
                                                                             val_size=0 if args.validate_on_train else 0.3)

    if not args.skip_validation:
        val_loaders = []
        for task_id in range(args.num_tasks):
            if not args.validate_on_train:
                val_data = val_dataset_splits[task_id]
            else:
                val_data = train_dataset_splits[task_id]
            val_loader = data.DataLoader(dataset=val_data,
                                         batch_size=args.microbatch if args.microbatch > 0 else args.batch_size,
                                         shuffle=args.schedule_sampler == "only_dae", drop_last=True)
            val_loaders.append(val_loader)

        stats_file_name = f"seed_{args.seed}_tasks_{args.num_tasks}_random_{args.random_split}_dirichlet_{args.dirichlet}_limit_{args.limit_data}"
        if args.use_gpu_for_validation:
            device_for_validation = dist_util.dev()
        else:
            device_for_validation = torch.device("cpu")
        if args.dataset.lower() != "cern":
            validator = Validator(n_classes=n_classes, device=dist_util.dev(), dataset=args.dataset,
                                  stats_file_name=stats_file_name,
                                  score_model_device=device_for_validation, dataloaders=val_loaders)
        else:
            raise NotImplementedError()  # Adapt CERN validator
            # validator = CERN_Validator(dataloaders=val_loaders, stats_file_name=stats_file_name, device=dist_util.dev())
    else:
        validator = None

    fid_table = OrderedDict()
    precision_table = OrderedDict()
    recall_table = OrderedDict()
    for task_id in range(args.num_tasks):
        if task_id == 0:
            train_dataset_loader = data.DataLoader(dataset=train_dataset_splits[task_id],
                                                   batch_size=args.batch_size, shuffle=True,
                                                   drop_last=True, num_workers=32)
            dataset_yielder = yielder(train_dataset_loader)
        elif not args.generate_previous_examples_at_start_of_new_task:
            train_dataset_loader = data.DataLoader(dataset=train_dataset_splits[task_id],
                                                   batch_size=args.batch_size // (task_id + 1), shuffle=True,
                                                   drop_last=True, num_workers=32)
            dataset_yielder = yielder(train_dataset_loader)
        else:
            print("Preparing dataset for rehearsal...")
            if args.n_generated_examples_per_task <= args.batch_size:
                batch_size = args.n_generated_examples_per_task
            else:
                batch_size = args.batch_size
            generated_previous_examples, generated_previous_examples_tasks = train_loop.generate_examples(task_id - 1,
                                                                                                          args.n_generated_examples_per_task,
                                                                                                          batch_size=batch_size)
            generated_dataset = AppendName(
                data.TensorDataset(generated_previous_examples, generated_previous_examples_tasks),
                generated_previous_examples_tasks.cpu().numpy(), args.class_cond, args.use_task_index)
            joined_dataset = data.ConcatDataset([train_dataset_splits[task_id], generated_dataset])
            train_dataset_loader = data.DataLoader(dataset=joined_dataset,
                                                   batch_size=args.batch_size, shuffle=True,
                                                   drop_last=True)
            dataset_yielder = yielder(train_dataset_loader)
            print("Done")

        if args.class_cond:
            if args.use_task_index:
                max_class = task_id
            else:
                max_class = n_classes
                # raise NotImplementedError()  # Classes seen so far for plotting and sampling
        else:
            max_class = None
        logger.log(f"training task {task_id}")
        if task_id == 0:
            num_steps = args.first_task_num_steps
        else:
            num_steps = args.num_steps
        train_loop = TrainLoop(
            params=args,
            model=model,
            prev_model=copy.deepcopy(model),
            diffusion=diffusion,
            task_id=task_id,
            data=dataset_yielder,
            batch_size=args.batch_size,
            microbatch=args.microbatch,
            lr=args.lr,
            scheduler_rate=args.scheduler_rate,
            ema_rate=args.ema_rate,
            log_interval=args.log_interval,
            skip_save=args.skip_save,
            save_interval=args.save_interval,
            plot_interval=args.plot_interval,
            resume_checkpoint=args.resume_checkpoint,
            use_fp16=args.use_fp16,
            fp16_scale_growth=args.fp16_scale_growth,
            schedule_sampler=schedule_sampler,
            weight_decay=args.weight_decay,
            lr_anneal_steps=args.lr_anneal_steps,
            num_steps=num_steps,
            image_size=args.image_size,
            in_channels=args.in_channels,
            class_cond=args.class_cond,
            max_class=max_class,
            generate_previous_examples_at_start_of_new_task=args.generate_previous_examples_at_start_of_new_task,
            generate_previous_samples_continuously=args.generate_previous_samples_continuously,
            validator=validator,
            validation_interval=args.validation_interval
        )
        if args.load_model_path is not None:
            if args.model_name == "UNetModel":
                model.load_state_dict(
                    dist_util.load_state_dict(args.load_model_path + ".pt", map_location="cpu")
                )
            else:
                model.unet_1.load_state_dict(
                    dist_util.load_state_dict(args.load_model_path + "_part_1.pt", map_location="cpu")
                )
                model.unet_2.load_state_dict(
                    dist_util.load_state_dict(args.load_model_path + "_part_2.pt", map_location="cpu")
                )
            model.to(dist_util.dev())
            train_loop.step = int(args.load_model_path[-8:-2])
        train_loop.run_loop()

        if args.schedule_sampler == "only_dae":
            train_loop.plot_dae_only(task_id, step=0)
        else:
            train_loop.plot(task_id, step=0)
        fid_table[task_id] = OrderedDict()
        precision_table[task_id] = OrderedDict()
        recall_table[task_id] = OrderedDict()
        if args.skip_validation or (args.schedule_sampler == "only_dae"):
            for j in range(task_id + 1):
                fid_table[j][task_id] = -1
                precision_table[j][task_id] = -1
                recall_table[j][task_id] = -1
        else:
            print("Validation")
            for j in range(task_id + 1):
                fid_result, precision, recall = validator.calculate_results(train_loop=train_loop,
                                                                            task_id=j,
                                                                            dataset=args.dataset,
                                                                            n_generated_examples=args.n_examples_validation,
                                                                            batch_size=args.microbatch if args.microbatch > 0 else args.batch_size)
                fid_table[j][task_id] = fid_result
                precision_table[j][task_id] = precision
                recall_table[j][task_id] = recall
                print(f"FID task {j}: {fid_result}")
            results_to_log(fid_table, precision_table, recall_table)
    print(fid_table)


def create_argparser():
    defaults = dict(
        seed=None,
        wandb_api_key="",
        experiment_name="test",
        dataroot="data/",
        dataset="MNIST",
        schedule_sampler="uniform",
        alpha=4,
        beta=1.2,
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=32,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=50,
        snr_log_interval=-1,
        num_points_plot=2,
        skip_save=False,
        save_interval=20000,
        plot_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        gpu_id=-1,
        reverse=False,
        dirichlet=None,
        num_tasks=1,
        limit_classes=-1,
        random_split=False,
        train_aug=False,
        limit_data=None,
        num_steps=10000,
        scheduler_rate=1.0,
        use_task_index=True,
        skip_validation=False,
        n_examples_validation=5000,
        validation_interval=25000,
        validate_on_train=True,
        use_gpu_for_validation=True,
        n_generated_examples_per_task=1000,
        first_task_num_steps=5000,
        skip_gradient_thr=-1,
        generate_previous_examples_at_start_of_new_task=False,
        generate_previous_samples_continuously=True,
        load_model_path=None,
        load_pretrained_for_dae=False,
        img_size=None
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
