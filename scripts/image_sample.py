"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import logger

if os.uname().nodename == "titan4":
    from guided_diffusion import dist_util_titan as dist_util
elif os.uname().nodename == "node7001.grid4cern.if.pw.edu.pl":
    from guided_diffusion import dist_util_dwarf as dist_util
else:
    from guided_diffusion import dist_util

from guided_diffusion.script_util import (
    # NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


def main():
    args = create_argparser().parse_args()
    args.num_classes = args.num_tasks
    os.environ["OPENAI_LOGDIR"] = f"results/{args.experiment_name}"
    args.model_path = f"results/{args.experiment_name}/" + args.model_path


    dist_util.setup_dist(args)
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    if args.model_name == "UNetModel":
        model.load_state_dict(
            dist_util.load_state_dict(args.model_path, map_location="cpu")
        )
    else:
        model.unet_1.load_state_dict(
            dist_util.load_state_dict(args.model_path + "_part_1.pt", map_location="cpu")
        )
        model.unet_2.load_state_dict(
            dist_util.load_state_dict(args.model_path + "_part_2.pt", map_location="cpu")
        )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            # classes = th.randint(
            #     low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            # )
            classes = th.zeros(size=(args.batch_size,), device=dist_util.dev())
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, args.in_channels, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

        arr = np.concatenate(all_images, axis=0)
        arr = arr[: args.num_samples]
        if args.class_cond:
            label_arr = np.concatenate(all_labels, axis=0)
            label_arr = label_arr[: args.num_samples]
        if dist.get_rank() == 0:
            shape_str = "x".join([str(x) for x in arr.shape])
            out_path = os.path.join(logger.get_dir(), f"samples_{args.model_path.split('/')[-1][:-3]}.npz")
            logger.log(f"saving to {out_path}")
            if args.class_cond:
                np.savez(out_path, arr, label_arr)
            else:
                np.savez(out_path, arr)

    dist.barrier()
    plt.figure()
    plt.axis('off')
    samples_grid = make_grid(th.from_numpy((arr.swapaxes(1, 3))), 4).permute(2, 1, 0)
    plt.imshow(samples_grid)
    out_plot = os.path.join(logger.get_dir(), f"samples_{args.model_path.split('/')[-1][:-3]}")
    plt.savefig(out_plot)
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        experiment_name="test",
        clip_denoised=False,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        gpu_id=-1,
        num_tasks=1,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
