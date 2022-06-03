# DAED

This is the codebase for [On Analyzing Generative and Denoising Capabilities
of Diffusion-based Deep Generative Models](https://arxiv.org/abs/2206.00070).

This repository is based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion), with modifications to analyze denoising and generative capabilities of DDGM

# Training models

Training diffusion models is based on the original training from [openai repository](https://github.com/openai/improved-diffusion) with few modifications

```
mpiexec -n N python scripts/classifier_train.py --dataroot data/ --experiment_name experiment_name $TRAIN_FLAGS $CLASSIFIER_FLAGS
```

All generated artifacts will be stored in `results/$experiment_name` Make sure to include external data (such as ImageNet) in `dataroot`, for standard benchmarks in torchvision it will be downloaded automatically
Make sure to divide the batch size in `TRAIN_FLAGS` by the number of MPI processes you are using.

Additional flags for DAED training:
- `--first_step_beta` - amount of noise added in the first diffusion step
- `--noise_schedule` - specify as `linear_combine` to use a noise scheduler with changed first beta value
- `--schedule_sampler` - specify how timesteps are sampled for training. Use `uniform_dae` to sample steps according to the amount of noise, or `dae_only` to train DAE part of DAED only
- `--model_name` - `UNetModel` by default, to train as DAED switch to `TwoPartsUNetModelDAE`

For sampling and evaluation, follow instructions in [openai repository](https://github.com/openai/improved-diffusion)

##Training hyperparameters to reproduce results from paper:

FashionMNIST Simple DAED with beta<sub>1</sub> = 0.1
```
python3 -m scripts.image_train --experiment_name FashionMNIST_dae_468_200k_simple --dataset FashionMNIST --num_channels 64 
--num_res_blocks 3 --learn_sigma False --dropout 0.3 --diffusion_steps 468 
--noise_schedule linear_combine --first_step_beta 0.1 --use_kl False --schedule_sampler dae_uniform 
--lr 1e-4 --batch_size 128 --first_task_num_steps 100000 --model_name TwoPartsUNetModelDAE
```

FashionMNIST Simple DAED with beta<sub>1</sub> = 0.001
```
python3 -m scripts.image_train --experiment_name FashionMNIST_dae_500_200k_simple --dataset FashionMNIST --num_channels 64 
--num_res_blocks 3 --learn_sigma False --dropout 0.3 --diffusion_steps 500 --noise_schedule linear 
--use_kl False --lr 1e-4 --batch_size 128 --schedule_sampler uniform 
--first_task_num_steps 100000 --model_name TwoPartsUNetModelDAE
```

FashionMNIST VLB DAED with beta<sub>1</sub> = 0.001
```
python3 -m scripts.image_train --experiment_name FashionMNIST_dae_500_200k_vlb --dataset FashionMNIST --num_channels 64 
--num_res_blocks 3 --learn_sigma True --use_kl True  --dropout 0.3 --diffusion_steps 500 
--noise_schedule linear --lr 1e-4 --batch_size 128 --schedule_sampler uniform 
--first_task_num_steps 100000 --model_name TwoPartsUNetModelDAE
```

CIFAR10 Simple DAED with beta<sub>1</sub> = 0.1
```
python3 -m scripts.image_train --experiment_name CIFAR10_dae_900_200k_simple --dataset CIFAR10 --num_channels 128 
--num_res_blocks 3 --learn_sigma False --dropout 0.3 --diffusion_steps 900 
--noise_schedule linear_combine --first_step_beta 0.1 --use_kl False --schedule_sampler dae_uniform 
--lr 1e-4 --batch_size 160 --first_task_num_steps 500000 --model_name TwoPartsUNetModelDAE
```

CIFAR10 Simple DAED with beta<sub>1</sub> = 0.001
```
python3 -m scripts.image_train --experiment_name CIFAR10_dae_1000_200k_simple --dataset CIFAR10 --num_channels 128 
--num_res_blocks 3 --learn_sigma False --dropout 0.3 --diffusion_steps 1000 --noise_schedule linear 
--use_kl False --lr 1e-4 --batch_size 160 --schedule_sampler uniform 
--first_task_num_steps 500000 --model_name TwoPartsUNetModelDAE
```

CIFAR10 VLB DAED with beta<sub>1</sub> = 0.001
```
python3 -m scripts.image_train --experiment_name CIFAR_dae_1000_200k_vlb --dataset CIFAR10 --num_channels 128 
--num_res_blocks 3 --learn_sigma True --use_kl True  --dropout 0.3 --diffusion_steps 1000 
--noise_schedule linear --lr 1e-4 --batch_size 160 --schedule_sampler uniform 
--first_task_num_steps 500000 --model_name TwoPartsUNetModelDAE
```

CelebA Simple DAED with beta<sub>1</sub> = 0.1
```
python3 -m scripts.image_train --experiment_name CelebA_dae_900_200k_simple --dataset CelebA --num_channels 128 
--num_res_blocks 3 --learn_sigma False --dropout 0.3 --diffusion_steps 900 
--noise_schedule linear_combine --first_step_beta 0.1 --use_kl False --schedule_sampler dae_uniform 
--lr 1e-4 --batch_size 160 --first_task_num_steps 200000 --model_name TwoPartsUNetModelDAE
```

CelebA Simple DAED with beta<sub>1</sub> = 0.001
```
python3 -m scripts.image_train --experiment_name CelebA_dae_1000_200k_simple --dataset CelebA --num_channels 128 
--num_res_blocks 3 --learn_sigma False --dropout 0.3 --diffusion_steps 1000 --noise_schedule linear 
--use_kl False --lr 1e-4 --batch_size 160 --schedule_sampler uniform 
--first_task_num_steps 200000 --model_name TwoPartsUNetModelDAE
```

CelebA VLB DAED with beta<sub>1</sub> = 0.1
```
python3 -m scripts.image_train --experiment_name CelebA_dae_900_200k_vlb --dataset CelebA --num_channels 128 
--num_res_blocks 3 --learn_sigma True --use_kl True --dropout 0.3 --diffusion_steps 900 
--noise_schedule linear_combine --first_step_beta 0.1 --schedule_sampler dae_uniform 
--lr 1e-4 --batch_size 160 --first_task_num_steps 200000 --model_name TwoPartsUNetModelDAE
```

## Sampling with trained models

For example, for Celeba DAE with beta<sub>1</sub> = 0.1
```
python -m scripts.image_sample --experiment_name CelebA_dae_900_200k_simple --model_path model200000_0 
--image_size 64 --num_channels 128 --num_res_blocks 3 --learn_sigma False --dropout 0.3 --diffusion_steps 900 
--noise_schedule linear_combine --first_step_beta 0.1 --use_kl False --batch_size 100 --num_samples 1500 --num_tasks 1 
--model_name TwoPartsUNetModelDAE --in_channels 3
```

## Sampling with combined models

```
python3 -m scripts.image_sample_with_two_models --experiment_name sample_two_parts
--model_path pretrained_ddgm_path/model200000_0_part_2.pt --dae_path pretrained_dae_path/model020000_0_part_1.pt
 --image_size 64 --num_channels 128 --num_res_blocks 3 --learn_sigma_2 False --learn_sigma_1 False --diffusion_steps 1000 
--noise_schedule linear_combine --first_step_beta 0.1 --use_kl False --batch_size 1000 --num_samples 15000 
--in_channels 3
```

where `--diffusion_steps` is the original numbr of DDGM diffusion steps, that are automatically adjusted according to the `--frst_step_beta` parameter.
For compatibility with DDGM models from [openai repository](https://github.com/openai/improved-diffusion) use `--old_repo_compatible True` option
