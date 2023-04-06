# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Based on timm, DeiT, DINO, MoCo-v3, BEiT, MAE-priv and MAE code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# https://github.com/facebookresearch/moco-v3
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/BUPT-PRIV/MAE-priv
# https://github.com/facebookresearch/mae
# --------------------------------------------------------
import argparse
import datetime
import json
import math
import os
import sys
import time
import warnings
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms
import yaml
from einops import rearrange

import utils
import utils.data_constants as data_constants
from multimae import multimae
from multimae.criterion import (MaskedCrossEntropyLoss, MaskedL1Loss,
                                MaskedMSELoss)
from multimae.input_adapters import PatchedInputAdapter, SemSegInputAdapter
from multimae.output_adapters import SpatialOutputAdapter
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import create_model
from utils.data_constants import COCO_SEMSEG_NUM_CLASSES
from utils.datasets import build_multimae_pretraining_dataset
from utils.datasets import build_pretraining_dataset
from utils.optim_factory import create_optimizer
from utils.task_balancing import (NoWeightingStrategy,
                                  UncertaintyWeightingStrategy)
from utils.plot_utils import plot_predictions, plot_predictions_test



import torchvision.transforms.functional as TF
from utils.data_constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageOps
import seaborn as sns
import pdb
import pandas as pd

DOMAIN_CONF = {
    'Sat_RGB': {
        'channels': 3,
        'stride_level': 1,
        'input_adapter': partial(PatchedInputAdapter, num_channels=3),
        'output_adapter': partial(SpatialOutputAdapter, num_channels=3),
        'loss': MaskedMSELoss,
    },

    'UAV_RGB': {
        'channels': 3,
        'stride_level': 1,
        'input_adapter': partial(PatchedInputAdapter, num_channels=3),
        'output_adapter': partial(SpatialOutputAdapter, num_channels=3),
        'loss': MaskedMSELoss,
    },

    'depth': {
        'channels': 1,
        'stride_level': 1,
        'input_adapter': partial(PatchedInputAdapter, num_channels=1),
        'output_adapter': partial(SpatialOutputAdapter, num_channels=1),
        'loss': MaskedL1Loss,
    },
    'semseg': {
        'num_classes': 133,
        'stride_level': 4,
        'input_adapter': partial(SemSegInputAdapter, num_classes=COCO_SEMSEG_NUM_CLASSES,
                                 dim_class_emb=64, interpolate_class_emb=False),
        'output_adapter': partial(SpatialOutputAdapter, num_channels=COCO_SEMSEG_NUM_CLASSES),
        'loss': partial(MaskedCrossEntropyLoss, label_smoothing=0.0),
    },
}



def get_args():
    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')

    parser = argparse.ArgumentParser('MultiMAE pre-training script', add_help=False)

    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size per GPU (default: %(default)s)')
    parser.add_argument('--epochs', default=100, type=int,
                        help='Number of epochs (default: %(default)s)')
    parser.add_argument('--save_ckpt_freq', default=100, type=int,
                        help='Checkpoint saving frequency in epochs (default: %(default)s)')

    # Task parameters
    parser.add_argument('--in_domains', default='Sat_RGB-UAV_RGB', type=str,
                        help='Input domain names, separated by hyphen (default: %(default)s)')
    parser.add_argument('--out_domains', default='Sat_RGB-UAV_RGB', type=str,
                        help='Output domain names, separated by hyphen (default: %(default)s)')
    parser.add_argument('--standardize_depth',  action='store_true')
    parser.add_argument('--no_standardize_depth', action='store_false', dest='standardize_depth')
    parser.set_defaults(standardize_depth=False)
    parser.add_argument('--extra_norm_pix_loss', action='store_true')
    parser.add_argument('--no_extra_norm_pix_loss', action='store_false', dest='extra_norm_pix_loss')
    parser.set_defaults(extra_norm_pix_loss=True)


    # Model parameters
    parser.add_argument('--model', default='pretrain_multimae_base', type=str, metavar='MODEL',
                        help='Name of model to train (default: %(default)s)')
    parser.add_argument('--num_encoded_tokens', default=98, type=int,
                        help='Number of tokens to randomly choose for encoder (default: %(default)s)')
    parser.add_argument('--num_global_tokens', default=1, type=int,
                        help='Number of global tokens to add to encoder (default: %(default)s)')
    parser.add_argument('--patch_size', default=16, type=int,
                        help='Base patch size for image-like modalities (default: %(default)s)')
    parser.add_argument('--input_size', default=224, type=int,
                        help='Images input size for backbone (default: %(default)s)')
    parser.add_argument('--alphas', type=float, default=1.0, 
                        help='Dirichlet alphas concentration parameter (default: %(default)s)')
    parser.add_argument('--sample_tasks_uniformly', default=False, action='store_true',
                        help='Set to True/False to enable/disable uniform sampling over tasks to sample masks for.')

    parser.add_argument('--decoder_use_task_queries', default=True, action='store_true',
                        help='Set to True/False to enable/disable adding of task-specific tokens to decoder query tokens')
    parser.add_argument('--decoder_use_xattn', default=True, action='store_true',
                        help='Set to True/False to enable/disable decoder cross attention.')
    parser.add_argument('--decoder_dim', default=256, type=int,
                        help='Token dimension inside the decoder layers (default: %(default)s)')
    parser.add_argument('--decoder_depth', default=2, type=int,
                        help='Number of self-attention layers after the initial cross attention (default: %(default)s)')
    parser.add_argument('--decoder_num_heads', default=8, type=int,
                        help='Number of attention heads in decoder (default: %(default)s)')
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: %(default)s)')

    parser.add_argument('--loss_on_unmasked', default=False, action='store_true',
                        help='Set to True/False to enable/disable computing the loss on non-masked tokens')
    parser.add_argument('--no_loss_on_unmasked', action='store_false', dest='loss_on_unmasked')
    parser.set_defaults(loss_on_unmasked=False)


    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: %(default)s)')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer epsilon (default: %(default)s)')
    parser.add_argument('--opt_betas', default=[0.9, 0.95], type=float, nargs='+', metavar='BETA',
                        help='Optimizer betas (default: %(default)s)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='CLIPNORM',
                        help='Clip gradient norm (default: %(default)s)')
    parser.add_argument('--skip_grad', type=float, default=None, metavar='SKIPNORM',
                        help='Skip update if gradient norm larger than threshold (default: %(default)s)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: %(default)s)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay (default: %(default)s)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD.  (Set the same value as args.weight_decay to keep weight decay unchanged)""")
    parser.add_argument('--decoder_decay', type=float, default=None, help='decoder weight decay')

    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256 (default: %(default)s)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='Warmup learning rate (default: %(default)s)')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Lower lr bound for cyclic schedulers that hit 0 (default: %(default)s)')
    parser.add_argument('--task_balancer', type=str, default='none',
                        help='Task balancing scheme. One out of [uncertainty, none] (default: %(default)s)')
    parser.add_argument('--balancer_lr_scale', type=float, default=1.0,
                        help='Task loss balancer LR scale (if used) (default: %(default)s)')


    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='Epochs to warmup LR, if scheduler supports (default: %(default)s)')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='Epochs to warmup LR, if scheduler supports (default: %(default)s)')

    parser.add_argument('--fp32_output_adapters', type=str, help='Tasks output adapters to compute in fp32 mode, separated by hyphen.')

    # Augmentation parameters
    parser.add_argument('--hflip', type=float, default=0.5,
                        help='Probability of horizontal flip (default: %(default)s)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic) (default: %(default)s)')

    # Dataset parameters
    parser.add_argument('--data_path', default='Processed', type=str, help='dataset path')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')

    # Misc.
    parser.add_argument('--output_dir', default='',
                        help='Path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training / testing')

    parser.add_argument('--seed', default=0, type=int, help='Random seed ')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int) #10
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--find_unused_params', action='store_true')
    parser.add_argument('--no_find_unused_params', action='store_false', dest='find_unused_params')
    parser.set_defaults(find_unused_params=True)

    # Wandb logging
    parser.add_argument('--log_wandb', default=False, action='store_true',
                        help='Log training and validation metrics to wandb')
    parser.add_argument('--no_log_wandb', action='store_false', dest='log_wandb')
    parser.set_defaults(log_wandb=False)
    parser.add_argument('--wandb_project', default='None', type=str,
                        help='Project name on wandb')
    parser.add_argument('--wandb_entity', default=None, type=str,
                        help='User or team name on wandb')
    parser.add_argument('--wandb_run_name', default=None, type=str,
                        help='Run name on wandb')
    parser.add_argument('--show_user_warnings', default=False, action='store_true')

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    return args


def get_model(args):
    """Creates and returns model from arguments
    """
    print(f"Creating model: {args.model} for inputs {args.in_domains} and outputs {args.out_domains}")

    input_adapters = {
        domain: DOMAIN_CONF[domain]['input_adapter'](
            stride_level=DOMAIN_CONF[domain]['stride_level'],
            patch_size_full=args.patch_size,
        )
        for domain in args.in_domains
    }

    output_adapters = {
        domain: DOMAIN_CONF[domain]['output_adapter'](
            stride_level=DOMAIN_CONF[domain]['stride_level'],
            patch_size_full=args.patch_size,
            dim_tokens=args.decoder_dim,
            depth=args.decoder_depth,
            num_heads=args.decoder_num_heads,
            use_task_queries=args.decoder_use_task_queries,
            task=domain,
            context_tasks=list(args.in_domains),
            use_xattn=args.decoder_use_xattn
        )
        for domain in args.out_domains
    }

    # Add normalized pixel output adapter if specified
    if args.extra_norm_pix_loss:
        for k in list(args.in_domains):
            if 'RGB' in k:
                output_adapters['norm_'+k] = DOMAIN_CONF[k]['output_adapter'](
                    stride_level=DOMAIN_CONF[k]['stride_level'],
                    patch_size_full=args.patch_size,
                    dim_tokens=args.decoder_dim,
                    depth=args.decoder_depth,
                    num_heads=args.decoder_num_heads,
                    use_task_queries=args.decoder_use_task_queries,
                    task=k,
                    context_tasks=list(args.in_domains),
                    use_xattn=args.decoder_use_xattn
                )

    model = create_model(
        args.model,
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        num_global_tokens=args.num_global_tokens,
        drop_path_rate=args.drop_path
    )

    return model

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device















def main(args):
    utils.init_distributed_mode(args)
    dev_name = get_device()
    device = torch.device(dev_name)

    # Fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True



    args.in_domains = args.in_domains.split('-')
    args.out_domains = args.out_domains.split('-')
    args.all_domains = list(set(args.in_domains) | set(args.out_domains))
    model = get_model(args)

    
    checkpoint = torch.load("/work/mech-ai/ayanlade/sat_uav/su_multimae/pretrain/checkpoint-1999.pth", map_location='cpu')
    checkpoint_model = checkpoint['model']
    

    model.load_state_dict(checkpoint_model)
    print("loaded model")
    # pdb.set_trace()

    model.eval()
    model.to(device)




   

    with torch.no_grad():

        path_1 = "/work/mech-ai/ayanlade/data/Ames_Experimentation/Sat_RGB/W1"
        path_2 = "/work/mech-ai/ayanlade/data/Ames_Experimentation/UAV_RGB/W1"


        # path_1 = "/work/mech-ai/ayanlade/data/New_Ames_moreValid/valid/Sat_RGB/W1"
        # path_2 = "/work/mech-ai/ayanlade/data/New_Ames_moreValid/valid/UAV_RGB/W1"



        for vi in ["gli","vari","rgbvi","ngrdi"]:

            table = []
            for item in os.listdir(path_1):
                row = {"Plotname": item}

                for week in ["W1","W2","W3"]:

                    sattest = os.path.join(path_1,item).replace("W1",week)
                    uavtest = os.path.join(path_2,item).replace("W1",week)

                    im = Image.open(sattest)
                    # itme = Image.open(item)
                    width, height = im.size
                    crop_box = (1, 1, width-1, height-1)
                    im = im.crop(crop_box)

                    dp = Image.open(uavtest)
                    bands = dp.split()
                    if len(bands) != 3:
                        dp = Image.open(uavtest).convert('RGB')
                    
                    im = im.resize(dp.size)
                    mask__ = mask = Image.new('RGB', dp.size, color=(1, 1, 1))

                    if dp.size[0] > 224:
                        crop_left = (dp.size[0] - 224) // 2
                        crop_right = crop_left + 224
                        dp = dp.crop((crop_left, 0, crop_right, dp.size[1]))
                        im = im.crop((crop_left, 0, crop_right, dp.size[1]))
                        mask__ = mask__.crop((crop_left, 0, crop_right, dp.size[1]))

                    if dp.size[1] > 224:
                        crop_top = (dp.size[1] - 224) // 2
                        crop_bottom = crop_top + 224
                        dp = dp.crop((0, crop_top, dp.size[0], crop_bottom))
                        im = im.crop((0, crop_top, dp.size[0], crop_bottom))
                        mask__ = mask__.crop((0, crop_top, dp.size[0], crop_bottom))


                    # if dp.size[0] < 224:
                    #     padding_left = (224 - dp.size[0]) // 2
                    #     padding_right = 224 - dp.size[0] - padding_left
                    #     dp = ImageOps.expand(dp, (padding_left, 0, padding_right, 0), fill=(0, 0, 0))
                    #     im = ImageOps.expand(im, (padding_left, 0, padding_right, 0), fill=(0, 0, 0))
                    #     mask__ = ImageOps.expand(mask__, (padding_left, 0, padding_right, 0), fill=(0, 0, 0))
                    # if dp.size[1] < 224:
                    #     padding_top = (224 - dp.size[1]) // 2
                    #     padding_bottom = 224 - dp.size[1] - padding_top
                    #     dp = ImageOps.expand(dp, (0, padding_top, 0, padding_bottom), fill=(0, 0, 0))
                    #     im = ImageOps.expand(im, (0, padding_top, 0, padding_bottom), fill=(0, 0, 0))
                    #     mask__ = ImageOps.expand(mask__, (0, padding_top, 0, padding_bottom), fill=(0, 0, 0))


                    # im = im.resize(dp.size)
                    # if im.size[0] < 224:
                    #     padding = (0, 0, 224 - im.size[0], 0)
                    #     im = ImageOps.expand(im, padding, fill=(0, 0, 0))

                    # if im.size[1] < 224:
                    #     padding = (0, 0, 0, 224 - im.size[1])
                    #     im = ImageOps.expand(im, padding, fill=(0, 0, 0))

                    # mask__ = Image.new('1', dp.size, color=1)
                    # if im.size[0] < 224:
                    #     padding = (0, 0, 224 - im.size[0], 0)
                    #     mask__ = ImageOps.expand(mask__, padding, fill=0)

                    # if im.size[1] < 224:
                    #     padding = (0, 0, 0, 224 - im.size[1])
                    #     mask__ = ImageOps.expand(mask__, padding, fill=0)

                    # if dp.size[0] < 224:
                    #     padding = (0, 0, 224 - dp.size[0], 0)
                    #     dp = ImageOps.expand(dp, padding, fill=(0, 0, 0))

                    # if dp.size[1] < 224:
                    #     padding = (0, 0, 0, 224 - dp.size[1])
                    #     dp = ImageOps.expand(dp, padding, fill=(0, 0, 0))


                    input_dict = {}
                    image_size = (224,224) #224 # Train resolution
                    img = TF.to_tensor(im)
                    depth = TF.to_tensor(dp)

                    img = TF.resize(img, image_size)
                    depth = TF.resize(depth, image_size)
                    # pdb.set_trace()
                    
                    input_dict['Sat_RGB'] = TF.normalize(img, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD).unsqueeze(0)
                    input_dict['UAV_RGB'] = TF.normalize(depth, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD).unsqueeze(0)

                    input_dict = {k: v.to(device) for k,v in input_dict.items()}
                    print("loaded data")
                    

                    preds, masks, encoded_tokens = model(input_dict,num_encoded_tokens=196,alphas=1.0,sample_tasks_uniformly=False,samples_p_task=True)
                    # fp32_output_adapters= args.fp32_output_adapters.split('-'))

                    
                    preds = {domain: pred.detach().cpu() for domain, pred in preds.items()}
                    masks = {domain: mask.detach().cpu() for domain, mask in masks.items()}

                    epoch =0
                    out_path=args.output_dir
                    sat_1_RGBVI_avg, sat_1_GLI_avg, sat_1_VARI_avg,sat_1_NGRDI_avg, sat_2_RGBVI_avg, sat_2_GLI_avg, sat_2_VARI_avg,sat_2_NGRDI_avg, uav_1_RGBVI_avg,    uav_1_GLI_avg , uav_1_VARI_avg ,    uav_1_NGRDI_avg ,    uav_2_RGBVI_avg , uav_2_GLI_avg ,    uav_2_VARI_avg , uav_2_NGRDI_avg = plot_predictions_test(input_dict, preds, masks, epoch, out_path, sattest, uavtest, mask__)
                    

                    row[week + "_s_real"] = sat_1_GLI_avg if vi == "gli" else sat_1_VARI_avg if vi == "vari" else sat_1_RGBVI_avg if vi == "rgbvi" else sat_1_NGRDI_avg
                    row[week + "_s_pred"] = sat_2_GLI_avg if vi == "gli" else sat_2_VARI_avg if vi == "vari" else sat_2_RGBVI_avg if vi == "rgbvi" else sat_2_NGRDI_avg
                    row[week + "_u_real"] = uav_1_GLI_avg if vi == "gli" else uav_1_VARI_avg if vi == "vari" else uav_1_RGBVI_avg if vi == "rgbvi" else uav_1_NGRDI_avg
                    row[week + "_u_pred"] = uav_2_GLI_avg if vi == "gli" else uav_2_VARI_avg if vi == "vari" else uav_2_RGBVI_avg if vi == "rgbvi" else uav_2_NGRDI_avg

                table.append(row)
            df = pd.DataFrame(table)
            print(vi)
            print(df)
            df.to_excel(vi+'.xlsx', index=False)


                
       






                    





if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
