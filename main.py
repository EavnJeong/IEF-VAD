import os
import json
import argparse
import wandb

import torch
import torch.distributed as dist

from parser import update_ucf_args, update_xd_args, update_shang_args, update_msad_args
from data.__getter__ import get_loader
from model.imf_vad import MMFMIL
from train.ucf_train import train
from train.xd_train import train as xd_train
import warnings
warnings.filterwarnings("ignore")


def main(args, label_map):
    wandb.init(project='EVCLIP', config=vars(args), name=args.exp_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MMFMIL(
        args.classes_num,
        args.embed_dim,
        args.visual_length,
        args.visual_width,
        args.visual_head,
        args.visual_layers,
        args.attn_window,
        args.prompt_prefix,
        args.prompt_postfix,
        device='cuda',
        args=args
    ).to(device)

    print('Start training...')
    if args.dataset == 'ucfcrime' or args.dataset == 'shang' or args.dataset == 'msad': 
        n_loader, ab_loader, test_loader = get_loader(args, label_map)
        train(
            args,
            model,
            n_loader,
            ab_loader,
            test_loader,
            label_map,
            device=device
        )
    elif args.dataset == 'xd':
        train_loader, test_loader = get_loader(args, label_map)
        xd_train(
            args,
            model,
            train_loader,
            test_loader,
            label_map,
            device=device
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EVCLIP')
    # data parameters
    parser.add_argument('--exp_name', default='ucfcrime', type=str)
    parser.add_argument('--dataset', default='ucfcrime', choices=['ucfcrime', 'xd', 'shang', 'msad'], type=str)
    parser.add_argument('--ds', default='vitl_rgb', type=str)
    # training parameters
    parser.add_argument('--max_epoch', default=10, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=2e-5, type=float)
    # architecture parameters
    parser.add_argument('--nu', default=8, type=int)
    parser.add_argument('--noise_model', default='StudentT', type=str)
    parser.add_argument('--num_refinement_steps', default=10, type=int)
    parser.add_argument('--lambda_ref', default=0.5, type=float)
    parser.add_argument('--epsilon', default=1e-8, type=float)
    parser.add_argument('--visual_head', default=8, type=int)
    parser.add_argument('--visual_layers', default=2, type=int)
    # utils
    parser.add_argument('--print_steps', default=1280, type=int)
    parser.add_argument('--vis_steps', default=15000, type=int)

    args = parser.parse_args()
    if args.dataset == 'ucfcrime':
        args = update_ucf_args(args)
        with open('configs/ucfcrime_label_map.json', 'r') as f:
            label_map = json.load(f)
    elif args.dataset == 'xd':
        args = update_xd_args(args)
        with open('configs/xd_label_map.json', 'r') as f:
            label_map = json.load(f)
    elif args.dataset == 'shang':
        args = update_shang_args(args)
        with open('configs/shang_label_map.json', 'r') as f:
            label_map = json.load(f)
    elif args.dataset == 'msad':
        args = update_msad_args(args)
        with open('configs/msad_label_map.json', 'r') as f:
            label_map = json.load(f)
    print(args)
    main(args, label_map)