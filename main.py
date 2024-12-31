import os
import json
import argparse
import wandb

import torch
import torch.distributed as dist

from parser import update_ucfcrime_args
from data.__getter__ import get_loader
from model.VADCLIP import CLIPVAD
from train.ucf_train import train
import warnings
warnings.filterwarnings("ignore")


def main(args, label_map):
    wandb.init(project='EVCLIP', config=vars(args), name=args.exp_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_loader, ab_loader, test_loader = get_loader(args, label_map)
    model = CLIPVAD(
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
    train(
        args,
        model,
        n_loader,
        ab_loader,
        test_loader,
        label_map,
        device=device
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EVCLIP')
    parser.add_argument('--exp_name', default='ucfcrime', type=str)
    parser.add_argument('--dataset', default='ucfcrime', type=str)
    parser.add_argument('--ds', default='vitb_rgb', type=str)

    args = parser.parse_args()
    if args.dataset == 'ucfcrime':
        args = update_ucfcrime_args(args)
        with open('configs/ucfcrime_label_map.json', 'r') as f:
            label_map = json.load(f)
    main(args, label_map)