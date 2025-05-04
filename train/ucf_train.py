import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR

import math
import numpy as np
import wandb
from tqdm import tqdm

from .loss import CLAS2, CLASM
from .utils import get_batch_label, get_prompt_text
from .ucf_test import test


def train(
        args, 
        model, 
        normal_loader,
        abnormal_loader,
        test_loader,
        label_map,
        device,
    ):
    model.to(device)
    gt = np.load(args.gt_path)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(
        optimizer, 
        args.scheduler_milestones,
        args.scheduler_rate
    )
    prompt_text = get_prompt_text(label_map)
    auc_best = 0
    
    for e in range(args.max_epoch):
        normal_iter = iter(normal_loader)
        abnormal_iter = iter(abnormal_loader)
        
        for i in range(min(len(normal_loader), len(abnormal_loader))):
            step = 0
            model.train()
            img_normal_features, ev_normal_features, normal_label, normal_lengths = next(normal_iter)
            img_abnormal_features, ev_abnormal_features, abnormal_label, abnormal_lengths = next(abnormal_iter)
            
            img_visual_features = torch.cat([img_normal_features, img_abnormal_features], dim=0).to(device)
            ev_visual_features = torch.cat([ev_normal_features, ev_abnormal_features], dim=0).to(device)

            if torch.isnan(img_visual_features).any():
                img_visual_features = torch.nan_to_num(img_visual_features, nan=0.0)
            if torch.isnan(ev_visual_features).any():
                ev_visual_features = torch.nan_to_num(ev_visual_features, nan=0.0)
            
            text_labels = list(normal_label) + list(abnormal_label)
            feat_lengths = torch.cat([normal_lengths, abnormal_lengths], dim=0).to(device)
            # one-hot vector
            text_labels = get_batch_label(text_labels, prompt_text, label_map).to(device)

            outputs = model(
                img_visual_features, 
                ev_visual_features,
                None, 
                prompt_text, 
                feat_lengths,
            ) 

            logits = outputs['logits']
            image_mu, event_mu = outputs['image_mu'], outputs['event_mu']
            image_logvar, event_logvar = outputs['image_logvar'], outputs['event_logvar']

            # classification loss
            loss_classification = CLAS2(logits, text_labels, feat_lengths, device)
            # loss reg: cosine similarity and norm difference between image_mu and event_mu
            image_mu_norm = F.normalize(image_mu, p=2, dim=-1)
            event_mu_norm = F.normalize(event_mu, p=2, dim=-1)
            cos_sim = F.cosine_similarity(image_mu_norm, event_mu_norm, dim=-1)
            loss_cos = 1 - cos_sim  # (B, T)
            norm_image = torch.norm(image_mu, p=2, dim=-1)
            norm_event = torch.norm(event_mu, p=2, dim=-1)
            loss_norm = torch.abs(norm_image - norm_event)
            loss_reg = loss_cos.mean() + loss_norm.mean()

            # Gaussian
            # kl divergence
            if args.noise_model == 'Gaussian':
                kl_loss_image = -0.5 * torch.mean(1 + image_logvar - image_mu.pow(2) - image_logvar.exp())
                kl_loss_event = -0.5 * torch.mean(1 + event_logvar - event_mu.pow(2) - event_logvar.exp())
                loss_kl = kl_loss_image + kl_loss_event

            # Student T
            # effective logvar: log(sigma_eff^2) = log(sigma^2) + log(nu/(nu+1))
            if args.noise_model == 'StudentT':
                effective_logvar_image = image_logvar + math.log(model.temporal.nu / (model.temporal.nu + 1))
                effective_logvar_event = event_logvar + math.log(model.temporal.nu / (model.temporal.nu + 1))
                kl_loss_image = -0.5 * torch.mean(1 + effective_logvar_image - image_mu.pow(2) - effective_logvar_image.exp())
                kl_loss_event = -0.5 * torch.mean(1 + effective_logvar_event - event_mu.pow(2) - effective_logvar_event.exp())
                loss_kl = kl_loss_image + kl_loss_event
            
            lambda_reg = 1
            lambda_kl = 1
            loss = loss_classification + lambda_reg * loss_reg + lambda_kl * loss_kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += i * normal_loader.batch_size * 2
            
            if step % args.print_steps == 0 and step != 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/loss_classification': loss_classification.item(),
                    'train/loss_reg': loss_reg.item(),
                    'train/loss_kl': loss_kl.item(),
                    'train/loss_kl_image': kl_loss_image.item(),
                    'train/loss_kl_event': kl_loss_event.item(),
                    'train/image_mu': image_mu.mean().item(),
                    'train/event_mu': event_mu.mean().item(),
                    'train/image_logvar': image_logvar.mean().item(),
                    'train/event_logvar': event_logvar.mean().item(),
                })
                print(f'Epoch {e}, Step {step}, Loss {loss.item()}')
                print(f"Loss Classification: {loss_classification.item()}")
                print(f"Loss Regression: {loss_reg.item()}")
                print(f"Loss KL: {loss_kl.item()}")
                print(f"Loss KL Image: {kl_loss_image.item()}")
                print(f"Loss KL Event: {kl_loss_event.item()}")
                print('-------------------------------------------------')
                model.eval()
                AUC, AP = test(
                    args,
                    model, 
                    test_loader, 
                    args.visual_length, 
                    prompt_text, 
                    gt, 
                    device,
                    vis= True if (e+1) % 5 ==0 and e > 0 and step > args.vis_steps else False
                )

                if AUC > auc_best:
                    auc_best = AUC 
                    checkpoint = {
                        'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'ap': auc_best,
                    }
                    torch.save(checkpoint, f'checkpoints/{args.exp_name}.pth')            

        scheduler.step()
        checkpoint = torch.load(f'checkpoints/{args.exp_name}.pth', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
    checkpoint = torch.load(f'checkpoints/{args.exp_name}.pth', weights_only=False)
    torch.save(checkpoint['model_state_dict'], f'checkpoints/{args.exp_name}.pth')