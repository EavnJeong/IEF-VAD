import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR

import numpy as np
import math
import wandb

from .utils import get_prompt_text, get_batch_label
from .loss import CLAS2, CLASM
from .xd_test import test


def train(
        args,
        model, 
        train_loader, 
        test_loader, 
        label_map: dict, 
        device
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
    ap_best = 0

    for e in range(args.max_epoch):
        for i, item in enumerate(train_loader):
            step = 0
            model.train()
        
            img_features, ev_features, text_labels, feat_lengths = item
            img_features = img_features.to(device)
            ev_features = ev_features.to(device)
            feat_lengths = feat_lengths.to(device)
            text_labels = get_batch_label(text_labels, prompt_text, label_map).to(device)

            outputs = model(
                img_features, 
                ev_features, 
                None, 
                prompt_text,
                feat_lengths
            )
            logits = outputs['logits']
            image_mu, event_mu = outputs['image_mu'], outputs['event_mu']
            image_logvar, event_logvar = outputs['image_logvar'], outputs['event_logvar']
            
            loss_classification = CLAS2(logits, text_labels, feat_lengths, device)
            
            image_mu_norm = F.normalize(image_mu, p=2, dim=-1)
            event_mu_norm = F.normalize(event_mu, p=2, dim=-1)
            cos_sim = F.cosine_similarity(image_mu_norm, event_mu_norm, dim=-1)
            loss_cos = 1 - cos_sim  # (B, T)
            norm_image = torch.norm(image_mu, p=2, dim=-1)
            norm_event = torch.norm(event_mu, p=2, dim=-1)
            loss_norm = torch.abs(norm_image - norm_event)
            loss_reg = loss_cos.mean() + loss_norm.mean()

            effective_logvar_image = image_logvar + math.log(model.temporal.nu / (model.temporal.nu + 1))
            effective_logvar_event = event_logvar + math.log(model.temporal.nu / (model.temporal.nu + 1))
            kl_loss_image = -0.5 * torch.mean(1 + effective_logvar_image - image_mu.pow(2) - effective_logvar_image.exp())
            kl_loss_event = -0.5 * torch.mean(1 + effective_logvar_event - event_mu.pow(2) - effective_logvar_event.exp())
            loss_kl = kl_loss_image + kl_loss_event
            
            lambda_reg = 0.01
            lambda_kl = 0.01   
            loss = loss_classification + lambda_kl * loss_kl + lambda_reg * loss_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += i * train_loader.batch_size
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
                    label_map,
                    vis= True if (e+1) % 5 ==0 and e > 0 and step > 33000 else False
                )

                if AP > ap_best:
                    ap_best = AP 
                    checkpoint = {
                        'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'ap': ap_best,
                    }
                    torch.save(checkpoint, f'checkpoints/{args.exp_name}.pth')            

        scheduler.step()
        checkpoint = torch.load(f'checkpoints/{args.exp_name}.pth', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
    checkpoint = torch.load(f'checkpoints/{args.exp_name}.pth', weights_only=False)
    torch.save(checkpoint['model_state_dict'], f'checkpoints/{args.exp_name}.pth')