import torch
from torch.optim.lr_scheduler import MultiStepLR

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
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(
        optimizer, 
        args.scheduler_milestones, 
        args.scheduler_rate
    )
    prompt_text = get_prompt_text(label_map)
    ap_best = 0
    
    for e in range(args.max_epoch):
        model.train()
        loss_total1, loss_total2 = 0, 0
        normal_iter = iter(normal_loader)
        abnormal_iter = iter(abnormal_loader)

        for i in range(min(len(normal_loader), len(abnormal_loader))):
            step = 0
            normal_features, normal_label, normal_lengths = next(normal_iter)
            abnormal_features, abnormal_label, abnormal_lengths = next(abnormal_iter)
            
            visual_features = torch.cat([normal_features, abnormal_features], dim=0).to(device)
            if torch.isnan(visual_features).any():
                visual_features = torch.nan_to_num(visual_features, nan=0.0)
            
            text_labels = list(normal_label) + list(abnormal_label)
            feat_lengths = torch.cat([normal_lengths, abnormal_lengths], dim=0).to(device)
            # one-hot vector
            text_labels = get_batch_label(text_labels, prompt_text, label_map).to(device)
            
            # forward
            text_features, logits1, logits2 = model(
                visual_features, None, prompt_text, feat_lengths) 
            
            loss1 = CLAS2(logits1, text_labels, feat_lengths, device)
            loss_total1 += loss1.item()
            loss2 = CLASM(logits2, text_labels, feat_lengths, device)
            loss_total2 += loss2.item()
            loss3 = torch.zeros(1).to(device)
            text_features_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
            for j in range(1, text_features.shape[0]):
                text_feature_abr = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
                loss3 += torch.abs(text_features_normal @ text_feature_abr)
            loss3 = loss3 / 13 * 1e-1
            loss = loss1 + loss2 + loss3
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += i * normal_loader.batch_size * 2
            
            if step % 1280 == 0 and step != 0:
                wandb.log({
                    'train/loss1': loss_total1 / (i+1),
                    'train/loss2': loss_total2 / (i+1),
                    'train/loss3': loss3.item()
                })
                print('epoch: ', e+1, '| step: ', step, '| loss1: ', loss_total1 / (i+1), '| loss2: ', loss_total2 / (i+1), '| loss3: ', loss3.item())
                AUC, AP = test(
                    model, 
                    test_loader, 
                    args.visual_length, 
                    prompt_text, 
                    gt, 
                    gtsegments, 
                    gtlabels, 
                    device,
                )
                AP = AUC

                if AP > ap_best:
                    ap_best = AP 
                    checkpoint = {
                        'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'ap': ap_best}
                    torch.save(checkpoint, args.checkpoint_path)
                
        scheduler.step()
        torch.save(model.state_dict(), 'checkpoints/ucfcrime_cur.pth')
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    checkpoint = torch.load(args.checkpoint_path)
    torch.save(model.state_dict(), args.model_path)