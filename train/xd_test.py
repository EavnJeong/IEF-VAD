import torch
import os
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score
import wandb

from .utils import get_batch_mask


def test(
        args,
        model, 
        test_loader, 
        maxlen, 
        prompt_text, 
        gt, 
        device,
        label_map,
        vis=False,
        attn=False,
    ):
    model.to(device)
    model.eval()
    if not os.path.exists('vis'):
        os.makedirs('vis')
    if not os.path.exists(f'vis/{args.exp_name}'):
        os.makedirs(f'vis/{args.exp_name}')

    if args.dataset == 'ucfcrime':
        classwise_roc = {
            'Abuse': [], 'Arrest': [], 'Arson': [], 'Assault': [],
            'Burglary': [], 'Explosion': [], 'Fighting': [], 'RoadAccidents': [],
            'Robbery': [], 'Shooting': [], 'Shoplifting': [], 'Stealing': [],
            'Vandalism': [], 'Normal': []
        }
    elif args.dataset == 'xd':
        classwise_roc = {
            'normal': [], 'fighting': [], 'shooting': [], 
            'riot': [], 'abuse': [], 'car accident': [],
            'explosion': []
        }
    elif args.dataset == 'shang':
        classwise_roc = {
            'car': [], 'chasing': [], 'fall': [], 'fighting': [], 'monocycle': [], 
            'robbery': [], 'running': [], 'skateboard': [], 'throwing_object': [], 'vehicle': [], 
            'vaudeville': [], 'normal': []
        }
    classwise_gt = { key: [] for key in classwise_roc.keys() }
    classwise_wi = { key: [] for key in classwise_roc.keys() }
    classwise_we = { key: [] for key in classwise_roc.keys() }
    classwise_fused = { key: [] for key in classwise_roc.keys() }
    classwise_image_mu = { key: [] for key in classwise_roc.keys() }
    classwise_event_mu = { key: [] for key in classwise_roc.keys() }

    st = 0
    attn_weights = []
    labels = []
    with torch.no_grad():
        for i, item in enumerate(test_loader):
            img_visual = item[0].squeeze(0)
            ev_visual = item[1].squeeze(0)
            cls = item[2][0]
            cls = label_map[cls.split('-')[0]]
            length = item[3]

            length = int(length)
            len_cur = length
            if len_cur < maxlen:
                img_visual = img_visual.unsqueeze(0)
                ev_visual = ev_visual.unsqueeze(0)

            if torch.isnan(img_visual).any():
                img_visual = torch.nan_to_num(img_visual, nan=0.0) 
            img_visual = img_visual.to(device)
            if torch.isnan(ev_visual).any():
                ev_visual = torch.nan_to_num(ev_visual, nan=0.0)
            ev_visual = ev_visual.to(device)

            lengths = torch.zeros(int(length / maxlen) + 1)
            for j in range(int(length / maxlen) + 1):
                if j == 0 and length < maxlen:
                    lengths[j] = length
                elif j == 0 and length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                elif length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                else:
                    lengths[j] = length
            lengths = lengths.to(int)
            padding_mask = get_batch_mask(lengths, maxlen).to(device)
            outputs  = model(
                img_visual, 
                ev_visual,
                padding_mask, 
                prompt_text, 
                lengths,
            )
            labels.append(cls)
            logits1 = outputs['logits']
            logits1 = logits1.reshape(logits1.shape[0] * logits1.shape[1], logits1.shape[2])
            prob1 = torch.sigmoid(logits1[0:len_cur].squeeze(-1))

            if i == 0:
                ap1 = prob1
            else:
                ap1 = torch.cat([ap1, prob1], dim=0)

            classwise_roc[cls].append(prob1.cpu().numpy())
            classwise_gt[cls].append(gt[16*st : 16*(st+prob1.shape[0])])

            w_i = outputs['w_i']
            w_i = w_i.reshape(w_i.shape[0] * w_i.shape[1], w_i.shape[2])
            w_i = w_i.mean(dim=-1).cpu().squeeze().numpy()[0:len_cur]  # (T,) : 각 타임스텝의 이미지 모달리티 불확실성 가중치
            w_e = outputs['w_e']
            w_e = w_e.reshape(w_e.shape[0] * w_e.shape[1], w_e.shape[2])
            w_e = w_e.mean(dim=-1).cpu().squeeze().numpy()[0:len_cur]  # (T,) : 각 타임스텝의 이벤트 모달리티 불확실성 가중치
            classwise_wi[cls].append(w_i)
            classwise_we[cls].append(w_e)

            fused = outputs['fused']
            fused = fused.reshape(fused.shape[0] * fused.shape[1], fused.shape[2])
            fused = fused.cpu().squeeze().numpy()[0:len_cur]
            classwise_fused[cls].append(fused)
            image_mu = outputs['image_mu']
            image_mu = image_mu.reshape(image_mu.shape[0] * image_mu.shape[1], image_mu.shape[2])
            image_mu = image_mu.cpu().squeeze().numpy()[0:len_cur]
            classwise_image_mu[cls].append(image_mu)
            event_mu = outputs['event_mu']
            event_mu = event_mu.reshape(event_mu.shape[0] * event_mu.shape[1], event_mu.shape[2])
            event_mu = event_mu.cpu().squeeze().numpy()[0:len_cur]
            classwise_event_mu[cls].append(event_mu)

            st += prob1.shape[0]

    ap1 = ap1.cpu().numpy()
    ap1 = ap1.tolist()

    ROC1 = roc_auc_score(gt, np.repeat(ap1, 16))
    AP1 = average_precision_score(gt, np.repeat(ap1, 16))

    # Ano-AUC
    Ano_ROC = compute_ano_auc(classwise_gt, classwise_roc)

    print("AUC1: {:.2f}  AP1: {:.2f}".format(ROC1 * 100, AP1 * 100))
    print("Ano-AUC: {:.2f}".format(Ano_ROC * 100))
    wandb.log({
        'test/AP1': AP1,
        'test/ROC1': ROC1,
        'test/Ano-AUC': Ano_ROC
    })

    roc_results = {}
    for cls in classwise_roc.keys():
        cls_pred = np.concatenate(classwise_roc[cls])
        cls_gt = np.concatenate(classwise_gt[cls])
        if len(cls_gt) == 0 or sum(cls_gt) == 0:
            continue
        cls_roc = roc_auc_score(cls_gt, np.repeat(cls_pred, 16))
        cls_ap = average_precision_score(cls_gt, np.repeat(cls_pred, 16))
        print(cls, 'ROC: {:.2f}  AP: {:.2f}'.format(cls_roc * 100, cls_ap * 100))
        roc_results[cls] = cls_roc
        wandb.log({
            'classwise/ROC/' + cls: cls_roc,
            'classwise/AP/' + cls: cls_ap
        })

    print('-------------------------------------------------')
    if vis:
        for key, value in classwise_fused.items():
            fused = np.concatenate(value)
            fused = np.repeat(fused, 16, 0)
            classwise_fused[key] = fused

            image_mu = np.concatenate(classwise_image_mu[key])
            image_mu = np.repeat(image_mu, 16, 0)
            classwise_image_mu[key] = image_mu

            event_mu = np.concatenate(classwise_event_mu[key])
            event_mu = np.repeat(event_mu, 16, 0)
            classwise_event_mu[key] = event_mu
        visualize_similarity_metrics_classwise(args, classwise_fused, classwise_image_mu, classwise_event_mu, classwise_gt)

        classwise_pred = {}
        for key, value in classwise_roc.items():
            cls_pred = np.concatenate(value)
            cls_pred = np.repeat(cls_pred, 16)
            classwise_pred[key] = cls_pred
        classwise_wis = {}
        for key, value in classwise_roc.items():
            cls_wi = np.concatenate(classwise_wi[key])
            cls_wi = np.repeat(cls_wi, 16)
            classwise_wis[key] = cls_wi
        classwise_wes = {}
        for key, value in classwise_roc.items():
            cls_we = np.concatenate(classwise_we[key])
            cls_we = np.repeat(cls_we, 16)
            classwise_wes[key] = cls_we
        visualize_class_results(args, classwise_pred, classwise_gt, classwise_wis, classwise_wes, roc_results)

    if attn:
        return ROC1, AP1, attn_weights, labels
    else:
        return ROC1, AP1
    

def visualize_similarity_metrics_classwise(args, classwise_fused, classwise_image_mu, classwise_event_mu, classwise_gt):
    for cls in classwise_fused.keys():
        fused_list = [np.atleast_2d(arr) for arr in classwise_fused[cls]]
        image_list = [np.atleast_2d(arr) for arr in classwise_image_mu[cls]]
        event_list = [np.atleast_2d(arr) for arr in classwise_event_mu[cls]]
        gt_list = [np.atleast_1d(arr) for arr in classwise_gt[cls]]
        
        fused_arr = np.concatenate(fused_list, axis=0)   # (Total_T, D)
        image_arr = np.concatenate(image_list, axis=0)   # (Total_T, D)
        event_arr = np.concatenate(event_list, axis=0)   # (Total_T, D)
        gt_arr = np.concatenate(gt_list, axis=0)           # (Total_T,)
        
        if fused_arr.shape[0] > 3000:
            fused_arr = fused_arr[:3000]
            image_arr = image_arr[:3000]
            event_arr = event_arr[:3000]
            gt_arr = gt_arr[:3000]
        
        fused_tensor = torch.tensor(fused_arr, dtype=torch.float, device='cuda')
        image_tensor = torch.tensor(image_arr, dtype=torch.float, device='cuda')
        event_tensor = torch.tensor(event_arr, dtype=torch.float, device='cuda')
        gt_tensor = torch.tensor(gt_arr, device='cuda')
        timestamps = torch.arange(fused_tensor.shape[0], device='cuda')
        
        cosine_sim_image = F.cosine_similarity(fused_tensor, image_tensor, dim=-1)
        cosine_sim_event = F.cosine_similarity(fused_tensor, event_tensor, dim=-1)
        
        euclidean_dist_image = torch.norm(fused_tensor - image_tensor, dim=-1)
        euclidean_dist_event = torch.norm(fused_tensor - event_tensor, dim=-1)
        
        cosine_sim_image_np = cosine_sim_image.cpu().numpy()
        cosine_sim_event_np = cosine_sim_event.cpu().numpy()
        euclidean_dist_image_np = euclidean_dist_image.cpu().numpy()
        euclidean_dist_event_np = euclidean_dist_event.cpu().numpy()
        timestamps_np = timestamps.cpu().numpy()
        gt_np = gt_tensor.cpu().numpy()
        
        fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        fig.suptitle(f"Similarity Metrics over Time for Class {cls}", fontsize=16)
        
        axs[0].plot(timestamps_np, cosine_sim_image_np, label='Cosine sim (fused, image)', color='skyblue')
        axs[0].plot(timestamps_np, cosine_sim_event_np, label='Cosine sim (fused, event)', color='#90EE90')
        indices = np.where(gt_np == 1)[0]
        if indices.size > 0:
            axs[0].scatter(timestamps_np[indices], cosine_sim_image_np[indices], color='red', marker='o', s=50)
            axs[0].scatter(timestamps_np[indices], cosine_sim_event_np[indices], color='red', marker='o', s=50)
        axs[0].set_ylabel("Cosine Similarity")
        axs[0].legend()
        axs[0].grid(True)
        
        axs[1].plot(timestamps_np, euclidean_dist_image_np, label='Euclidean dist (fused, image)', color='skyblue')
        axs[1].plot(timestamps_np, euclidean_dist_event_np, label='Euclidean dist (fused, event)', color='#90EE90')
        if indices.size > 0:
            axs[1].scatter(timestamps_np[indices], euclidean_dist_image_np[indices], color='red', marker='o', s=50)
            axs[1].scatter(timestamps_np[indices], euclidean_dist_event_np[indices], color='red', marker='o', s=50)
        axs[1].set_xlabel("Timestamp")
        axs[1].set_ylabel("Euclidean Distance")
        axs[1].legend()
        axs[1].grid(True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f'vis/{args.exp_name}/similarity_{cls}.png')
        plt.clf()
    plt.close()


def visualize_class_results(args, classwise_pred, classwise_gt, classwise_wi, classwise_we, roc_results):
    for cls in classwise_pred.keys():
        probs = torch.tensor(np.array(classwise_pred[cls]), device='cuda')
        gt = torch.tensor(np.concatenate(classwise_gt[cls]), device='cuda')
        wi = torch.tensor(np.array(classwise_wi[cls]), device='cuda')
        we = torch.tensor(np.array(classwise_we[cls]), device='cuda')
        timestamps = torch.arange(probs.shape[0], device='cuda')
        
        if probs.shape[0] > 3000:
            probs = probs[:3000]
            gt = gt[:3000]
            wi = wi[:3000]
            we = we[:3000]
            timestamps = timestamps[:3000]
        
        probs_np = probs.cpu().numpy()
        gt_np = gt.cpu().numpy()
        wi_np = wi.cpu().numpy()
        we_np = we.cpu().numpy()
        timestamps_np = timestamps.cpu().numpy()
        
        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        title = f'Class: {cls}'
        if cls not in ['Normal', 'normal'] and cls in roc_results:
            title += f' | ROC {roc_results[cls]:.2f}'
        fig.suptitle(title, fontsize=16)
        
        axs[0].plot(timestamps_np, probs_np, label='Predicted Probability', color='gray', linestyle='-')
        indices = np.where(gt_np == 1)[0]
        if indices.size > 0:
            axs[0].scatter(timestamps_np[indices], probs_np[indices], color='red', s=50)
        axs[0].set_ylabel('Predicted Probability')
        axs[0].legend(loc='upper right')
        axs[0].grid(True)
        
        axs[1].plot(timestamps_np, wi_np, label='$w_i$ (Image)', color='skyblue', linestyle='-')
        axs[1].plot(timestamps_np, we_np, label='$w_e$ (Event)', color='#90EE90', linestyle='-')
        if indices.size > 0:
            axs[1].scatter(timestamps_np[indices], wi_np[indices], color='red', marker='o', s=50)
            axs[1].scatter(timestamps_np[indices], we_np[indices], color='red', marker='o', s=50)
        axs[1].set_xlabel('Timestamp (across samples)')
        axs[1].set_ylabel('Uncertainty Weight')
        axs[1].legend(loc='upper right')
        axs[1].grid(True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f'vis/{args.exp_name}/{cls}.png')
        plt.clf()
    plt.close()


def compute_ano_auc(classwise_gt, classwise_roc, repeat_factor=16):
    gt_abnormal = []
    pred_abnormal = []
    for key in classwise_gt.keys():
        if key != 'normal' and len(classwise_gt[key]) > 0:
            gt_concat = np.concatenate(classwise_gt[key])
            pred_concat = np.concatenate(classwise_roc[key])
            gt_abnormal.extend(gt_concat.tolist())
            pred_abnormal.extend(pred_concat.tolist())
    gt_abnormal = np.array(gt_abnormal)
    pred_abnormal = np.array(pred_abnormal)
    
    if len(np.unique(gt_abnormal)) > 1:
        ano_auc = roc_auc_score(gt_abnormal, np.repeat(pred_abnormal, repeat_factor))
    else:
        ano_auc = float('nan')
    
    return ano_auc