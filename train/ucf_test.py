import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import wandb

from .utils import get_batch_mask
from .metrics import getDetectionMAP as dmAP


def test(
        model, 
        test_loader, 
        maxlen, 
        prompt_text, 
        gt, 
        gtsegments, 
        gtlabels, 
        device,
    ):
    model.to(device)
    model.eval()

    classwise_roc = {
        'Abuse': [], 'Arrest': [], 'Arson': [], 'Assault': [],
        'Burglary': [], 'Explosion': [], 'Fighting': [], 'RoadAccidents': [],
        'Robbery': [], 'Shooting': [], 'Shoplifting': [], 'Stealing': [],
        'Vandalism': [], 'Normal': []
    }
    classwise_gt = {
        'Abuse': [], 'Arrest': [], 'Arson': [], 'Assault': [],
        'Burglary': [], 'Explosion': [], 'Fighting': [], 'RoadAccidents': [],
        'Robbery': [], 'Shooting': [], 'Shoplifting': [], 'Stealing': [],
        'Vandalism': [], 'Normal': []
    }
    st = 0

    element_logits2_stack = []
    with torch.no_grad():
        for i, item in enumerate(test_loader):
            visual = item[0].squeeze(0)
            cls = item[1][0]
            length = item[2]

            length = int(length)
            len_cur = length
            if len_cur < maxlen:
                visual = visual.unsqueeze(0)

            if torch.isnan(visual).any():
                visual = torch.nan_to_num(visual, nan=0.0)
            visual = visual.to(device)

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
            _, logits1, logits2 = model(visual, padding_mask, prompt_text, lengths)
            logits1 = logits1.reshape(logits1.shape[0] * logits1.shape[1], logits1.shape[2])
            logits2 = logits2.reshape(logits2.shape[0] * logits2.shape[1], logits2.shape[2])
            prob2 = (1 - logits2[0:len_cur].softmax(dim=-1)[:, 0].squeeze(-1))
            prob1 = torch.sigmoid(logits1[0:len_cur].squeeze(-1))

            if i == 0:
                ap1 = prob1
                ap2 = prob2
                #ap3 = prob3
            else:
                ap1 = torch.cat([ap1, prob1], dim=0)
                ap2 = torch.cat([ap2, prob2], dim=0)

            element_logits2 = logits2[0:len_cur].softmax(dim=-1).detach().cpu().numpy()
            element_logits2 = np.repeat(element_logits2, 16, 0)
            element_logits2_stack.append(element_logits2)

            classwise_roc[cls].append(prob1.cpu().numpy())
            classwise_gt[cls].append(gt[16*st : 16*(st+prob1.shape[0])])
            st += prob1.shape[0]

    ap1 = ap1.cpu().numpy()
    ap2 = ap2.cpu().numpy()
    ap1 = ap1.tolist()
    ap2 = ap2.tolist()

    ROC1 = roc_auc_score(gt, np.repeat(ap1, 16))
    AP1 = average_precision_score(gt, np.repeat(ap1, 16))
    ROC2 = roc_auc_score(gt, np.repeat(ap2, 16))
    AP2 = average_precision_score(gt, np.repeat(ap2, 16))

    print("AUC1: ", ROC1, " AP1: ", AP1)
    print("AUC2: ", ROC2, " AP2:", AP2)

    # dmap, iou = dmAP(element_logits2_stack, gtsegments, gtlabels, excludeNormal=False)
    # averageMAP = 0
    # for i in range(5):
    #     print('mAP@{0:.1f} ={1:.2f}%'.format(iou[i], dmap[i]))
    #     averageMAP += dmap[i]
    # averageMAP = averageMAP/(i+1)
    # print('average MAP: {:.2f}'.format(averageMAP))

    wandb.log({
        'test/AP1': AP1,
        'test/AP2': AP2,
        'test/ROC1': ROC1,
        'test/ROC2': ROC2,
        # 'test/mAP': averageMAP
    })

    for cls in classwise_roc.keys():
        cls_pred = np.concatenate(classwise_roc[cls])
        cls_gt = np.concatenate(classwise_gt[cls])
        if len(cls_gt) == 0 or sum(cls_gt) == 0:
            continue
        cls_roc = roc_auc_score(cls_gt, np.repeat(cls_pred, 16))
        cls_ap = average_precision_score(cls_gt, np.repeat(cls_pred, 16))
        print(cls, 'ROC:', cls_roc, 'AP:', cls_ap)
        wandb.log({
            'classwise/ROC/' + cls: cls_roc,
            'classwise/AP/' + cls: cls_ap
        })

    return ROC1, AP1