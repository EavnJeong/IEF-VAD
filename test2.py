import os, json, argparse, warnings, itertools
from torch.distributions import StudentT
import pandas as pd, textwrap
import numpy as np
import torch
from itertools import product
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from train.utils import get_batch_mask
from data.__getter__ import get_loader
from model.imf_vad import MMFMIL
from parser import update_ucf_args, update_xd_args, update_shang_args, update_msad_args
warnings.filterwarnings("ignore")


def brier_score(pred, gt):
    return np.mean((pred - gt) ** 2)


def kl_divergence(pred_clean, pred_noisy, eps=1e-8):
    p, q = np.clip(pred_clean, eps, 1 - eps), np.clip(pred_noisy, eps, 1 - eps)
    return np.mean(p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q)))


def weight_ratio(w_img, w_ev):
    return (w_img / (w_img + w_ev)).mean()


SWEEP_CFGS = {
    "IMG_NOISE":   {"sigma_img": [0,0.05,0.1,0.2,0.3,0.5], "sigma_ev": [0]},
    "EV_NOISE":    {"sigma_ev": [0,0.05,0.1,0.2,0.3,0.5], "sigma_img": [0]},
}


def run_test(args, model, loader, gt, device,
             sigma_img=0, sigma_ev=0,
             outlier_img=0, outlier_ev=0):
    model.eval()
    preds_clean = None
    preds_noisy = None
    w_img_orig = []
    w_ev_orig = []
    w_img_all = []
    w_ev_all = []
    maxlen = args.visual_length
    repeat = 16
    img_logvar = []
    ev_logvar = []

    with torch.no_grad():
        for visuals, events, _, length in loader:
            visuals = visuals.squeeze(0)
            events = events.squeeze(0)
            
            if length < maxlen:
                visuals = visuals.unsqueeze(0)
                events = events.unsqueeze(0)

            visuals = torch.nan_to_num(visuals).to(device)
            events = torch.nan_to_num(events).to(device)

            out_c = model(visuals, events, None, None, torch.tensor([length]))
            logits_c = out_c['logits'].reshape(-1, out_c['logits'].size(-1))
            p_c = torch.sigmoid(logits_c[:length].squeeze(-1))
            out_c['w_i'] = out_c['w_i'].reshape(-1, 768)
            out_c['w_e'] = out_c['w_e'].reshape(-1, 768)
            w_img_orig.append(out_c['w_i'][:length].cpu().squeeze())
            w_ev_orig.append(out_c['w_e'][:length].cpu().squeeze())

            v_p, e_p = visuals.clone(), events.clone()
            if sigma_img:
                indexs = torch.randperm(v_p.shape[1])[: int(v_p.shape[1] * sigma_img)]
                v_p[:, indexs] = v_p[:, indexs] * 0.01
                
            if sigma_ev:
                indexs = torch.randperm(e_p.shape[1])[: int(e_p.shape[1] * sigma_ev)]
                e_p[:, indexs] = e_p[:, indexs] * 0.01
    
            out_n = model(v_p, e_p, None, None, torch.tensor([length]))
            logits_n = out_n['logits'].reshape(-1, out_n['logits'].size(-1))
            p_n = torch.sigmoid(logits_n[:length].squeeze(-1))

            preds_clean = p_c if preds_clean is None else torch.cat([preds_clean, p_c])
            preds_noisy = p_n if preds_noisy is None else torch.cat([preds_noisy, p_n])
            if len(out_n['w_i'].shape)==3:
                out_n['w_i'] = out_n['w_i'].reshape(-1, 768)
                out_n['w_e'] = out_n['w_e'].reshape(-1, 768)
            w_img_all.append(out_n['w_i'][:length].cpu().squeeze())
            w_ev_all.append(out_n['w_e'][:length].cpu().squeeze())

    yc = preds_clean.cpu().numpy()
    yn = preds_noisy.cpu().numpy()
    yc_rep = np.repeat(yc, repeat)
    yn_rep = np.repeat(yn, repeat)
    gt_slice = gt[: len(yn_rep)]

    w_img_all = torch.cat(w_img_all)
    w_ev_all = torch.cat(w_ev_all)
    w_img_orig = torch.cat(w_img_orig)
    w_ev_orig = torch.cat(w_ev_orig)

    w_img_change = w_img_orig.mean(0) - w_img_all.mean(0)
    w_ev_change = w_ev_orig.mean(0) - w_ev_all.mean(0)

    w_img = w_img_all.mean(1).numpy()
    w_ev = w_ev_all.mean(1).numpy()
    w_img = np.repeat(w_img, repeat)
    w_ev = np.repeat(w_ev, repeat)

    brier = brier_score(yn_rep, gt_slice)
    kl    = kl_divergence(yc_rep, yn_rep)
    
    w_img_mean = w_img.mean()
    w_ev_mean  = w_ev.mean()
    auc   = roc_auc_score(gt_slice, yn_rep)
    ap    = average_precision_score(gt_slice, yn_rep)

    w_img_anomaly = np.mean(w_img[gt_slice == 1])
    w_ev_anomaly  = np.mean(w_ev[gt_slice == 1])
    w_img_normal  = np.mean(w_img[gt_slice == 0])
    w_ev_normal   = np.mean(w_ev[gt_slice == 0])

    return brier, kl, w_img_mean, w_ev_mean, auc, ap, w_img_anomaly, w_ev_anomaly, w_img_normal, w_ev_normal, w_img_change, w_ev_change


def main(args, label_map):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = []
    results2 = []
    results3 = []

    noise_model = args.noise_model
    if args.dataset == 'xd':
        _, loader = get_loader(args, label_map)
    else:
        _, _, loader = get_loader(args, label_map)
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
        device=device,
        args=args    
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt_path))

    gt = np.load(args.gt_path)
    for perturb, cfg in SWEEP_CFGS.items():
        keys        = list(cfg.keys())      
        level_lists = [cfg[k] for k in keys]
        for lv_combo in product(*level_lists):
            kw = dict(zip(keys, lv_combo))
    
            brier, kl, w_img, w_ev, auc, ap, w_img_ab, w_ev_ab, w_img_n, w_ev_n, w_img_change, w_ev_change  = run_test(
                args, model, loader, gt, device, **kw
            )
            results.append((perturb, lv_combo, noise_model, brier, kl, w_img, w_ev, auc, ap, w_img_ab, w_ev_ab, w_img_n, w_ev_n))
            results2.append(w_img_change.cpu().numpy())
            results3.append(w_ev_change.cpu().numpy())
            print(
                f"Perturb: {perturb}, Levels: {kw}, "
                f"Brier: {brier:.4f}, KL: {kl:.4f}, "
                f"w_img: {w_img:.4f}, w_ev: {w_ev:.4f}, " 
                f"AUC: {auc:.4f}, AP: {ap:.4f} "
                f"Abnormal w_img: {w_img_ab:.4f}, w_ev: {w_ev_ab:.4f}, "
                f"Normal w_img: {w_img_n:.4f}, w_ev: {w_ev_n:.4f}"
            )
    print_table(results, args)
    df1 = pd.DataFrame(results2)
    df2 = pd.DataFrame(results3)
    df1.to_csv(f'{args.exp_name}_w_img_change.csv', index=False, header=False)
    df2.to_csv(f'{args.exp_name}_w_ev_change.csv', index=False, header=False)
    

def print_table(rows, args):
    df = pd.DataFrame(
        rows,
        columns=["Perturb", "Level", "Model", "Brier", "KL", "w_img", "w_ev", "AUC", "AP", "w_img_ab", "w_ev_ab", "w_img_n", "w_ev_n"],
    )
    wide = df.pivot_table(
        index=["Perturb", "Level"],
        columns="Model",
        values=["Brier", "KL", "w_img", "w_ev", "AUC", "AP", "w_img_ab", "w_ev_ab", "w_img_n", "w_ev_n"],
    ).sort_index()

    print(textwrap.dedent(
        wide.to_string(float_format=lambda x: f"{x:.4f}")
    ))
    wide.to_csv(f'{args.exp_name}.csv', float_format='%.4f')
    print(">> Saved results table to results_table.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EVCLIP Test')
    # data & training parameters
    parser.add_argument('--exp_name', default='ucfcrime', type=str)
    parser.add_argument('--dataset',  default='ucfcrime', type=str)
    parser.add_argument('--ds',       default='vitl_rgb', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--max_epoch',  default=10, type=int)
    parser.add_argument('--lr',         default=2e-5, type=float)
    # architecture parameters
    parser.add_argument('--nu',                 default=8,    type=int)
    parser.add_argument('--noise_model',        default='StudentT', type=str)
    parser.add_argument('--num_refinement_steps', default=10, type=int)
    parser.add_argument('--lambda_ref',         default=0.5,  type=float)
    parser.add_argument('--epsilon',            default=1e-8, type=float)
    # Fusion 
    parser.add_argument('--noise_sigma_img', type=float, default=0.0,
                        help='Gaussian noise std for image')
    parser.add_argument('--noise_sigma_ev',  type=float, default=0.0,
                        help='Gaussian noise std for event')
    parser.add_argument('--outlier_ratio',   type=float, default=0.0,
                        help='Fraction of frames to corrupt with outliers')
    parser.add_argument('--dropout_prob',    type=float, default=0.0,
                        help='Probability to drop one modality')
    # visualization flag
    parser.add_argument('--vis', action='store_true', help='Enable visualizations')
    # checkpoint & model settings
    parser.add_argument('--ckpt_path',     default='checkpoints/best.pth', type=str)
    parser.add_argument('--visual_head',   default=8, type=int)
    parser.add_argument('--visual_layers', default=2, type=int)

    args = parser.parse_args()
    if args.dataset == 'ucfcrime':
        args = update_ucf_args(args)
        with open('configs/ucfcrime_label_map.json','r') as f:
            label_map = json.load(f)
    elif args.dataset == 'xd':
        args = update_xd_args(args)
        with open('configs/xd_label_map.json','r') as f:
            label_map = json.load(f)
    elif args.dataset == 'shang':
        args = update_shang_args(args)
        with open('configs/shang_label_map.json','r') as f:
            label_map = json.load(f)
    elif args.dataset == 'msad':
        args = update_msad_args(args)
        with open('configs/msad_label_map.json', 'r') as f:
            label_map = json.load(f)
    print(args)
    main(args, label_map)


# python test2.py --ckpt_path checkpoints/ucf.pth
# python test2.py --dataset shang --ckpt_path checkpoints/shang.pth
# python test2.py --dataset xd --ckpt_path checkpoints/xd.pth
# python test2.py --dataset msad --ckpt_path checkpoints/msad.pth