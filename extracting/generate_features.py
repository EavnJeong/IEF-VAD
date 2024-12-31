import cv2
import os
import argparse
import numpy as np

import torch
from model.clip import clip
from PIL import Image
from tqdm import tqdm

from crop import video_crop


def load_model(args, device):
    model, preprocess = clip.load("ViT-B/32", device)
    return model, preprocess


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_model(args, device)
    chunk_size = 16

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    
    for label_path in os.listdir(args.video_dir):
        print(f'Processing {label_path}...')
        if not os.path.isdir(os.path.join(args.save_dir, label_path)):
            os.makedirs(os.path.join(args.save_dir, label_path))

        if label_path in ['Abuse', 'Arrest', 'Assault', 'Explosion', 'Fighting', 'Normal', 'Shooting', 'Stealing', 'Vandalism']:
            continue

        for video_name in tqdm(os.listdir(os.path.join(args.video_dir, label_path))):
            for crop_type in range(10):
                video_path = os.path.join(args.video_dir, label_path, video_name)
                save_path = os.path.join(args.save_dir, label_path, video_name)
                save_path = save_path.replace('.mp4', f'__{str(crop_type)}.npy')
                if os.path.exists(save_path):
                    print(f'{save_path} already exists')
                    continue
                
                video = cv2.VideoCapture(video_path)
                frames = []
                while True:
                    ret, frame = video.read()
                    if not ret:
                        break
                    frames.append(frame)
                video = np.array(frames)
            
                corp_video = video_crop(video, crop_type)    
                video_features = torch.zeros(0).to(device)
                with torch.no_grad():
                    for i in range(video.shape[0]):
                        img = Image.fromarray(corp_video[i])
                        img = preprocess(img).unsqueeze(0).to(device)
                        feature = model.encode_image(img)
                        video_features = torch.cat([video_features, feature], dim=0)
                
                num_chunks = video_features.shape[0] // chunk_size
                video_features = video_features[:num_chunks * chunk_size]
                video_features = video_features.reshape(num_chunks, chunk_size, -1)
                video_features = video_features.mean(dim=1)                
                video_features = video_features.detach().cpu().numpy()
                
                np.save(save_path, video_features)
                print(f'Saved to {save_path}')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract video features')
    parser.add_argument('--video_dir', type=str, default='/mnt/Data_3/UCFCrime_raw/videos')
    parser.add_argument('--save_dir', type=str, default='/mnt/Data_3/UCFCrime_clip_vitb32')
    parser.add_argument('--clip_ckpt', type=str, default='checkpoints/eventclip_vitb.pt')
    args = parser.parse_args()
    main(args)