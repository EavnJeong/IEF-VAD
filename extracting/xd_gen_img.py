import cv2
import os
import argparse
import numpy as np

import torch
import sys

from clip import clip
from PIL import Image
from tqdm import tqdm

from crop import video_crop


def load_model(args, device):
    if args.model == 'vitb':
        model, preprocess = clip.load("ViT-B/32", device)
    elif args.model == 'vitl':
        model, preprocess = clip.load("ViT-L/14", device)
    return model, preprocess


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_model(args, device)
    model.eval()
    chunk_size = 16

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    video_list = os.listdir(args.video_dir)
    video_list.sort(key=lambda x: os.path.getsize(os.path.join(args.video_dir, x)), reverse=True)
    for video_name in tqdm(video_list):
        for crop_type in range(10):
            try:
                video_path = os.path.join(args.video_dir, video_name)
                save_path = os.path.join(args.save_dir, video_name)
                save_path = save_path.replace('.mp4', f'__{str(crop_type)}.npy')
                if os.path.exists(save_path):
                    print(f'{save_path} already exists')
                    continue

                f_video = cv2.VideoCapture(video_path)
                frames = []
                video_features = torch.zeros(0).to(device)
                while True:
                    ret, frame = f_video.read()
                    if not ret:
                        break
                    frames.append(frame)

                    if len(frames) == chunk_size * args.batch_size:
                        video = np.array(frames)
                        crop_video = video_crop(video, crop_type)
                        crop_video = crop_video.reshape(args.batch_size, chunk_size, 224, 224, 3)
                        with torch.no_grad():
                            for i in range(args.batch_size):
                                imgs = []
                                for j in range(chunk_size):
                                    img = Image.fromarray(crop_video[i][j])
                                    img = preprocess(img).to(device)
                                    imgs.append(img)
                                
                                imgs = torch.stack(imgs, dim=0)
                                feature = model.encode_image(imgs)
                                feature = feature.mean(dim=0).unsqueeze(0)
                                video_features = torch.cat([video_features, feature], dim=0)
                        frames = []

                if len(frames) > chunk_size:
                    video = np.array(frames)
                    crop_video = video_crop(video, crop_type)
                    batch_size = len(crop_video) // chunk_size
                    crop_video = crop_video[:batch_size * chunk_size]
                    crop_video = crop_video.reshape(batch_size, chunk_size, 224, 224, 3)
                    
                    with torch.no_grad():
                        for i in range(batch_size):
                            imgs = []
                            for j in range(chunk_size):
                                img = Image.fromarray(crop_video[i][j])
                                img = preprocess(img).to(device)
                                imgs.append(img)

                            imgs = torch.stack(imgs, dim=0)
                            feature = model.encode_image(imgs)
                            feature = feature.mean(dim=0).unsqueeze(0)
                            video_features = torch.cat([video_features, feature], dim=0)

                video_features = video_features.cpu().numpy()
                np.save(save_path, video_features)
                print(f'Saved to {save_path}')
            except Exception as e:
                print(f"Error occurred for video {video_name}, crop type {crop_type}: {e}")
                continue
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract video features')
    parser.add_argument('--video_dir', type=str, default='/media/Data1/xd_raw/videos')
    parser.add_argument('--save_dir', type=str, default='/media/Data1/xd_dataset/vitl/rgb')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model', type=str, default='vitl', choices=['vitb', 'vitl'])

    args = parser.parse_args()
    main(args)