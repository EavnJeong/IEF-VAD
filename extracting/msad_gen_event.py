import cv2
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms

import sys
from clip import clip

from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from crop import video_crop


def generate_event_image(frames, threshold=25):
    frames = torch.tensor(frames, dtype=torch.float32)

    gray_weights = torch.tensor(
        [0.2989, 0.5870, 0.1140], 
        dtype=torch.float32, 
        device=frames.device
    )
    grayscale_tensor = torch.tensordot(
        frames, 
        gray_weights, 
        dims=([-1], [0])
    )
    diffs = torch.abs(grayscale_tensor[:, 1:] - grayscale_tensor[:, :-1])

    event_images = (diffs > threshold).float()
    return event_images.sum(1)
    

def load_model(args, device):
    model, preprocess = clip.load("ViT-L/14", device)
    
    state_dict = torch.load(args.clip_ckpt)['checkpoint']
    new_state_dict = {}
    for key in state_dict:
        if 'encoder_k' in key:
            new_state_dict[key.replace('encoder_k.', '')] = state_dict[key]
    model.load_state_dict(new_state_dict)
    return model, preprocess


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_model(args, device)
    chunk_size = args.chunk_size
    batch_size = args.batch_size
    
    event_transform = transforms.Compose([
        transforms.Resize((224, 224))
    ])

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    
    for label_path in os.listdir(args.video_dir):
        print(f'Processing {label_path}...')
        if not os.path.isdir(os.path.join(args.save_dir, label_path)):
            os.makedirs(os.path.join(args.save_dir, label_path))

        for video_name in tqdm(sorted(os.listdir(os.path.join(args.video_dir, label_path)))):
            for crop_type in range(10):
                video_path = os.path.join(args.video_dir, label_path, video_name)
                save_path = os.path.join(args.save_dir, label_path, video_name)
                save_path = save_path.replace('.mp4', f'__{str(crop_type)}.npy')
                if os.path.exists(save_path):
                    print(f'{save_path} already exists')
                    continue

                video = cv2.VideoCapture(video_path)
                video_features, frames = [], []

                while True:
                    ret, frame = video.read()
                    if not ret:
                        break
                    frames.append(frame)
                    if len(frames) == chunk_size*args.batch_size:
                        frames = video_crop(np.array(frames), crop_type)
                        frames = frames.reshape(args.batch_size, args.chunk_size, 224, 224, 3)
                        
                        event = generate_event_image(frames, args.threshold)
                        event = torch.clamp(event, 0, args.clamp)
                        if event.numel() != 0:
                            event = event / event.max()
                        event = torch.stack([event, event, event], 1)

                        events = event_transform(event)
                        with torch.no_grad():
                            feature = model.encode_image(events.to(device))
                        video_features.append(feature.cpu())
                        frames = []

                if len(frames) > args.chunk_size:
                    frames = video_crop(np.array(frames), crop_type)
                    batch_size = len(frames) // args.chunk_size
                    frames = frames[:batch_size*args.chunk_size]
                    frames = frames.reshape(batch_size, args.chunk_size, 224, 224, 3)
                    
                    event = generate_event_image(frames, args.threshold)
                    event = event / 255.
                    event = torch.clamp(event, 0, args.clamp)
                    if event.numel() != 0:
                        event = event / event.max()
                    event = torch.stack([event, event, event], 1)
                    event_input = event_transform(event)
                    with torch.no_grad():
                        feature = model.encode_image(event_input.to(device))
                    video_features.append(feature.cpu())
                    
                video_features = torch.cat(video_features, dim=0).cpu().numpy()
                np.save(save_path, video_features)
                print(f'Saved to {save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract video features')
    parser.add_argument('--video_dir', type=str, default='/mnt/Data_1/MSAD/MSAD_blur/videos')
    parser.add_argument('--save_dir', type=str, default='/mnt/Data_1/msad_dataset/vitl/event_thr_10')
    parser.add_argument('--clip_ckpt', type=str, default='event_vitl.pt')
    parser.add_argument('--threshold', type=int, default=10)
    parser.add_argument('--chunk_size', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--clamp', type=float, default=16)

    args = parser.parse_args()
    main(args)