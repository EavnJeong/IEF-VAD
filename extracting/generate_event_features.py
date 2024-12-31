import cv2
import os
import argparse
import numpy as np

import torch
import torchvision.transforms as transforms
from model.clip import clip
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


def video_crop(video_frame, type):
    l = video_frame.shape[0]
    new_frame = []
    for i in range(l):
        img = cv2.resize(video_frame[i], dsize=(340, 256))
        new_frame.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    #1
    img = np.array(new_frame)
    if type == 0:
        img = img[:, 16:240, 58:282, :]
    #2
    elif type == 1:
        img = img[:, :224, :224, :]
    #3
    elif type == 2:
        img = img[:, :224, -224:, :]
    #4
    elif type == 3:
        img = img[:, -224:, :224, :]
    #5
    elif type == 4:
        img = img[:, -224:, -224:, :]
    #6
    elif type == 5:
        img = img[:, 16:240, 58:282, :]
        for i in range(img.shape[0]):
            img[i] = cv2.flip(img[i], 1)
    #7
    elif type == 6:
        img = img[:, :224, :224, :]
        for i in range(img.shape[0]):
            img[i] = cv2.flip(img[i], 1)
    #8
    elif type == 7:
        img = img[:, :224, -224:, :]
        for i in range(img.shape[0]):
            img[i] = cv2.flip(img[i], 1)
    #9
    elif type == 8:
        img = img[:, -224:, :224, :]
        for i in range(img.shape[0]):
            img[i] = cv2.flip(img[i], 1)
    #10
    elif type == 9:
        img = img[:, -224:, -224:, :]
        for i in range(img.shape[0]):
            img[i] = cv2.flip(img[i], 1)
    return img


def generate_event_image(frames, threshold=25):
    # gray scale
    frames = frames.astype(np.uint8)
    
    num_frames, _, _, _ = frames.shape
    event_images = []
    
    for i in range(1, num_frames):
        diff = cv2.absdiff(frames[i], frames[i-1])
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        _, event_image = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)
        event_images.append(event_image.astype(np.float32))
    return torch.tensor(event_images).sum(dim=0)


def load_model(args, device):
    # model, preprocess = clip.load("ViT-B/16", device)
    model, preprocess = clip.load("ViT-B/32", device)
    
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
    
    transform = transforms.Compose([
        transforms.Resize((224, 224))
    ])

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    
    for label_path in os.listdir(args.video_dir):
        print(f'Processing {label_path}...')
        if not os.path.isdir(os.path.join(args.save_dir, label_path)):
            os.makedirs(os.path.join(args.save_dir, label_path))

# -------------------------------------------------------------------
        video_dir = os.path.join(args.video_dir, label_path)
        video_files = [
            (video_name, os.path.getsize(os.path.join(video_dir, video_name)))
            for video_name in os.listdir(video_dir)
        ]
        video_files = sorted(video_files, key=lambda x: x[1])
        sorted_video_names = [video_name for video_name, _ in video_files]
        for video_name in tqdm(sorted_video_names):
# -------------------------------------------------------------------

        # for video_name in tqdm(sorted(os.listdir(os.path.join(args.video_dir, label_path)))):
            for crop_type in range(10):
                video_path = os.path.join(args.video_dir, label_path, video_name)
                save_path = os.path.join(args.save_dir, label_path, video_name)
                save_path = save_path.replace('.mp4', f'__{str(crop_type)}.npy')
                if os.path.exists(save_path):
                    print(f'{save_path} already exists')
                    continue

                # Too expensive to process
                if 'Normal' in video_name:
                    exclude_numbers = [
                        '563', '548', '540', '541', '136', '137', '331', '530', 
                        '450', '666', '529', '449', '138', '547', '947', '472', 
                        '471', '425', '946', '633', '307', '308']
                    if any(number in video_name for number in exclude_numbers):
                        print(f'Skipping {video_name} due to high cost')
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
                
                events = []
                for idx in range(0, len(corp_video), chunk_size):
                    event = corp_video[idx:idx+chunk_size]
                    if event.shape[0] < chunk_size:
                        continue
                    event = generate_event_image(event, args.threshold)
                    event = event / 255.
                    event = torch.clamp(event, 0, args.clamp)
                    event = event / event.max()
                    event = torch.stack([event, event, event])
                    event = transform(event)
                    events.append(event)
                events = torch.stack(events)
                
                video_features = []
                with torch.no_grad():
                    for idx in range(0, len(events), batch_size):
                        event = events[idx : idx+batch_size]
                        feature = model.encode_image(event.to(device))
                        video_features.append(feature)
                        
                video_features = torch.cat(video_features, dim=0)
                video_features = video_features.cpu().numpy()
                np.save(save_path, video_features)
                print(f'Saved to {save_path}')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract video features')
    parser.add_argument('--video_dir', type=str, default='/mnt/Data_3/UCFCrime_raw/videos')
    parser.add_argument('--save_dir', type=str, default='/mnt/Data_3/UCFCrime_clip_vitb32_event')
    parser.add_argument('--clip_ckpt', type=str, default='checkpoints/eventclip_vitb.pt')
    parser.add_argument('--threshold', type=int, default=25)
    parser.add_argument('--chunk_size', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--clamp', type=float, default=10)

    args = parser.parse_args()
    main(args)