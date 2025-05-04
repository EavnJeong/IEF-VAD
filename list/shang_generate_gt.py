import argparse
import pandas as pd
import cv2
import numpy as np

def main(args):
    df = pd.read_csv(args.csv_path)
    video_paths = df.iloc[:, 0]
    
    lbl_list = {}
    with open(args.frame_lbl_txt, 'r') as f:
        lines = f.readlines()
        for line in lines:
            tokens = line.split()  
            intervals = []
            for i in range(2, len(tokens), 2):
                st = int(tokens[i])
                ed = int(tokens[i+1])
                if st != -1 and ed != -1:
                    intervals.append((st, ed))
            lbl_list[tokens[0]] = intervals
    gt = []
    for video_path in video_paths:
        frames = np.load(video_path)
        frame_label = np.zeros(frames.shape[0] * 16)
        key = video_path.split('/')[-1].split('__')[0] + '.avi'
        intervals = lbl_list.get(key, [])
        for (start, end) in intervals:
            frame_label[start:end] = 1.0
        print(f"Video: {video_path}")
        for (start, end) in intervals:
            print(f"Abnormal interval: start={start}, end={end}, length={end - start}")
        print(f"Total frames for video / Abnormal frames: {frame_label.shape[0]} / {np.sum(frame_label)}")
        gt += list(frame_label)
    gt = np.array(gt)
    print(f"Total frames: {gt.shape[0]}")
    print(f"Total abnormal (1.0) count in gt: {np.sum(gt)}")
    np.save(args.save_path, gt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EVCLIP')
    parser.add_argument('--csv_path', default='shang/rgb/vitl/test.csv', type=str)
    parser.add_argument('--frame_lbl_txt', default='shang_annotations.txt')
    parser.add_argument('--save_path', default='shang/rgb/vitl/gt.npy')

    args = parser.parse_args()
    main(args)