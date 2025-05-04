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
            line = line.split('  ')
            lbl_list[line[0]] = {
                'st': [int(line[2]), int(line[4])],
                'ed': [int(line[3]), int(line[5])]
            }
    gt = []
    for video_path in video_paths:
        frames = np.load(video_path)
        frame_label = np.zeros(frames.shape[0]*16)

        key = video_path.split('/')[-1].split('__')[0] + '.mp4'
        lbl = lbl_list[key]

        st = lbl['st']
        ed = lbl['ed']

        frame_label[int(st[0]) : int(ed[0])] = 1.0
        if st[1] != -1:
            frame_label[int(st[1]) : int(ed[1])] = 1.0

        print(f"Video: {video_path}")
        print(f"Start: {st[0]}, End: {ed[0]}, End-Start: {ed[0]-st[0]}")
        print(f"Start: {st[1]}, End: {ed[1]}, End-Start: {ed[1]-st[1]}")
        print(f"Total / Abnomal number : {frame_label.shape[0]} / {np.sum(frame_label)}")
        gt += list(frame_label)
    gt = np.array(gt)
    print(f"Total frames: {gt.shape[0]}")
    print(f"Total 1.0 number in gt: {np.sum(gt)}")
    np.save(args.save_path, gt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EVCLIP')
    parser.add_argument('--csv_path', default='msad/rgb/vitl/test.csv', type=str)
    parser.add_argument('--frame_lbl_txt', default='msad_annotations.txt')
    parser.add_argument('--save_path', default='msad/rgb/vitl/gt.npy')

    args = parser.parse_args()
    main(args)