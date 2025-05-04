import argparse
import pandas as pd
import numpy as np

def main(args):
    df = pd.read_csv(args.csv_path)
    video_paths = df.iloc[:, 0]

    anno_dict = {}
    with open(args.frame_lbl_txt, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 6:
                continue  
            video_filename = tokens[0]
            video_class = tokens[1]
            intervals = []
            for i in range(2, len(tokens), 2):
                if i+1 >= len(tokens):
                    break
                start = int(tokens[i])
                end = int(tokens[i+1])
                if start != -1 and end != -1:
                    intervals.append((start, end))
            anno_dict[video_filename] = {
                'class': video_class,
                'intervals': intervals
            }
    gt = []
    for video_path in video_paths:
        frames = np.load(video_path)
        frame_label = np.zeros(frames.shape[0] * 16)

        base_file = video_path.split('/')[-1]
        base_file_name = base_file.rsplit('.', 1)[0]
        key = base_file_name[:-3] + '.mp4'

        if key not in anno_dict:
            print(f"Warning: {key} not found in annotation file.")
            continue

        anno = anno_dict[key]
        print("=========================================")
        print(f"Video: {video_path}")
        print(f"Class: {anno['class']}")
        for idx, (start, end) in enumerate(anno['intervals']):
            print(f"Interval {idx+1}: Start {start}, End {end}, Duration {end - start}")
            frame_label[start:end] = 1.0

        print(f"Total frames: {frame_label.shape[0]}, Abnormal frames: {np.sum(frame_label)}")
        gt += list(frame_label)

    gt = np.array(gt)
    print(f"Total frames in gt: {gt.shape[0]}")
    print(f"Total abnormal frames in gt: {np.sum(gt)}")
    np.save(args.save_path, gt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EVCLIP')
    parser.add_argument('--csv_path', default='xd/rgb/vitl/test.csv', type=str, help='CSV file path containing video (npy) paths')
    parser.add_argument('--frame_lbl_txt', default='xd_annotation.txt', help='Text file containing frame annotations')
    parser.add_argument('--save_path', default='xd/rgb/vitl/gt.npy', help='Output path for the saved ground truth numpy array')

    args = parser.parse_args()
    main(args)