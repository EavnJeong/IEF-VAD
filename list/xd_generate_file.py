import os
import csv
import argparse

def create_csv_from_directory(base_path, train_path, test_path):
    test_dict = {}
    with open('xd_test_list.txt', 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                video_name = parts[0] 
                label = parts[1]
                test_dict[video_name] = label

    with open(train_path, mode='w', newline='', encoding='utf-8') as train_file, \
         open(test_path, mode='w', newline='', encoding='utf-8') as test_file:
        
        train_writer = csv.writer(train_file)
        test_writer = csv.writer(test_file)
        train_writer.writerow(['path', 'label'])
        test_writer.writerow(['path', 'label'])        
        
        for file_name in sorted(os.listdir(base_path)):
            if file_name.endswith('.npy'):
                file_path = os.path.join(base_path, file_name)
                base_file_name = os.path.splitext(file_name)[0][:-3]
                
                if base_file_name in test_dict:
                    if '__5' in file_path:
                        cls_name = file_path.split('__')[-2].split('_')[-1]
                        test_writer.writerow([file_path, cls_name])
                else:
                    cls_name = file_path.split('__')[-2].split('_')[-1]
                    train_writer.writerow([file_path, cls_name])
    print('CSV file created!')


parser = argparse.ArgumentParser(description='Create CSV file from directory')
parser.add_argument('--base_path', type=str, default='/media/Data1/xd_dataset/vitl/rgb/', help='Base path of directory')
parser.add_argument('--train_save_path', type=str, default='xd/rgb/vitl/train.csv', help='Output CSV file')
parser.add_argument('--test_save_path', type=str, default='xd/rgb/vitl/test.csv', help='Output CSV file')
args = parser.parse_args()

base_path = args.base_path
train_path = args.train_save_path
test_path = args.test_save_path

create_csv_from_directory(base_path, train_path, test_path)