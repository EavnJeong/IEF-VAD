import os
import csv
import argparse


def create_csv_from_directory(base_path, train_path, test_path):
    with open('shang_test_list.txt', 'r') as f:
        test_list = f.readlines()
    test_list = [line.strip() for line in test_list]

    train_num, test_num = 0, 0
    
    with open(train_path, mode='w', newline='') as train_file, \
            open(test_path, mode='w', newline='') as test_file:

        train_writer = csv.writer(train_file)
        test_writer = csv.writer(test_file)
        train_writer.writerow(['path', 'label'])
        test_writer.writerow(['path', 'label'])        
        
        for class_name in sorted(os.listdir(base_path)):
            class_dir = os.path.join(base_path, class_name)
            
            if os.path.isdir(class_dir):
                for file_name in sorted(os.listdir(class_dir)):
                    if file_name.endswith('.npy'):
                        file_path = os.path.join(class_dir, file_name)

                        base_file_name = os.path.splitext(file_name)[0]
                        base_file_name = base_file_name.split('__')[0]

                        if base_file_name in test_list:
                            if '__5' in file_path:
                                test_writer.writerow([file_path, class_name])
                                test_num += 1
                        else:
                            train_writer.writerow([file_path, class_name])
                            train_num += 1
    print(f'Train: {train_num/10}, Test: {test_num}')
    print('CSV file created!')


parser = argparse.ArgumentParser(description='Create CSV file from directory')
parser.add_argument('--base_path', type=str, default='/media/Data1/shang_dataset/vitl/rgb', help='Base path of directory')
parser.add_argument('--train_save_path', type=str, default='shang/rgb/vitl/train.csv', help='Output CSV file')
parser.add_argument('--test_save_path', type=str, default='shang/rgb/vitl/test.csv', help='Output CSV file')
args = parser.parse_args()

base_path = args.base_path
train_path = args.train_save_path
test_path = args.test_save_path

create_csv_from_directory(base_path, train_path, test_path)