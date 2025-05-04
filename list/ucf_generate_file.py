import os
import csv
import argparse


def create_csv_from_directory(base_path, train_path, test_path):
    with open('ucf_test_list.txt', 'r') as f:
        test_list = f.readlines()
    test_list = [line.strip() for line in test_list]
    
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
                        base_file_name = base_file_name.split('_x')[0]

                        if base_file_name in test_list:
                            if '__5' in file_path:
                                test_writer.writerow([file_path, class_name])
                        else:
                            train_writer.writerow([file_path, class_name])
    print('CSV file created!')


parser = argparse.ArgumentParser(description='Create CSV file from directory')
parser.add_argument('--base_path', type=str, default='/media/Data1/UCFCrime_dataset/vitl/event_thr_10/', help='Base path of directory')
parser.add_argument('--train_save_path', type=str, default='ucf/rgb/vitl/train.csv', help='Output CSV file')
parser.add_argument('--test_save_path', type=str, default='ucf/rgb/vitl/test.csv', help='Output CSV file')
args = parser.parse_args()

base_path = args.base_path
train_path = args.train_save_path
test_path = args.test_save_path

create_csv_from_directory(base_path, train_path, test_path)