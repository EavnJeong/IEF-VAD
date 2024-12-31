from torch.utils.data import DataLoader
from data.dataset import UCF_Dataset


def get_loader(args, label_map):
    if args.dataset == 'ucfcrime':
        normal_dataset = UCF_Dataset(
            clip_dim=args.visual_length,
            file_path=args.train_list,
            test_mode=False,
            label_map=label_map,
            normal=True
        )
        abnormal_dataset = UCF_Dataset(
            clip_dim=args.visual_length,
            file_path=args.train_list,
            test_mode=False,
            label_map=label_map,
            normal=False
        )
        test_dataset = UCF_Dataset(
            clip_dim=args.visual_length,
            file_path=args.test_list,
            test_mode=True,
            label_map=label_map
        )
        
        normal_loader = DataLoader(
            normal_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True
        )
        abnormal_loader = DataLoader(
            abnormal_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True
        )    
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False
        )
    return normal_loader, abnormal_loader, test_loader