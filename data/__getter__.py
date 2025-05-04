from torch.utils.data import DataLoader
from data.dataset import UCF_Dataset, XD_Dataset, Shang_Dataset


def get_ucf_dataset(args, label_map):
    normal_dataset = UCF_Dataset(
        clip_dim=args.visual_length, file_path=args.train_list,
        test_mode=False, label_map=label_map, normal=True
    )
    abnormal_dataset = UCF_Dataset(
        clip_dim=args.visual_length, file_path=args.train_list,
        test_mode=False, label_map=label_map, normal=False
    )
    test_dataset = UCF_Dataset(
        clip_dim=args.visual_length, file_path=args.test_list,
        test_mode=True, label_map=label_map
    )
    
    normal_loader = DataLoader(
        normal_dataset, batch_size=args.batch_size,
        shuffle=True, drop_last=True
    )
    abnormal_loader = DataLoader(
        abnormal_dataset, batch_size=args.batch_size,
        shuffle=True, drop_last=True
    )    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return normal_loader, abnormal_loader, test_loader


def get_xd_dataset(args, label_map):
    train_dataset = XD_Dataset(
        args.visual_length, args.train_list, False, label_map)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = XD_Dataset(
        args.visual_length, args.test_list, True, label_map)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False)
    return train_loader, test_loader


def get_shang_dataset(args, label_map):
    normal_dataset = Shang_Dataset(
        clip_dim=args.visual_length, file_path=args.train_list,
        test_mode=False, label_map=label_map, normal=True
    )
    abnormal_dataset = Shang_Dataset(
        clip_dim=args.visual_length, file_path=args.train_list,
        test_mode=False, label_map=label_map, normal=False
    )
    test_dataset = Shang_Dataset(
        clip_dim=args.visual_length, file_path=args.test_list,
        test_mode=True, label_map=label_map
    )

    normal_loader = DataLoader(
        normal_dataset, batch_size=args.batch_size,
        shuffle=True, drop_last=True
    )
    abnormal_loader = DataLoader(
        abnormal_dataset, batch_size=args.batch_size,
        shuffle=True, drop_last=True
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return normal_loader, abnormal_loader, test_loader


def get_loader(args, label_map):
    if args.dataset == 'ucfcrime':
        normal_loader, abnormal_loader, test_loader = get_ucf_dataset(args, label_map)
        return normal_loader, abnormal_loader, test_loader
    elif args.dataset == 'xd':
        train_loader, test_loader = get_xd_dataset(args, label_map)
        return train_loader, test_loader
    elif args.dataset == 'shang':
        normal_loader, abnormal_loader, test_loader = get_shang_dataset(args, label_map)
        return normal_loader, abnormal_loader, test_loader
    elif args.dataset == 'msad':
        normal_loader, abnormal_loader, test_loader = get_ucf_dataset(args, label_map)
        return normal_loader, abnormal_loader, test_loader
    else:
        raise ValueError('Dataset not supported')