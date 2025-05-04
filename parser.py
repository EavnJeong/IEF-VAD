def update_ucf_args(args):
    ucfcrime_defaults = {
        'visual_length': 256,
        'visual_head': 8,
        'visual_layers': 2,
        'attn_window': 8,
        'prompt_prefix': 10,
        'prompt_postfix': 10,
        'classes_num': 14,
        'model_path': 'model/model_ucf.pth',
        'use_checkpoint': False,
        'checkpoint_path': 'checkpoints/ucfcrime.pth',
        'gt_segment_path': 'list/gt_segment_ucf.npy',
        'gt_label_path': 'list/gt_label_ucf.npy',
        'scheduler_rate': 0.1,
        'scheduler_milestones': [4, 8]
    }

    ucf_dataset = {
        'vitb_rgb': {
            'train_list': 'list/ucf/rgb/vitb/train.csv',
            'test_list': 'list/ucf/rgb/vitb/test.csv',
            'gt_path': 'list/ucf/rgb/vitb/gt.npy',
            'embed_dim': 512,
            'visual_width': 512
        },
        'vitl_rgb': {
            'train_list': 'list/ucf/rgb/vitl/train.csv',
            'test_list': 'list/ucf/rgb/vitl/test.csv',
            'gt_path': 'list/ucf/rgb/vitl/gt.npy',
            'embed_dim': 768,
            'visual_width': 768
        },
    }
    for key, value in ucfcrime_defaults.items():
        setattr(args, key, value)
    for key, value in ucf_dataset[args.ds].items():
        setattr(args, key, value)
    return args


def update_xd_args(args):
    xd_defaults = {
        'visual_length': 256,
        'attn_window': 64,
        'prompt_prefix': 10,
        'prompt_postfix': 10,
        'classes_num': 7,
        'model_path': 'model/model_xd.pth',
        'use_checkpoint': False,
        'checkpoint_path': 'checkpoints/xd.pth',
        'gt_segment_path': 'list/gt_segment_ucf.npy',
        'gt_label_path': 'list/gt_label_ucf.npy',
        'scheduler_rate': 0.1,
        'scheduler_milestones': [2, 6, 10],
        'print_steps': 512,
        'batch_size': 2,
        # 'lr': 1e-5,
    }

    xd_dataset = {
        'vitl_rgb': {
            'train_list': 'list/xd/rgb/vitl/train.csv',
            'test_list': 'list/xd/rgb/vitl/test.csv',
            'gt_path': 'list/xd/rgb/vitl/gt.npy',
            'embed_dim': 768,
            'visual_width': 768       
        }
    }
    for key, value in xd_defaults.items():
        setattr(args, key, value)
    for key, value in xd_dataset[args.ds].items():
        setattr(args, key, value)
    return args


def update_shang_args(args):
    shang_defaults = {
        'visual_length': 256,
        'visual_head': 8,
        'visual_layers': 2,
        'attn_window': 64,
        'prompt_prefix': 10,
        'prompt_postfix': 10,
        'classes_num': 16,
        'model_path': 'model/model_shang.pth',
        'use_checkpoint': False,
        'checkpoint_path': 'checkpoints/shang.pth',
        'gt_segment_path': 'list/gt_segment_ucf.npy',
        'gt_label_path': 'list/gt_label_ucf.npy',
        'scheduler_rate': 0.5,
        'scheduler_milestones': [8, 15],
        'print_steps': 512,
        'vis_steps': 1150,
        'batch_size': 8,
        'max_epoch': 20,
        'lr': 1e-4,
        'num_refinement_steps': 10,
    }

    shang_dataset = {
        'vitl_rgb': {
            'train_list': 'list/shang/rgb/vitl/train.csv',
            'test_list': 'list/shang/rgb/vitl/test.csv',
            'gt_path': 'list/shang/rgb/vitl/gt.npy',
            'embed_dim': 768,
            'visual_width': 768,      
        }
    }
    for key, value in shang_defaults.items():
        setattr(args, key, value)
    for key, value in shang_dataset[args.ds].items():
        setattr(args, key, value)
    return args


def update_msad_args(args):
    msad_defaults = {
        'visual_length': 256,
        'visual_head': 8,
        'visual_layers': 2,
        'attn_window': 8,
        'prompt_prefix': 10,
        'prompt_postfix': 10,
        'classes_num': 14,
        'model_path': 'model/model_msad.pth',
        'use_checkpoint': False,
        'checkpoint_path': 'checkpoints/msad.pth',
        'gt_segment_path': 'list/gt_segment_ucf.npy',
        'gt_label_path': 'list/gt_label_ucf.npy',
        'scheduler_rate': 0.1,
        'scheduler_milestones': [4, 8]
    }

    msad_dataset = {
        'vitl_rgb': {
            'train_list': 'list/msad/rgb/vitl/train.csv',
            'test_list': 'list/msad/rgb/vitl/test.csv',
            'gt_path': 'list/msad/rgb/vitl/gt.npy',
            'embed_dim': 768,
            'visual_width': 768
        }
    }
    for key, value in msad_defaults.items():
        setattr(args, key, value)
    for key, value in msad_dataset[args.ds].items():
        setattr(args, key, value)
    return args