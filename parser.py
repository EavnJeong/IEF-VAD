def update_ucfcrime_args(args):
    ucfcrime_defaults = {
        'seed': 234,
        'visual_length': 256,
        'visual_head': 1,
        'visual_layers': 2,
        'attn_window': 8,
        'prompt_prefix': 10,
        'prompt_postfix': 10,
        'classes_num': 14,
        'max_epoch': 10,
        'model_path': 'model/model_ucf.pth',
        'use_checkpoint': False,
        'checkpoint_path': 'checkpoints/ucfcrime.pth',
        'batch_size': 64,
        
        'gt_segment_path': 'list/gt_segment_ucf.npy',
        'gt_label_path': 'list/gt_label_ucf.npy',
        'lr': 2e-5,
        'scheduler_rate': 0.1,
        'scheduler_milestones': [4, 8]
    }

    ucf_dataset = {
        'vitb_event_thr_25': {
            'train_list': 'list/event/vitb/thr_25/train.csv',
            'test_list': 'list/event/vitb/thr_25/test.csv',
            'gt_path': 'list/event/vitb/thr_25/gt.npy',
            'embed_dim': 512,
            'visual_width': 512
        },
        'vitb_event_thr_10': {
            'train_list': 'list/event/vitb/thr_10/train.csv',
            'test_list': 'list/event/vitb/thr_10/test.csv',
            'gt_path': 'list/event/vitb/thr_10/gt.npy',
            'embed_dim': 512,
            'visual_width': 512
        },
        'vitb_rgb': {
            'train_list': 'list/ucf_CLIP_rgb.csv',
            'test_list': 'list/ucf_CLIP_rgbtest.csv',
            'gt_path': 'list/gt_ucf.npy',
            'embed_dim': 512,
            'visual_width': 512
        },
        'vitl_event_thr_25': {
            'train_list': 'list/event/vitl/thr_25/train.csv',
            'test_list': 'list/event/vitl/thr_25/test.csv',
            'gt_path': 'list/event/vitl/thr_25/gt.npy',
            'embed_dim': 768,
            'visual_width': 768
        },
        'vitl_event_thr_10': {
            'train_list': 'list/event/vitl/thr_10/train.csv',
            'test_list': 'list/event/vitl/thr_10/test.csv',
            'gt_path': 'list/event/vitl/thr_10/gt.npy',
            'embed_dim': 768,
            'visual_width': 768
        },
        'vitl_rgb': {
            'train_list': 'list/ucf_CLIP_rgb.csv',
            'test_list': 'list/ucf_CLIP_rgbtest.csv',
            'gt_path': 'list/gt_ucf.npy',
            'embed_dim': 768,
            'visual_width': 768
        }
    }

    for key, value in ucfcrime_defaults.items():
        setattr(args, key, value)
    for key, value in ucf_dataset[args.ds].items():
        setattr(args, key, value)
    return args