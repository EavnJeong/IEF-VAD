# Uncertainty-Weighted Image-Event Multimodal Fusion for Video Anomaly Detection

![Architecture](figure/fig1.png)

## Environment

Python==3.9.12

    conda env create -f env.yaml


## Dataset Prepare

### UCF-Crime  ([LINK](https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/))

    videos
       ├── Abuse
       │   ├── Abuse001_x264.mp4
       │   ├── Abuse002_x264.mp4
       │   ├── Abuse003_x264.mp4
       │   ...

### XD-Violence ([LINK](https://roc-ng.github.io/XD-Violence/))

    videos
       ├── A.Beautiful.Mind.2001__#00-01-45_00-02-50_label_A.mp4
       ├── A.Beautiful.Mind.2001__#00-03-00_00-04-05_label_A.mp4
       ├── A.Beautiful.Mind.2001__#00-04-20_00-05-35_label_A.mp4
       │   ...

### ShanghaiTech ([LINK](https://svip-lab.github.io/dataset/campus_dataset.html))

    videos
       ├── car
       │   ├── 01_0130.avi
       │   ├── 01_0135.avi
       │   ├── 01_0136.avi
       │   ...

### MSAD ([LINK](https://msad-dataset.github.io/))

    videos
       ├── Assault
       │   ├── Assault_10.mp4
       │   ├── Assault_11.mp4
       │   ├── Assault_12.mp4
       │   ...


## Embedding Extract
Extract Image embbeddings and Synthetic Event embeddings with CLIP weights

### CLIP Event weights is possible to use in ([HERE](https://github.com/EavnJeong/Event_Modality_Application))
also possible to download ([HuggingFace](https://huggingface.co/Eavn/event-clip))

### Embeddings ([LINK](https://drive.google.com/drive/folders/11b6tiAa8Lsbd9hvO1F1U9oEdWOGja89H?usp=sharing))

---
    cd extracting
    
    # Extract frame from video and preproessing by CLIP
    python ucf_gen_ima.py --video_dir ".../videos" --save_dir "e.g., .../rgb"
    
    # Extract Synthetic event and preprocessing by CLIP
    python ucf_gen_event.py --save_dir ".../rgb" --save_dir "e.g., .../event_thr_10" --clip_ckpt "event clip checkpoint" 

## Config 

Generate training list and ground truth label(0, 1) for anomaly detection for generated embeddings.

**It is already generate in ./list, you can skip it, just change the csv file contents**

    cd list

    # Generate Train, Test csv files including .npy path
    python ucf_generate_file.py --base_path "UCF embedding path .../rgb" --train_save_path "ucf/rgb/vitl/train.csv" --test_save_path "ucf/rgb/vitl/test.csv"

    # Generate frame based anomaly label(0, 1) sequences.
    python ucf_generate_gt.py --csv_path "test.csv path generate above." --save_path "gt generated path"

## Training
![Classwise Anomaly score](figure/fig2.png)

DATASET in ['ucfcrime', 'xd', 'shang', 'msad']

    python main.py --dataset "DATASET"


## Test
![Uncertainty Change](figure/fig3.png)

### Checkpoint files in [LINK](https://drive.google.com/drive/folders/12pf7kZuICRlgzE9WyeXrKZi_B5RLoqxh?usp=sharing).

    # Performance and visualization
    python test.py --ckpt_path checkpoints/ucf/ucf.pth --exp_name ucf
    python test.py --dataset shang --ckpt_path checkpoints/shang/shang.pth --exp_name shang
    python test.py --dataset xd --ckpt_path checkpoints/xd/xd.pth --exp_name xd
    python test.py --dataset msad --ckpt_path checkpoints/msad/msad.pth --exp_name msad

    # Uncertainty explore
    python test2.py --ckpt_path checkpoints/ucf/ucf.pth
    python test2.py --dataset shang --ckpt_path checkpoints/shang/shang.pth
    python test2.py --dataset xd --ckpt_path checkpoints/xd/xd.pth
    python test2.py --dataset msad --ckpt_path checkpoints/msad/msad.pth

## 📖 Citation

If you find our work useful, please cite:

    @article{jeong2025uncertainty,
      title={Uncertainty-Weighted Image-Event Multimodal Fusion for Video Anomaly Detection},
      author={Jeong, Sungheon and Park, Jihong and Imani, Mohsen},
      journal={arXiv preprint arXiv:2505.02393},
      year={2025}
    }

## Acknowledgement

This project is based on [VADCLIP](https://github.com/nwpu-zxr/VadCLIP).  
Special thanks to the original authors for their great work.  
