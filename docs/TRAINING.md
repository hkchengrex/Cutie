# Training Cutie

## Setting Up Data

We put datasets out-of-source, as in XMem. You do not need BL30K. The directory structure should look like this:

```bash
├── Cutie
├── DAVIS
│   └── 2017
│       ├── test-dev
│       │   ├── Annotations
│       │   └── ...
│       └── trainval
│           ├── Annotations
│           └── ...
├── BURST
│   ├── frames
│   ├── val
│   │   ├── all_classes.json
│   │   └── first_frame_annotations.json
│   ├── train
│   │   └── train.json
│   └── train-vos
│       ├── JEPGImages
│       └── Annotations
├── static
│   ├── BIG_small
│   └── ...
└── YouTube
│   ├── all_frames
│   │   └── valid_all_frames
│   ├── train
│   └── valid
├── OVIS-VOS-train
│   ├── JPEGImages
│   └── Annotations
└── MOSE
    ├── JPEGImages
    └── Annotations
```

DEVA has a script for downloading some of these datasets: <https://github.com/hkchengrex/Tracking-Anything-with-DEVA/blob/main/docs/TRAINING.md>.

To generate `train-vos` for BURST, use the script `scripts/convert_burst_to_vos_train.py` which extracts masks from the JSON file into the DAVIS/YouTubeVOS format for training:
```bash
python scripts/convert_burst_to_vos_train.py --json_path ../BURST/train/train.json --frames_path ../BURST/frames/train --output_path ../BURST/train-vos
```

To generate OVIS-VOS-train, use something like https://github.com/youtubevos/vis2vos or download our preprocessed version from https://drive.google.com/uc?id=1AZPyyqVqOl6j8THgZ1UdNJY9R1VGEFrX.

Links to the datasets:
- DAVIS: https://davischallenge.org/
- YouTubeVOS: https://youtube-vos.org/
- BURST: https://github.com/Ali2500/BURST-benchmark
- MOSE: https://henghuiding.github.io/MOSE/
- LVOS: https://lingyihongfd.github.io/lvos.github.io/
- OVIS: https://songbai.site/ovis/

## Training Command

We trained with four A100 GPUs, which took around 30 hours.

```
OMP_NUM_THREADS=4 torchrun --master_port 25357 --nproc_per_node=4 cutie/train.py exp_id=[some unique id] model=[small/base] data=[base/with-mose/mega]
```

- Change `nproc_per_node` to change the number of GPUs.
- Prepend `CUDA_VISIBLE_DEVICES=...` if you want to use specific GPUs.
- Change `master_port` if you encounter port collision.
- `exp_id` is a unique experiment identifier that does not affect how the training is done.
- Models and visualizations will be saved in `./output/`.
- For pre-training only, specify `main_training.enabled=False`.
- For main training only, specify `pre_training.enabled=False`.
- To load a pre-trained model, e.g., to continue main training from the final model from pre-training, specify `weights=[path to the model]`.
