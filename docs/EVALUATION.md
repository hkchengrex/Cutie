# Inference and Evaluation

We provide:

1. Pretrained Cutie model: <https://github.com/hkchengrex/Cutie/releases/tag/v1.0> or <https://drive.google.com/drive/folders/1E9ESHFlGU2KQkeRfH14kZzbdnwA0dH0f?usp=share_link>
2. Pre-computed outputs: <https://drive.google.com/drive/folders/1x-jf5GHB4hypU9cDZ0VSkMKGm8MR0eQJ?usp=share_link>

Note: the provided BURST visualizations were not done correctly. You can use `scripts/convert_burst_to_vos_train.py` to visualize from the prediction JSON instead.

## Preparation

1. Datasets should be placed out-of-source. [See TRAINING.md for the directory structure](https://github.com/hkchengrex/Cutie/blob/main/docs/TRAINING.md).

2. For the LVOS validation set, pre-process it by keeping only the first annotations:

```bash
python scripts/data/preprocess_lvos.py ../LVOS/valid/Annotations ../LVOS/valid/Annotations_first_only
```

3. Download the models that you need and place them in `./output`.

## Video Object Segmentation

```bash
python cutie/eval_vos.py dataset=[dataset] weights=[path to model file] model=[small/base]
```

- Possible options for `dataset`: see `cutie/config/eval_config.yaml`.
- To test on your own data, use `dataset=generic` and directly specify the image/mask directories. Example:

```bash
python cutie/eval_vos.py dataset=generic image_directory=examples/images mask_directory=examples/masks size=480
```

- To separate outputs from different models/settings, specify `exp_id=[some unique id]`.
- By default, the results are saved out-of-source in `../output/`.
- By default, we only use the first-frame annotation in the generic mode. Specify `--use_all_masks` to incorporate new objects (as in the YouTubeVOS dataset).
- To evaluate with the "plus" setting, specify `--config-name eval_plus_config.yaml` immediately after `python cutie/eval_vos.py` before other arguments.

To get quantitative results:

- DAVIS 2017 validation: [davis2017-evaluation](https://github.com/davisvideochallenge/davis2017-evaluation) or [vos-benchmark](https://github.com/hkchengrex/vos-benchmark).
- DAVIS 2016 validation: [vos-benchmark](https://github.com/hkchengrex/vos-benchmark).
- DAVIS 2017 test-dev: [CodaLab](https://codalab.lisn.upsaclay.fr/competitions/6812)
- YouTubeVOS 2018 validation: [CodaLab](https://codalab.lisn.upsaclay.fr/competitions/7685)
- YouTubeVOS 2019 validation: [CodaLab](https://codalab.lisn.upsaclay.fr/competitions/6066)
- LVOS val: [LVOS](https://github.com/LingyiHongfd/lvos-evaluation)
- LVOS test: [CodaLab](https://codalab.lisn.upsaclay.fr/competitions/8767)
- MOSE val: [CodaLab](https://codalab.lisn.upsaclay.fr/competitions/10703#participate-submit_results)
- BURST: [CodaLab](https://github.com/Ali2500/BURST-benchmark)

## Understanding Palette, Or, What is Going on with the Colored PNGs?

See <https://github.com/hkchengrex/XMem/blob/main/docs/PALETTE.md>.
