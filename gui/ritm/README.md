## Reviving Iterative Training with Mask Guidance for Interactive Segmentation

<p align="center">
    <a href="https://paperswithcode.com/sota/interactive-segmentation-on-grabcut?p=reviving-iterative-training-with-mask">
        <img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/reviving-iterative-training-with-mask/interactive-segmentation-on-grabcut"/>
    </a>
    <a href="https://paperswithcode.com/sota/interactive-segmentation-on-berkeley?p=reviving-iterative-training-with-mask">
        <img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/reviving-iterative-training-with-mask/interactive-segmentation-on-berkeley"/>
    </a>
</p>

<p align="center">
  <img src="./assets/img/teaser.gif" alt="drawing", width="420"/>
  <img src="./assets/img/miou_berkeley.png" alt="drawing", width="400"/>
</p>

<p align="center">
    <a href="https://arxiv.org/abs/2102.06583">
        <img src="https://img.shields.io/badge/arXiv-2102.06583-b31b1b"/>
    </a>
    <a href="https://colab.research.google.com/github/saic-vul/ritm_interactive_segmentation/blob/master/notebooks/colab_test_any_model.ipynb">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
    </a>
    <a href="https://opensource.org/licenses/MIT">
        <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="The MIT License"/>
    </a>
</p>

This repository provides the source code for training and testing state-of-the-art click-based interactive segmentation models with the official PyTorch implementation of the following paper:

> **Reviving Iterative Training with Mask Guidance for Interactive Segmentation**<br>
> [Konstantin Sofiiuk](https://github.com/ksofiyuk), [Ilia Petrov](https://github.com/ptrvilya), [Anton Konushin](https://scholar.google.com/citations?user=ZT_k-wMAAAAJ) <br>
> Samsung Research<br>
> <https://arxiv.org/abs/2102.06583>
>
> **Abstract:** *Recent works on click-based interactive segmentation have demonstrated state-of-the-art results by
> using various inference-time optimization schemes. These methods are considerably more computationally expensive
> compared to feedforward approaches, as they require performing backward passes through a network during inference and
> are hard to deploy on mobile frameworks that usually support only forward passes. In this paper, we extensively
> evaluate various design choices for interactive segmentation and discover that new state-of-the-art results can be
> obtained without any additional optimization schemes. Thus, we propose a simple feedforward model for click-based
> interactive segmentation that employs the segmentation masks from previous steps. It allows not only to segment an
> entirely new object, but also to start with an external mask and correct it. When analyzing the performance of models
> trained on different datasets, we observe that the choice of a training dataset greatly impacts the quality of
> interactive segmentation. We find that the models trained on a combination of COCO and LVIS with diverse and
> high-quality annotations show performance superior to all existing models.*

## Setting up an environment

This framework is built using Python 3.6 and relies on the PyTorch 1.4.0+. The following command installs all
necessary packages:

```.bash
pip3 install -r requirements.txt
```

You can also use our [Dockerfile](./Dockerfile) to build a container with the configured environment.

If you want to run training or testing, you must configure the paths to the datasets in [config.yml](config.yml).

## Interactive Segmentation Demo

<p align="center">
  <img src="./assets/img/demo_gui.jpg" alt="drawing" width="99%"/>
</p>

The GUI is based on TkInter library and its Python bindings. You can try our interactive demo with any of the
[provided models](#pretrained-models). Our scripts automatically detect the architecture of the loaded model, just
specify the path to the corresponding checkpoint.

Examples of the script usage:

```.bash
# This command runs interactive demo with HRNet18 ITER-M model from cfg.INTERACTIVE_MODELS_PATH on GPU with id=0
# --checkpoint can be relative to cfg.INTERACTIVE_MODELS_PATH or absolute path to the checkpoint
python3 demo.py --checkpoint=hrnet18_cocolvis_itermask_3p --gpu=0

# This command runs interactive demo with HRNet18 ITER-M model from /home/demo/../weights/
# If you also do not have a lot of GPU memory, you can reduce --limit-longest-size (default=800)
python3 demo.py --checkpoint=/home/demo/fBRS/weights/hrnet18_cocolvis_itermask_3p --limit-longest-size=400

# You can try the demo in CPU only mode
python3 demo.py --checkpoint=hrnet18_cocolvis_itermask_3p --cpu
```

<details>
<summary><b>Running demo in docker</b></summary>
<pre><code># activate xhost
xhost +
docker run -v "$PWD":/tmp/ \
           -v /tmp/.X11-unix:/tmp/.X11-unix \
           -e DISPLAY=$DISPLAY &lt;id-or-tag-docker-built-image&gt; \
           python3 demo.py --checkpoint resnet34_dh128_sbd --cpu
</code></pre>
</details>

**Controls**:

| Key                                                           | Description                        |
| ------------------------------------------------------------- | ---------------------------------- |
| <kbd>Left Mouse Button</kbd>                                  | Place a positive click             |
| <kbd>Right Mouse Button</kbd>                                 | Place a negative click             |
| <kbd>Scroll Wheel</kbd>                                       | Zoom an image in and out           |
| <kbd>Right Mouse Button</kbd> + <br> <kbd>Move Mouse</kbd>    | Move an image                      |
| <kbd>Space</kbd>                                              | Finish the current object mask     |

<details>
<summary><b>Initializing the ITER-M models with an external segmentation mask</b></summary>
<p align="center">
  <img src="./assets/img/modifying_external_mask.jpg" alt="drawing" width="80%"/>
</p>
  
According to our paper, ITER-M models take an image, encoded user input, and a previous step mask as their input. Moreover, a user can initialize the model with an external mask before placing any clicks and correct this mask using the same interface.  As it turns out, our models successfully handle this situation and make it possible to change the mask.

To initialize any ITER-M model with an external mask use the "Load mask" button in the menu bar.
</details>

<details>
<summary><b>Interactive segmentation options</b></summary>
<ul>
    <li>ZoomIn (can be turned on/off using the checkbox)</li>
    <ul>
        <li><i>Skip clicks</i> - the number of clicks to skip before using ZoomIn.</li>
        <li><i>Target size</i> - ZoomIn crop is resized so its longer side matches this value (increase for large objects).</li>
        <li><i>Expand ratio</i> - object bbox is rescaled with this ratio before crop.</li>
        <li><i>Fixed crop</i> - ZoomIn crop is resized to (<i>Target size</i>, <i>Target size</i>).</li>
    </ul>
    <li>BRS parameters (BRS type can be changed using the dropdown menu)</li>
    <ul>
        <li><i>Network clicks</i> - the number of first clicks that are included in the network's input. Subsequent clicks are processed only using BRS  (NoBRS ignores this option).</li>
        <li><i>L-BFGS-B max iterations</i> - the maximum number of function evaluation for each step of optimization in BRS (increase for better accuracy and longer computational time for each click).</li>  
    </ul>
    <li>Visualisation parameters</li>
    <ul>
        <li><i>Prediction threshold</i> slider adjusts the threshold for binarization of probability map for the current object.</li>
        <li><i>Alpha blending coefficient</i> slider adjusts the intensity of all predicted masks.</li>
        <li><i>Visualisation click radius</i> slider adjusts the size of red and green dots depicting clicks.</li>
    </ul>
</ul>
</details>

## Datasets

We train all our models on SBD and COCO+LVIS and evaluate them on GrabCut, Berkeley, DAVIS, SBD and PascalVOC. We also provide links to additional datasets: ADE20k and OpenImages, that are used in ablation study.

| Dataset   |                      Description             |           Download Link              |
|-----------|----------------------------------------------|:------------------------------------:|
|ADE20k     |  22k images with 434k instances (total)      |  [official site][ADE20k]             |
|OpenImages |  944k images with 2.6M instances (total)     |  [official site][OpenImages]         |
|MS COCO    |  118k images with 1.2M instances (train)     |  [official site][MSCOCO]             |
|LVIS v1.0  |  100k images with 1.2M instances (total)     |  [official site][LVIS]               |
|COCO+LVIS* |  99k images with 1.5M instances (train)      |  [original LVIS images][LVIS] + <br> [our combined annotations][COCOLVIS_annotation] |
|SBD        |  8498 images with 20172 instances for (train)<br>2857 images with 6671 instances for (test) |[official site][SBD]|
|Grab Cut   |  50 images with one object each (test)       |  [GrabCut.zip (11 MB)][GrabCut]      |
|Berkeley   |  96 images with 100 instances (test)         |  [Berkeley.zip (7 MB)][Berkeley]     |
|DAVIS      |  345 images with one object each (test)      |  [DAVIS.zip (43 MB)][DAVIS]          |
|Pascal VOC |  1449 images with 3417 instances (validation)|  [official site][PascalVOC]          |
|COCO_MVal  |  800 images with 800 instances (test)        |  [COCO_MVal.zip (127 MB)][COCO_MVal] |

[ADE20k]: http://sceneparsing.csail.mit.edu/
[OpenImages]: https://storage.googleapis.com/openimages/web/download.html
[MSCOCO]: https://cocodataset.org/#download
[LVIS]: https://www.lvisdataset.org/dataset
[SBD]: http://home.bharathh.info/pubs/codes/SBD/download.html
[GrabCut]: https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/GrabCut.zip
[Berkeley]: https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/Berkeley.zip
[DAVIS]: https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/DAVIS.zip
[PascalVOC]: http://host.robots.ox.ac.uk/pascal/VOC/
[COCOLVIS_annotation]: https://github.com/saic-vul/ritm_interactive_segmentation/releases/download/v1.0/cocolvis_annotation.tar.gz
[COCO_MVal]: https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/COCO_MVal.zip

Don't forget to change the paths to the datasets in [config.yml](config.yml) after downloading and unpacking.

(*) To prepare COCO+LVIS, you need to download original LVIS v1.0, then download and unpack our
pre-processed annotations that are obtained by combining COCO and LVIS dataset into the folder with LVIS v1.0.

## Testing

### Pretrained models

We provide pretrained models with different backbones for interactive segmentation.

You can find model weights and evaluation results in the tables below:

<table>
    <thead align="center">
        <tr>
            <th rowspan="2"><span style="font-weight:bold">Train</span><br><span style="font-weight:bold">Dataset</span></th>
            <th rowspan="2">Model</th>
            <th colspan="2">GrabCut</th>
            <th>Berkeley</th>
            <th colspan="2">SBD</th>
            <th colspan="2">DAVIS</th>
            <th>Pascal<br>VOC</th>
            <th>COCO<br>MVal</th>
        </tr>
        <tr>
            <td>NoC<br>85%</td>
            <td>NoC<br>90%</td>
            <td>NoC<br>90%</td>
            <td>NoC<br>85%</td>
            <td>NoC<br>90%</td>
            <td>NoC<br>85%</td>
            <td>NoC<br>90%</td>
            <td>NoC<br>85%</td>
            <td>NoC<br>90%</td>
        </tr>
    </thead>
    <tbody align="center">
        <tr>
            <td rowspan="1">SBD</td>
            <td align="left"><a href="https://github.com/saic-vul/ritm_interactive_segmentation/releases/download/v1.0/sbd_h18_itermask.pth">HRNet18 IT-M<br>(38.8 MB)</a></td>
            <td>1.76</td>
            <td>2.04</td>
            <td>3.22</td>
            <td><b>3.39</b></td>
            <td><b>5.43</b></td>
            <td>4.94</td>
            <td>6.71</td>
            <td><ins>2.51</ins></td>
            <td>4.39</td>
        </tr>
        <tr>
            <td rowspan="4">COCO+<br>LVIS</td>
            <td align="left"><a href="https://github.com/saic-vul/ritm_interactive_segmentation/releases/download/v1.0/coco_lvis_h18_baseline.pth">HRNet18<br>(38.8 MB)</a></td>
            <td>1.54</td>
            <td>1.70</td>
            <td>2.48</td>
            <td>4.26</td>
            <td>6.86</td>
            <td>4.79</td>
            <td>6.00</td>
            <td>2.59</td>
            <td>3.58</td>
        </tr>
        <tr>
            <td align="left"><a href="https://github.com/saic-vul/ritm_interactive_segmentation/releases/download/v1.0/coco_lvis_h18s_itermask.pth">HRNet18s IT-M<br>(16.5 MB)</a></td>
            <td>1.54</td>
            <td>1.68</td>
            <td>2.60</td>
            <td>4.04</td>
            <td>6.48</td>
            <td>4.70</td>
            <td>5.98</td>
            <td>2.57</td>
            <td>3.33</td>
        </tr>
        <tr>
            <td align="left"><a href="https://github.com/saic-vul/ritm_interactive_segmentation/releases/download/v1.0/coco_lvis_h18_itermask.pth">HRNet18 IT-M<br>(38.8 MB)</a></td>
            <td><b>1.42</b></td>
            <td><b>1.54</b></td>
            <td><ins>2.26</ins></td>
            <td>3.80</td>
            <td>6.06</td>
            <td><ins>4.36</ins></td>
            <td><ins>5.74</ins></td>
            <td><b>2.28</b></td>
            <td><ins>2.98</ins></td>
        </tr>
        <tr>
            <td align="left"><a href="https://github.com/saic-vul/ritm_interactive_segmentation/releases/download/v1.0/coco_lvis_h32_itermask.pth">HRNet32 IT-M<br>(119 MB)</a></td>
            <td><ins>1.46</ins></td>
            <td><ins>1.56</ins></td>
            <td><b>2.10</b></td>
            <td><ins>3.59</ins></td>
            <td><ins>5.71</ins></td>
            <td><b>4.11</b></td>
            <td><b>5.34</b></td>
            <td>2.57</td>
            <td><b>2.97</b></td>
        </tr>
    </tbody>
</table>

### Evaluation

We provide the script to test all the presented models in all possible configurations on GrabCut, Berkeley, DAVIS,
Pascal VOC, and SBD. To test a model, you should download its weights and put them in `./weights` folder (you can
change this path in the [config.yml](config.yml), see `INTERACTIVE_MODELS_PATH` variable). To test any of our models,
just specify the path to the corresponding checkpoint. Our scripts automatically detect the architecture of the loaded model.

The following command runs the NoC evaluation on all test datasets (other options are displayed using '-h'):

```.bash
python3 scripts/evaluate_model.py <brs-mode> --checkpoint=<checkpoint-name>
```

Examples of the script usage:

```.bash
# This command evaluates HRNetV2-W18-C+OCR ITER-M model in NoBRS mode on all Datasets.
python3 scripts/evaluate_model.py NoBRS --checkpoint=hrnet18_cocolvis_itermask_3p

# This command evaluates HRNet-W18-C-Small-v2+OCR ITER-M model in f-BRS-B mode on all Datasets.
python3 scripts/evaluate_model.py f-BRS-B --checkpoint=hrnet18s_cocolvis_itermask_3p

# This command evaluates HRNetV2-W18-C+OCR ITER-M model in NoBRS mode on GrabCut and Berkeley datasets.
python3 scripts/evaluate_model.py NoBRS --checkpoint=hrnet18_cocolvis_itermask_3p --datasets=GrabCut,Berkeley
```

### Jupyter notebook

You can also interactively experiment with our models using [test_any_model.ipynb](./notebooks/test_any_model.ipynb) Jupyter notebook.

## Training

We provide the scripts for training our models on the SBD dataset. You can start training with the following commands:

```.bash
# ResNet-34 non-iterative baseline model
python3 train.py models/noniterative_baselines/r34_dh128_cocolvis.py --gpus=0 --workers=4 --exp-name=first-try

# HRNet-W18-C-Small-v2+OCR ITER-M model
python3 train.py models/iter_mask/hrnet18s_cocolvis_itermask_3p.py --gpus=0 --workers=4 --exp-name=first-try

# HRNetV2-W18-C+OCR ITER-M model
python3 train.py models/iter_mask/hrnet18_cocolvis_itermask_3p.py --gpus=0,1 --workers=6 --exp-name=first-try

# HRNetV2-W32-C+OCR ITER-M model
python3 train.py models/iter_mask/hrnet32_cocolvis_itermask_3p.py --gpus=0,1,2,3 --workers=12 --exp-name=first-try
```

For each experiment, a separate folder is created in the `./experiments` with Tensorboard logs, text logs,
visualization and checkpoints. You can specify another path in the [config.yml](config.yml) (see `EXPS_PATH`
variable).

Please note that we trained ResNet-34 and HRNet-18s on 1 GPU, HRNet-18 on 2 GPUs, HRNet-32 on 4 GPUs
(we used Nvidia Tesla P40 for training). To train on a different GPU you should adjust the batch size using the command
line argument `--batch-size` or change the default value in a model script.

We used the pre-trained HRNetV2 models from [the official repository](https://github.com/HRNet/HRNet-Image-Classification).
If you want to train interactive segmentation with these models, you need to download the weights and specify the paths to
them in [config.yml](config.yml).

## License

The code is released under the MIT License. It is a short, permissive software license. Basically, you can do whatever you want as long as you include the original copyright and license notice in any copy of the software/source.

## Citation

If you find this work is useful for your research, please cite our papers:

```bibtex
@inproceedings{ritm2022,
  title={Reviving iterative training with mask guidance for interactive segmentation},
  author={Sofiiuk, Konstantin and Petrov, Ilya A and Konushin, Anton},
  booktitle={2022 IEEE International Conference on Image Processing (ICIP)},
  pages={3141--3145},
  year={2022},
  organization={IEEE}
}

@inproceedings{fbrs2020,
   title={f-brs: Rethinking backpropagating refinement for interactive segmentation},
   author={Sofiiuk, Konstantin and Petrov, Ilia and Barinova, Olga and Konushin, Anton},
   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
   pages={8623--8632},
   year={2020}
}
```
