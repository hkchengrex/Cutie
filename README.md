# Putting the Object Back into Video Object Segmentation

![titlecard](https://imgur.com/6K7BgZ7.png)

[Ho Kei Cheng](https://hkchengrex.github.io/), [Seoung Wug Oh](https://sites.google.com/view/seoungwugoh/), [Brian Price](https://www.brianpricephd.com/), [Joon-Young Lee](https://joonyoung-cv.github.io/), [Alexander Schwing](https://www.alexander-schwing.de/)

University of Illinois Urbana-Champaign and Adobe

[[arXiV]](https://arxiv.org/abs/2310.12982) [[PDF]](https://arxiv.org/pdf/2310.12982.pdf) [[Project Page]](https://hkchengrex.github.io/Cutie/) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1yo43XTbjxuWA7XgCUO9qxAi7wBI6HzvP?usp=sharing)

## Highlight

Cutie is a video object segmentation framework -- a follow-up work of [XMem](https://github.com/hkchengrex/XMem) with better consistency, robustness, and speed.
This repository contains code for standard video object segmentation and a GUI tool for interactive video segmentation.
The GUI tool additionally contains the "permanent memory" (from [XMem++](https://github.com/max810/XMem2)) option for better controllability.

![overview](https://imgur.com/k84c965.jpg)

## Demo Video

https://github.com/hkchengrex/Cutie/assets/7107196/83a8abd5-369e-41a9-bb91-d9cc1289af70

Source: https://raw.githubusercontent.com/hkchengrex/Cutie/main/docs/sources.txt

## Installation

Tested on Ubuntu only.

**Prerequisite:**

- Python 3.8+
- PyTorch 1.12+ and corresponding torchvision

**Clone our repository:**

```bash
git clone https://github.com/hkchengrex/Cutie.git
```

**Install with pip:**

```bash
cd Cutie
pip install -e .
```

(If you encounter the File "setup.py" not found error, upgrade your pip with pip install --upgrade pip)

**Download the pretrained models:**

```python
python scripts/download_models.py
```

## Quick Start

Start the interactive demo with:

```bash
python interactive_demo.py --video ./examples/example.mp4 --num_objects 1
```

[See more instructions here](docs/INTERACTIVE.md).
If you are running this on a remote server, X11 forwarding is possible. Start by using `ssh -X`. Additional configurations might be needed but Google would be more helpful than me.

![demo](https://i.imgur.com/nqlYqTq.jpg)

(For single video evaluation, see the unofficial script `process_video.py` from https://github.com/hkchengrex/Cutie/pull/16)

## Training and Evaluation

1. [Running Cutie on video object segmentation data.](docs/EVALUATION.md)
2. [Training Cutie.](docs/TRAINING.md)

## Citation

```bibtex
@inproceedings{cheng2023putting,
  title={Putting the Object Back into Video Object Segmentation},
  author={Cheng, Ho Kei and Oh, Seoung Wug and Price, Brian and Lee, Joon-Young and Schwing, Alexander},
  booktitle={arXiv},
  year={2023}
}
```

## References

- The GUI tools uses [RITM](https://github.com/SamsungLabs/ritm_interactive_segmentation) for interactive image segmentation. This repository also contains a redistribution of their code in `gui/ritm`. That part of code follows RITM's license.

- For automatic video segmentation/integration with external detectors, see [DEVA](https://github.com/hkchengrex/Tracking-Anything-with-DEVA).

- The interactive demo is developed upon [IVS](https://github.com/seoungwugoh/ivs-demo), [MiVOS](https://github.com/hkchengrex/MiVOS), and [XMem](https://github.com/hkchengrex/XMem).

- We used [ProPainter](https://github.com/sczhou/ProPainter) in our video inpainting demo.

- Thanks to [RTIM](https://github.com/SamsungLabs/ritm_interactive_segmentation) and [XMem++](https://github.com/max810/XMem2) for making this possible.
