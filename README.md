# Generalizing Gaze Estimation with Outlier-guided Collaborative Adaptation

![Python 3.7](https://img.shields.io/badge/python-3.7-DodgerBlue.svg?style=plastic)
![Pytorch 1.10](https://img.shields.io/badge/pytorch-1.2.0-DodgerBlue.svg?style=plastic)
![CUDA 10.0](https://img.shields.io/badge/cuda-10.0-DodgerBlue.svg?style=plastic)
![License CC BY-NC](https://img.shields.io/badge/license-CC_BY--NC-DodgerBlue.svg?style=plastic)

 Our paper is accepted by **ICCV2021**. 

<div align=center>  <img src="figures/teaser.png" alt="Teaser" width="500" align="bottom" /> </div>

**Picture:**  *Overview of the proposed Plug-and-Play (PnP) adaption framework for generalizing gaze estimation to a new domain.*

<div align=center>  <img src="./figures/main_image.png" alt="Main image" width="800" align="center" /> </div>

**Picture:**  *The proposed architecture.*





This repository contains the official PyTorch implementation of the following paper:

> **Generalizing Gaze Estimation with Outlier-guided Collaborative Adaptation**<br>
Yunfei Liu, Ruicong Liu, Haofei Wang, Feng Lu<br> <!-- >  https://arxiv.org/abs/1911.09930  -->
> 
>**Abstract:**   Deep neural networks have significantly improved appearance-based gaze estimation accuracy. However, it still suffers from unsatisfactory performance when generalizing the trained model to new domains, e.g., unseen environments or persons. In this paper, we propose a plugand-play gaze adaptation framework (PnP-GA), which is an ensemble of networks that learn collaboratively with the guidance of outliers. Since our proposed framework does not require ground-truth labels in the target domain, the existing gaze estimation networks can be directly plugged into PnP-GA and generalize the algorithms to new domains. We test PnP-GA on four gaze domain adaptation tasks, ETH-to-MPII, ETH-to-EyeDiap, Gaze360-to-MPII, and Gaze360-to-EyeDiap. The experimental results demonstrate that the PnP-GA framework achieves considerable performance improvements of 36.9%, 31.6%, 19.4%, and 11.8% over the baseline system. The proposed framework also outperforms the state-of-the-art domain adaptation approaches on gaze domain adaptation tasks.

## Resources

Material related to our paper is available via the following links:

- Paper: comming soon!
- Project: https://liuyunfei.net/publication/iccv2021_pnp-ga/
- Code: https://github.com/DreamtaleCore/PnP-GA

## System requirements

* Only Linux is tested, Windows is under test.
* 64-bit Python 3.7 installation. 

## Playing with pre-trained networks and training

### Test and Train

Comming soon!

## Citation

If you find this work or code is helpful in your research, please cite:

```latex
@inproceedings{liu2021PnP_GA,
  title={Generalizing Gaze Estimation with Outlier-guided Collaborative Adaptation},
  author={Liu, Yunfei and Liu, Ruicong and Wang, Haofei and Lu, Feng},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2021}
}
```

## Contact

If you have any questions, feel free to E-mail me via: `lyunfei(at)buaa.edu.cn`
{"mode":"full","isActive":false}