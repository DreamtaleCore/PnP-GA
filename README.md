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

---

**Results**

|   Input    | Method               |           $\mathcal{D}_E\rightarrow\mathcal{D}_M$            |           $\mathcal{D}_E\rightarrow\mathcal{D}_D$            |           $\mathcal{D}_G\rightarrow\mathcal{D}_M$            |           $\mathcal{D}_G\rightarrow\mathcal{D}_D$            |
| :--------: | -------------------- | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|    Face    | Baseline             |                            8.767                             |                            8.578                             |                            7.662                             |                            8.977                             |
|    Face    | Baseline + PnP-GA    | **5.529** <span style="color:darkred;font-size:13px;">$\blacktriangledown$36.9</span> | **5.867** <span style="color:darkred;font-size:13px;">$\blacktriangledown$31.6</span> | **6.176** <span style="color:darkred;font-size:13px;">$\blacktriangledown$19.4</span> | **7.922** <span style="color:darkred;font-size:13px;">$\blacktriangledown$11.8</span> |
|    Face    | ResNet50             |                            8.017                             |                            8.310                             |                            8.328                             |                            7.549                             |
|    Face    | ResNet50 + PnP-GA    | **6.000** <span style="color:darkred;font-size:13px;">$\blacktriangledown$25.2</span> | **6.172** <span style="color:darkred;font-size:13px;">$\blacktriangledown$25.7</span> | **5.739** <span style="color:darkred;font-size:13px;">$\blacktriangledown$31.1</span> | **7.042** <span style="color:darkred;font-size:13px;">$\blacktriangledown$6.7</span> |
|    Face    | SWCNN                |                            10.939                            |                            24.941                            |                            10.021                            |                            13.473                            |
|    Face    | SWCNN + PnP-GA       | **8.139** <span style="color:darkred;font-size:13px;">$\blacktriangledown$25.6</span> | **15.794** <span style="color:darkred;font-size:13px;">$\blacktriangledown$36.7</span> | **8.740** <span style="color:darkred;font-size:13px;">$\blacktriangledown$12.8</span> | **11.376** <span style="color:darkred;font-size:13px;">$\blacktriangledown$15.6</span> |
| Face + Eye | CA-Net               |                              --                              |                              --                              |                            21.276                            |                            30.890                            |
| Face + Eye | CA-Net + PnP-GA      |                              --                              |                              --                              | **17.597** <span style="color:darkred;font-size:13px;">$\blacktriangledown$17.3</span> | **16.999** <span style="color:darkred;font-size:13px;">$\blacktriangledown$44.9</span> |
| Face + Eye | Dilated-Net          |                              --                              |                              --                              |                            16.683                            |                            18.996                            |
| Face + Eye | Dilated-Net + PnP-GA |                              --                              |                              --                              | **15.461** <span style="color:darkred;font-size:13px;">$\blacktriangledown$7.3</span> | **16.835** <span style="color:darkred;font-size:13px;">$\blacktriangledown$11.4</span> |



This repository contains the official PyTorch implementation of the following paper:

> **Generalizing Gaze Estimation with Outlier-guided Collaborative Adaptation**<br>
Yunfei Liu, Ruicong Liu, Haofei Wang, Feng Lu<br> <!-- >  https://arxiv.org/abs/1911.09930  -->
> 
>**Abstract:**   Deep neural networks have significantly improved appearance-based gaze estimation accuracy. However, it still suffers from unsatisfactory performance when generalizing the trained model to new domains, e.g., unseen environments or persons. In this paper, we propose a plugand-play gaze adaptation framework (PnP-GA), which is an ensemble of networks that learn collaboratively with the guidance of outliers. Since our proposed framework does not require ground-truth labels in the target domain, the existing gaze estimation networks can be directly plugged into PnP-GA and generalize the algorithms to new domains. We test PnP-GA on four gaze domain adaptation tasks, ETH-to-MPII, ETH-to-EyeDiap, Gaze360-to-MPII, and Gaze360-to-EyeDiap. The experimental results demonstrate that the PnP-GA framework achieves considerable performance improvements of 36.9%, 31.6%, 19.4%, and 11.8% over the baseline system. The proposed framework also outperforms the state-of-the-art domain adaptation approaches on gaze domain adaptation tasks.

## Resources

Material related to our paper is available via the following links:

- Paper: coming soon!
- Project: https://liuyunfei.net/publication/iccv2021_pnp-ga/
- Code: https://github.com/DreamtaleCore/PnP-GA

## System requirements

* Only Linux is tested, Windows is under test.
* 64-bit Python 3.7 installation. 

## Playing with pre-trained networks and training

### Test and Train

Coming soon!

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
