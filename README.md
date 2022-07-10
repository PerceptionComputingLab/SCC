# SCC

Code for our paper ["A Contrastive Consistency Semi-supervised Left Atrium Segmentation Model"](https://authors.elsevier.com/sd/article/S0895-6111(22)00065-9). 

- Proposed a class-aware semi-supervised 3D left atrium segmentaion model.
- Proposed a contrastive consistency loss function in class-level.

The pipeline of our method is shown below:

<p align="center">
    <img src="images/framework.png" width="750" height="500" caption='The framework of the proposed left atrium semi-supervised segmentation model. There are two sub-models: the segmentation model E2DNet shown in the black dash box and the classification model  shown in the blue dash box. The  indicates the segmentation loss function for the labeled data. The class-vector space is shown in the blue circle, as  indicate class-vectors of the labeled and unlabeled data respectively. Class-vectors with different colors indicate different classes.'> 



## Requirements

Python 3.6.2

Pytorch 1.7

CUDA 11.2


## Training

**Run**

```python
train: python train_LA_semi_contrastive.py
test: python test_LA_semi_contrast.py
```
## Cite

Please consider citing this project in your publications if it helps your research. The following is a BibTeX reference. The BibTeX entry requires the url LaTeX package.

    @article{LIU2022102092,
        title = {A contrastive consistency semi-supervised left atrium segmentation model},
        journal = {Computerized Medical Imaging and Graphics},
        volume = {99},
        pages = {102092},
        year = {2022},
        issn = {0895-6111},
        doi = {https://doi.org/10.1016/j.compmedimag.2022.102092},
        url = {https://www.sciencedirect.com/science/article/pii/S0895611122000659},
    ｝


## Acknowledgment
The development of this project is based on [SegWithDistMap](https://github.com/JunMa11/SegWithDistMap)
