# U-Net-based Models for Skin Lesion Segmentation: More Attention and Augmentation
This repository is associated with [this paper](https://arxiv.org/abs/2210.16399). In this work, ten [models](https://github.com/pooya-mohammadi/unet-skin-cancer/tree/master/models) and four augmentation configurations are trained on the ISIC 2016 dataset. Each augmentation configuration is a combination of [CutMix](https://github.com/pooya-mohammadi/unet-skin-cancer/blob/master/utils/combine_images.py), [Mosaic](https://github.com/pooya-mohammadi/unet-skin-cancer/blob/master/utils/mosaic.py), [Hair-Augmentation](https://github.com/pooya-mohammadi/unet-skin-cancer/blob/297e040ca5839a5578586d4c51acb649e66d5375/utils/hair_augmentation.py#L43), [Hair-Removal](https://github.com/pooya-mohammadi/unet-skin-cancer/blob/297e040ca5839a5578586d4c51acb649e66d5375/utils/hair_augmentation.py#L80), and basic geometrical augmentations. 

The performance and overfitting are compared utilizing five metrics. Our results show that the [UR50](https://github.com/pooya-mohammadi/unet-skin-cancer/blob/master/models/unet_res50.py) and the [R2U](https://github.com/pooya-mohammadi/unet-skin-cancer/blob/master/models/r2unet.py) have the highest metrics value, along with two data augmentation scenarios. We also investigate [CBAM](https://github.com/pooya-mohammadi/unet-skin-cancer/blob/master/utils/cbam.py) and [AG blocks](https://github.com/pooya-mohammadi/unet-skin-cancer/blob/master/utils/attentionGate.py) in the U-Net architecture, which enhances segmentation performance at a meager computational cost. In addition, we propose [UPCG](https://github.com/pooya-mohammadi/unet-skin-cancer/blob/master/models/unet_pyramid_cbam_gate.py), in which pyramid, AG, and CBAM blocks are used in a sequence in the basic U-Net architecture, which significantly surpasses the results of using the two individually and got the better of [UR50](https://github.com/pooya-mohammadi/unet-skin-cancer/blob/master/models/unet_res50.py) and [DU](https://github.com/pooya-mohammadi/unet-skin-cancer/blob/master/models/doubleunet.py) models in some situations. We also show that [R2UC](https://github.com/pooya-mohammadi/unet-skin-cancer/blob/master/models/r2unet_cbam.py) model can be more successful than its parent, [R2U](https://github.com/pooya-mohammadi/unet-skin-cancer/blob/master/models/r2unet.py), and other attention-less models in the boundaries and more difficult scenarios.

Our work is hereby available for reproduction purposes. Please follow the succeeding instructions.

## Download dataset
To download & preprocess the data, run the following module. In case of `URL fetch failure` turn on your VPN.
```
python data/preprocessing.py --train_path data/train --test_path data/test --mask_train_path data/mask_train --mask_test_path data/mask_test
```

## Train the model

`python train.py --model <model-name> ...`

`python train.py --model r2unet --epochs 25 --early_stopping_p 20 --cutmix_p 1 --hair_aug_p 0 --mosaic_p 0 --batch_size 16`

## Try Jupyter notebook
`train-2016.ipynb`


## Cite as
If you have benefited from our work, please consider citing us with the following Bibtex citation format:

```
@misc{https://doi.org/10.48550/arxiv.2210.16399,
  doi = {10.48550/ARXIV.2210.16399},
  
  url = {https://arxiv.org/abs/2210.16399},
  
  author = {Kazaj, Pooya Mohammadi and Koosheshi, MohammadHossein and Shahedi, Ali and Sadr, Alireza Vafaei},
   
  keywords = {Image and Video Processing (eess.IV), Computer Vision and Pattern Recognition (cs.CV), Machine Learning (cs.LG), FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {U-Net-based Models for Skin Lesion Segmentation: More Attention and Augmentation},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}
```