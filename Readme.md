# U-Net-based Models for Skin Lesion Segmentation: More Attention and Augmentation
This repository is associated with [this paper](https://arxiv.org/abs/2210.16399). In this work, ten [models](https://github.com/pooya-mohammadi/unet-skin-cancer/tree/master/models) and four augmentation configurations are trained on the ISIC 2016 dataset. The performance and overfitting are compared utilizing five metrics. Our results show that the [U-Net-Resnet50](https://github.com/pooya-mohammadi/unet-skin-cancer/blob/master/models/unet_res50.py) and the [R2U-Net](https://github.com/pooya-mohammadi/unet-skin-cancer/blob/master/models/r2unet.py) have the highest metrics value, along with two data augmentation scenarios. We also investigate CBAM and AG blocks in the U-Net architecture, which enhances segmentation performance at a meager computational cost. In addition, we propose using pyramid, AG, and CBAM blocks in a sequence, which significantly surpasses the results of using the two individually. Finally, our experiments show that models that have exploited attention modules successfully overcome common skin lesion segmentation problems. Lastly, in the spirit of reproducible research, we implement models and codes publicly available.

## Download dataset
To download & preprocess the data, run the following module. In case of `URL fetch failure` turn on your VPN.
```
python data/preprocessing.py --train_path data/train --test_path data/test --mask_train_path data/mask_train --mask_test_path data/mask_test
```

To train the model:

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