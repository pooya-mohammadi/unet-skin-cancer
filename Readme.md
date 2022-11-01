# U-Net-based Models for Skin Lesion Segmentation: More Attention and Augmentation
This repository is associated with [this paper](https://arxiv.org/abs/2210.16399). In this work we investigated making use of attention modules to overcome some problems with the basic U-Net architectures. Also, different augmentation methods are deployed. The project is publicized here for reproduction purposes. Please follow the succeeding instructions. 

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