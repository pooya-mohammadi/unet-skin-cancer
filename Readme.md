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
