# Skin-Cancer-Segmentation

## Download dataset
To download & preprocess the data:

`python data/preprocessing.py train-path data/train ...`

To train the model:

`python train.py --model <model-name> ...`
`python train.py --model r2unet --epochs 25 --early_stopping_p 20 --cutmix_p 0 --hair_aug_p 0 --mosaic_p 0 --batch_size 64`

## Try Jupyter notebook
`train-2016.ipynb`
