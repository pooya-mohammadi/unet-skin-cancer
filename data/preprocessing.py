from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
from argparse import *


def preprocessing(train_zip, test_zip, mask_train_zip, mask_test_zip,
                  train_path, test_path, mask_train_path, mask_test_path):
    with urlopen(train_zip) as zip_train:
        with ZipFile(BytesIO(zip_train.read())) as zfile:
            zfile.extractall(train_path)
    with urlopen(test_zip) as zip_test:
        with ZipFile(BytesIO(zip_test.read())) as zfile:
            zfile.extractall(test_path)
    with urlopen(mask_train_zip) as zip_mask_train:
        with ZipFile(BytesIO(zip_mask_train.read())) as zfile:
            zfile.extractall(mask_train_path)
    with urlopen(mask_test_zip) as zip_mask_test:
        with ZipFile(BytesIO(zip_mask_test.read())) as zfile:
            zfile.extractall(mask_test_path)
    print("Preprocessing is done")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--train_zip_url",
                        default="https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Training_Data.zip")
    parser.add_argument("--test_zip_url",
                        default="https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Test_Data.zip")
    parser.add_argument("--mask_train_zip_url",
                        default="https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Training_GroundTruth.zip")
    parser.add_argument("--mask_test_zip_url",
                        default="https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Test_GroundTruth.zip")
    parser.add_argument("--train_path", default="./train")
    parser.add_argument("--test_path", default="./test")
    parser.add_argument("--mask_train_path", default="./masktrain")
    parser.add_argument("--mask_test_path", default="./masktest")

    args = parser.parse_args(args=[])
    print(args.train_zip_url)
    preprocessing(args.train_zip_url, args.test_zip_url, args.mask_train_zip_url, args.mask_test_zip_url,
                  args.train_path, args.test_path, args.mask_train_path, args.mask_test_path)
