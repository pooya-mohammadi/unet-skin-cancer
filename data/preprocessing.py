from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
from argparse import *


def preprocessing(args):
    with urlopen(args[0]) as zip_train:
        with ZipFile(BytesIO(zip_train.read())) as zfile:
            zfile.extractall(args[4])
    with urlopen(args[1]) as zip_test:
        with ZipFile(BytesIO(zip_test.read())) as zfile:
            zfile.extractall(args[5])
    with urlopen(args[2]) as zip_mask_train:
        with ZipFile(BytesIO(zip_mask_train.read())) as zfile:
            zfile.extractall(args[6])
    with urlopen(args[3]) as zip_mask_test:
        with ZipFile(BytesIO(zip_mask_test.read())) as zfile:
            zfile.extractall(args[7])
    print("Preprocessing is done")
if __name__ == "main":
    parser = ArgumentParser()
    print("salam")
    # parser.add_argument("--url", default='mu url', help='this is the url')
    train_zip_url = "https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Training_Data.zip"
    test_zip_url = "https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Test_Data.zip"
    mask_train_zip_url = "https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Training_GroundTruth.zip"
    mask_test_zip_url = "https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Test_GroundTruth.zip"
    train_path = "./train"
    test_path = "/.test"
    mask_train_path = "./masktrain"
    mask_test_path = "./masktest"
    default_list = [train_zip_url, test_zip_url, mask_train_zip_url, mask_test_zip_url,
                        train_path, test_path, mask_train_path, mask_test_path]

    parser = ArgumentParser()
    parser.add_argument("--url", nargs="+", default=default_list)

    args = parser.parse_args()
    print(args.url)
    preprocessing(args.url)


#"https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Training_Data.zip",
#"https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Training_GroundTruth.zip"
#"https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Test_Data.zip"
#"https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Test_GroundTruth.zip"