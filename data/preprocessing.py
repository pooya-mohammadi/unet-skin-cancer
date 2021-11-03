import os.path

from deep_utils import download_file
from argparse import *
import shutil


def preprocessing(DATASET_NAME,
                  train_zip, test_zip, mask_train_zip, train_path, test_path, mask_train_path,
                  mask_test_zip=None, mask_test_path=None, val_zip=None, val_path=None, mask_val_zip=None,
                  mask_val_path=None):
    download_rename(train_zip, train_path)
    download_rename(mask_train_zip, mask_train_path)

    if DATASET_NAME == "ISIC_2016":
        download_rename(test_zip, test_path)
        download_rename(mask_test_zip, mask_test_path)

    elif DATASET_NAME == "ISIC_2018":
        download_rename(val_zip, val_path)
        download_rename(mask_val_zip, mask_val_path)
    print("Preprocessing is done")


def download_rename(train_zip, train_path):
    download_url = '.'
    file_name = train_zip.split("/")[-1][:-4]
    if not os.path.exists(file_name):
        download_file(url=train_zip, download_dir=download_url, unzip=True, remove_zip=True)
    else:
        print('file exists')
    if os.path.exists(train_path):
        shutil.rmtree(train_path)
    shutil.move(download_url + "/" + file_name, train_path)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--DATASET_NAME", type=str, default="ISIC_2016")

    args = parser.parse_args()

    print(args.DATASET_NAME)
    if args.DATASET_NAME == "ISIC_2016":
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
        parser.add_argument("--mask_train_path", default="./mask_train")
        parser.add_argument("--mask_test_path", default="./mask_test")

        args = parser.parse_args()
        preprocessing(args.DATASET_NAME,
                      args.train_zip_url, args.test_zip_url, args.mask_train_zip_url, args.train_path, args.test_path,
                      args.mask_train_path,
                      mask_test_zip=args.mask_test_zip_url, mask_test_path=args.mask_test_path)
    elif args.DATASET_NAME == "ISIC_2018":
        parser.add_argument("--train_zip_url",
                            default="https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Training_Input.zip")
        parser.add_argument("--test_zip_url",
                            default="https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Test_Input.zip")
        parser.add_argument("--val_zip_url",
                            default="https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Validation_Input.zip")
        parser.add_argument("--mask_train_zip_url",
                            default="https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Training_GroundTruth.zip")
        parser.add_argument("--mask_val_zip_url",
                            default="https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Validation_GroundTruth.zip")
        parser.add_argument("--train_path", default="./train")
        parser.add_argument("--test_path", default="./test")
        parser.add_argument("--val_path", default="./val")
        parser.add_argument("--mask_train_path", default="./mask_train")
        parser.add_argument("--mask_val_path", default="./mask_val")

        args = parser.parse_args()
        preprocessing(args.DATASET_NAME,
                      args.train_zip_url, args.test_zip_url, args.mask_train_zip_url, args.train_path, args.test_path,
                      args.mask_train_path,
                      mask_test_zip=None, mask_test_path=None,
                      val_zip=args.val_zip_url, val_path=args.val_path, mask_val_zip=args.mask_val_zip_url,
                      mask_val_path=args.mask_val_path)
