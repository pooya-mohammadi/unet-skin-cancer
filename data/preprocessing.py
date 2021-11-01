from deep_utils import download_file
from argparse import *


def preprocessing(DATASET_NAME,
                  train_zip, test_zip, mask_train_zip, train_path, test_path, mask_train_path,
                  mask_test_zip=None, mask_test_path=None, val_zip=None, val_path=None, mask_val_zip=None,
                  mask_val_path=None):
    download_file(url=train_zip, file_path=train_path, unzip=True)
    download_file(url=mask_train_zip, file_path=mask_train_path, unzip=True)

    if DATASET_NAME == "ISIC_2016":
        download_file(url=test_zip, file_path=test_path, unzip=True)
        download_file(url=mask_test_zip, file_path=mask_test_path, unzip=True)

    elif DATASET_NAME == "ISIC_2018":
        download_file(url=val_zip, file_path=val_path, unzip=True)
        download_file(url=mask_val_zip, file_path=mask_val_path, unzip=True)
    print("Preprocessing is done")


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
        parser.add_argument("--mask_train_path", default="./masktrain")
        parser.add_argument("--mask_test_path", default="./masktest")

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
                            default="https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task2_Validation_GroundTruth.zip")
        parser.add_argument("--train_path", default="./train")
        parser.add_argument("--test_path", default="./test")
        parser.add_argument("--val_path", default="./val")
        parser.add_argument("--mask_train_path", default="./masktrain")
        parser.add_argument("--mask_val_path", default="./maskval")

        args = parser.parse_args()
        preprocessing(args.DATASET_NAME,
                      args.train_zip_url, args.test_zip_url, args.mask_train_zip_url, args.train_path, args.test_path,
                      args.mask_train_path,
                      mask_test_zip=None, mask_test_path=None,
                      val_zip=args.val_zip_url, val_path=args.val_path, mask_val_zip=args.mask_val_zip_url,
                      mask_val_path=args.mask_val_path)
