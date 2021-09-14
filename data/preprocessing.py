from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

def preprocessing(train_zip, test_zip, mask_train_zip, mask_test_zip,
                 train_path, test_path, mask_train_path,mask_test_path):
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

#"https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Training_Data.zip",
#"https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Training_GroundTruth.zip"
#"https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Test_Data.zip"
#"https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Test_GroundTruth.zip"