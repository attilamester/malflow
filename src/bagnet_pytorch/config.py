import os


DATASETS = {}


data_dir = "/home/anansi/Data/bodmas/"
DATASETS["cifar"] = {
    "img_shape": (32, 32),
    "num_classes": 10,
    "color_channels": 3,
    "data_dir": data_dir,
    "metadata_file": "BODMAS_ground_truth.csv",
    "images_dir": "BODMAS_images_512_512_False_False",  
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
    "augm": False,
}


data_dir = "/home/anansi/Data/bodmas/"
DATASETS["bodmas"] = {
    "img_shape": (300, 300),
    "num_classes": 56,
    "color_channels": 3,
    "data_dir": data_dir,
    "metadata_file": "BODMAS_ground_truth.csv",
    "images_dir": "BODMAS_images_512_512_False_False",    
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
    "image_folders": False,
    "augm": False,
}

train_batch_size = 8
test_batch_size = 8

# change here to read the config for the data set you want to work with:
dataset_config = DATASETS["bodmas"]
