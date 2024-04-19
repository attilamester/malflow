## Training a BagNet model


### Obtaining the callgraphs

* each sample has to be scanned with `Radare2` to obtain the call graph, and then, the image
* follow the instructions in root `README.md` to install the required packages
* `src/.env`:

```
BODMAS_DIR_SAMPLES=/path/to/samples
BODMAS_DIR_R2_SCANS=/path/to/r2-callgraphs
```

### Working with existing callgraphs

* sample code for loading a compressed pickle file and calling its image generation method:

```python
from core.data.bodmas import Bodmas
from core.model import CallGraphCompressed
from util import config

if __name__ == "__main__":
    config.load_env()
    compressed_path = CallGraphCompressed.get_compressed_path(Bodmas.get_dir_callgraphs(),
                                                              md5="f880e2b38aa997a3f272f31437bafc28")
    cg = CallGraphCompressed.load(compressed_path).decompress()
```

* `.env`:

```
DATASETS_BODMAS_GROUND_TRUTH_PATH=path/to/ground_truth.csv
DATASETS_BODMAS_IMG_DIR_PATH=path/to/images
DATASETS_BODMAS_IMG_SHAPE=30,30
DATASETS_BODMAS_IMG_COLOR_CHANNELS=3

TRAIN_SKIP_EXISTING=f

HPARAM_SPACE_MODEL="bagnet<9/17/33>, resnet<18/50>"
HPARAM_SPACE_MODEL_PRETRAINED="f,t"
HPARAM_SPACE_DATA_MIN_ITEM_PER_CLASS="100"
HPARAM_SPACE_DATA_MAX_ITEM_PER_CLASS="0" # 0 means no limit
HPARAM_SPACE_DATA_BATCH_SIZE="32, 64, 128"
HPARAM_SPACE_DATA_AUGM="t,f"

```

* the train script is based on a fork of `pytorch/examples/imagenet/main.py`:
  * we will use the flag `-m` to load our model and dataset
  * help:
```
$ python3 main.py  --help
usage: main.py [-h] [-a ARCH] [-m MODULE] [-j N] [--epochs N] [--start-epoch N] [-b N] [--lr LR] [--momentum M] [--wd W] [-p N] [--resume PATH] [-e] [--pretrained]
               [--world-size WORLD_SIZE] [--rank RANK] [--dist-url DIST_URL] [--dist-backend DIST_BACKEND] [--seed SEED] [--gpu GPU] [--multiprocessing-distributed] [--dummy]
               [DIR]

PyTorch ImageNet Training

positional arguments:
  DIR                   path to dataset (default: imagenet)

optional arguments:
  -h, --help            show this help message and exit
  -a ARCH, --arch ARCH  model architecture: alexnet | convnext_base | convnext_large | convnext_small | convnext_tiny | densenet121 | densenet161 | densenet169 | densenet201 |
                        efficientnet_b0 | efficientnet_b1 | efficientnet_b2 | efficientnet_b3 | efficientnet_b4 | efficientnet_b5 | efficientnet_b6 | efficientnet_b7 |
                        efficientnet_v2_l | efficientnet_v2_m | efficientnet_v2_s | get_model | get_model_builder | get_model_weights | get_weight | googlenet | inception_v3 |
                        list_models | maxvit_t | mnasnet0_5 | mnasnet0_75 | mnasnet1_0 | mnasnet1_3 | mobilenet_v2 | mobilenet_v3_large | mobilenet_v3_small | regnet_x_16gf |
                        regnet_x_1_6gf | regnet_x_32gf | regnet_x_3_2gf | regnet_x_400mf | regnet_x_800mf | regnet_x_8gf | regnet_y_128gf | regnet_y_16gf | regnet_y_1_6gf |
                        regnet_y_32gf | regnet_y_3_2gf | regnet_y_400mf | regnet_y_800mf | regnet_y_8gf | resnet101 | resnet152 | resnet18 | resnet34 | resnet50 |
                        resnext101_32x8d | resnext101_64x4d | resnext50_32x4d | shufflenet_v2_x0_5 | shufflenet_v2_x1_0 | shufflenet_v2_x1_5 | shufflenet_v2_x2_0 | squeezenet1_0
                        | squeezenet1_1 | swin_b | swin_s | swin_t | swin_v2_b | swin_v2_s | swin_v2_t | vgg11 | vgg11_bn | vgg13 | vgg13_bn | vgg16 | vgg16_bn | vgg19 | vgg19_bn
                        | vit_b_16 | vit_b_32 | vit_h_14 | vit_l_16 | vit_l_32 | wide_resnet101_2 | wide_resnet50_2 (default: resnet18)
  -m MODULE, --use-module-definitions MODULE
                        load a custom py file for the model and/or dataset & loader.The file can contain the following functions: get_model() -> nn.Moduleget_train_dataset() ->
                        torch.utils.data.Datasetget_val_dataset() -> torch.utils.data.Datasetget_train_loader() -> torch.utils.data.DataLoaderget_val_loader() ->
                        torch.utils.data.DataLoader(default: None)
  -j N, --workers N     number of data loading workers (default: 4)
  --epochs N            number of total epochs to run
  --start-epoch N       manual epoch number (useful on restarts)
  -b N, --batch-size N  mini-batch size (default: 256), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum M          momentum
  --wd W, --weight-decay W
                        weight decay (default: 1e-4)
  -p N, --print-freq N  print frequency (default: 10)
  --resume PATH         path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  --pretrained          use pre-trained model
  --world-size WORLD_SIZE
                        number of nodes for distributed training
  --rank RANK           node rank for distributed training
  --dist-url DIST_URL   url used to set up distributed training
  --dist-backend DIST_BACKEND
                        distributed backend
  --seed SEED           seed for initializing training.
  --gpu GPU             GPU id to use.
  --multiprocessing-distributed
                        Use multi-processing distributed training to launch N processes per node, which has N GPUs. This is the fastest way to use PyTorch for either single node
                        or multi node data parallel training
  --dummy               use fake data to benchmark
```
