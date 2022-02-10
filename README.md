# open-lpr pretrained models

Pre trained models consisting of data from coco, open-lpr-dataset-plate-detection and other datasets.

This repository is part of the [OpenLPR project](https://github.com/faisalthaheem/open-lpr).

There are two models, a full precision fp32 model and a mixed prcision model. Both yield similar results. These models have been generated using the [Open LPR Dataset](https://github.com/faisalthaheem/open-lpr-dataset-plate-detection)

The training was conducated using images from [SSD Pytorch repository](https://github.com/faisalthaheem/SSD-pytorch)

To download the models, use the following urls
|Model|Download link|Command to download|
|---|---|---|
|fp32|https://1drv.ms/u/s!AvNYo0I9EXbwh1hpdp-r-1qENn_t?e=nKbmnB|wget -O ssd.pth 'https://api.onedrive.com/v1.0/shares/s!AvNYo0I9EXbwh1hpdp-r-1qENn_t/root/content'|
|amp|https://1drv.ms/u/s!AvNYo0I9EXbwh1b3L5sMzE-c7hoc?e=cvDKyY|wget -O ssd.pth 'https://api.onedrive.com/v1.0/shares/s!AvNYo0I9EXbwh1b3L5sMzE-c7hoc/root/content'|


![Jeep](docs/jeep-side-1.jpg)

# Training instructions

The following contents will assume that we are working in $HOME/work/repos which will be the root folder containing everything related to open lpr.
> IMPORTANT: If you have a different folder structure please review the configuration files for each tool/repo for adjustments.

Overall the process to train your custom model consists of the following steps
- Downloading the coco dataset
- [Checking out the open lpr plate detection dataset](https://github.com/faisalthaheem/open-lpr-dataset-plate-detection)
- Preparing open lpr training dataset
- [Cloning the SSD repository](https://github.com/faisalthaheem/SSD-pytorch)
- Building the training container and initiating the training process

## Downloading the coco dataset
Execute the following commands to checkout the coco dataset

```bash
mkdir datasets && cd datasets && \
wget -O download_coco.sh 'https://raw.githubusercontent.com/faisalthaheem/open-lpr-pretrained-models/main/download_coco.sh' && \
chmod +x download_coco.sh && sh download_coco.sh
```

## Checking out the open lpr plate detection dataset
> While being in the datasets directory

```bash
git clone https://github.com/faisalthaheem/open-lpr-dataset-plate-detection
```

## Preparing open lpr training dataset
> While being in the datasets directory

```bash
mkdir cars_from_coco
```

We begin with extracting the train and val images from coco dataset containing vehicles
```bash
docker run --rm -it -u $UID \
-v $PWD:/datasets \
faisalthaheem/simanno:scripts-2.0 \
"/usr/local/bin/python3.8 import-cat-from-coco.py -t val -c car -li 1 -af /datasets/coco/annotations/instances_val2017.json -dp /datasets/cars_from_coco -sp /datasets/coco/val2017/"
```

```bash
docker run --rm -it -u $UID \
-v $PWD:/datasets \
faisalthaheem/simanno:scripts-2.0 \
"/usr/local/bin/python3.8 import-cat-from-coco.py -t train -c car -li 1 -af /datasets/coco/annotations/instances_train2017.json -dp /datasets/cars_from_coco -sp /datasets/coco/train2017/"
```

Finally we merge the extracted coco dataset with the open lpr plate detection dataset to get the training dataset
```bash
cat <<EOT > mergedbs.yaml
dest:
  # Paths to the destination databases and directories
  train_db: /datasets/openlpr/train.db
  val_db: /datasets/openlpr/val.db
  train_path: /datasets/openlpr/train
  val_path: /datasets/openlpr/val
  # label_mapping defines how labels ids from source databases map into
  # the destination database
  label_mapping:
    coco_cars:
      1: 1
    plate_det:
      1: 2
src:
  - coco_cars:
      train_db: /datasets/cars_from_coco/train.db
      val_db: /datasets/cars_from_coco/val.db
      train_path: /datasets/cars_from_coco/train
      val_path: /datasets/cars_from_coco/val
  - plate_det:
      train_db: /datasets/open-lpr-dataset-plate-detection/train.db
      val_db: /datasets/open-lpr-dataset-plate-detection/val.db
      train_path: /datasets/open-lpr-dataset-plate-detection/train
      val_path: /datasets/open-lpr-dataset-plate-detection/val
EOT
```

```bash
docker run --rm -it -u $UID \
-v $PWD:/datasets \
faisalthaheem/simanno:scripts-2.0 "/usr/local/bin/python3.8 /simanno/scripts/mergedbs.py -c /datasets/mergedbs.yaml"
```

## Cloning the SSD repository

> Should be executed in $HOME/work/repos

```bash
git clone https://github.com/faisalthaheem/SSD-pytorch
```

## Building the training container and initiating the training process

The following command will build an image which shold allow training the models using mixed precision
```bash
cd SSD-pytorch && \
sh build-docker-image.sh
```

Once the image is built we can launch a container with the following command
```bash
export DS_DIR=$HOME/work/repos/datasets
export ROOT_CACHE_DIR=$HOME/work/repos/mountedvols/root-cache
export CHECKPOINT_DIR=$HOME/work/repos/checkpoints
sh run-training-container.sh
```

Once the container is running, the following command can be used to begin the training process using mixed precision
```bash
python3 train.py --save-folder /checkpoints/openlpr-amp --batch-size 42 --epochs 30 --amp --dist --num-workers 16
```

> Please note you may need to adjust the parameters in the preceding command given the amount of resources available, the above command was tailored for a system running stock ubuntu desktop 20.04 with an RTX 2070 (8GB).

Once the training is completed you should be able to see an output similar to following
```bash
Epoch: 1. Loss: 5.09479: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 348/348 [03:12<00:00,  1.80it/s]
Epoch: 2. Loss: 4.56174: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 348/348 [03:09<00:00,  1.84it/s]
Epoch: 3. Loss: 3.95576: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 348/348 [03:02<00:00,  1.90it/s]
.
.
.
.
Epoch: 29. Loss: 3.32882:  76%|████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                 | 263/348 [02:17<00:44,  1.92it/s]Gradient overflow.  Skipping step, loss scaler 0 reducing loss scale to 65536.0
Epoch: 29. Loss: 3.03714: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 348/348 [03:01<00:00,  1.92it/s]
Epoch: 30. Loss: 2.46069: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 348/348 [03:01<00:00,  1.92it/s]
```

At this point, the model should be available on the host machine under checkpoints/openlpr-amp directory and can be used with openalpr project.