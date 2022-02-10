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
