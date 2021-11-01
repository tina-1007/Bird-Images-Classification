# Bird-Images-Classification

In this homework, I apply [TranFG](https://github.com/TACJu/TransFG) on our task [2021 VRDL HW1](#). 


## Installation

```
pip3 install -r requirements.txt
```

## Prepare data
Download the dataset of this task from [here](https://competitions.codalab.org/competitions/35668?secret_key=09789b13-35ec-4928-ac0f-6c86631dda07#participate-get_starting_kit) and put the folders and `.txt` files in `dataset` folder. The structure will be like: 
```
train.py
...
dataset
|- training_images
|- testing_images
|- classes.txt
|- testing_img_order.txt
|- training_labels.txt
```

## Testing
#### 1. Download the trained weights 
Get my trained model from [here](https://drive.google.com/file/d/1frwD4lEvk7e-xmWrdhHcMdLXeLVl31pU/view?usp=sharing) and put it in `/checkpoints/best/` or other name like `/checkpoints/{name}/`

#### 2. Generate the submission file by running

``` 
python inference.py --name best
```
The result will be generated in `/result/best/` or `/result/{name}/`

## Training
#### 1. Download Google pre-trained ViT models
Get models in [this link](https://console.cloud.google.com/storage/browser/vit_models) and put the `.npz` in `pretrained_models/` 

I use [imagenet21k/ViT-B_16.npz](https://console.cloud.google.com/storage/browser/_details/vit_models/imagenet21k/ViT-B_16.npz?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))) to get my best score.
rm
#### 2. Start training by running
```
python train.py --name best
```

## Reference

1. TransFG: A Transformer Architecture for Fine-grained Recognition
    * [Paper](https://arxiv.org/abs/2103.07976)
    * [Github](https://github.com/TACJu/TransFG) (Official Code)
3. Pytorch Official Totorial > Transfer Learning For Computer Vision Tutorial - [website](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) 
