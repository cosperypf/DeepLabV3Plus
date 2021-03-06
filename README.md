# DeepLabv3Plus-Pytorch

### Based on
https://github.com/VainF/DeepLabV3Plus-Pytorch.git
and thanks VainF



### Changes

VainF did more work in this, but when I want to train with my one dataset, I found a lot of problem, such as: I can't specify a given path for my dataset and pretrained model; I can't train directly with gray pictures because the original code is deeply bound with the VOC data set structure; and so on.

So I make some changes, such as:
1. No longer rely on separate data sets (such as VOC). Add some parameters instead
2. Add test process
3. Add pretrained model reading ways


Some new parameters for train :

```2aq
jpg_dir: dir full path for jpg files
png_dir: dir full path for mask png files
list_dir: include train.txt and test.txt 
save_prediction_dir: full path to save predicted files
```

Some new parameters for test:

```2aq
test_dir: dir full path to test image. Support .jpg and .png files
pretrained_backbone_dir: will download and use specified pretrained model if specified

```



### Prepare

1. install requirements
```
pip install -r requirements.txt
```
2. download dataset(if need)
* [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/)

You need to download additional labels from [Dropbox](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0) or [Tencent Weiyun](https://share.weiyun.com/5NmJ6Rk). Those labels come from [DrSleep's repo](https://github.com/DrSleep/tensorflow-deeplab-resnet).

* [Cityscapes](https://www.cityscapes-dataset.com/)




### How to use

#### Train:
1. prepare data like here. you should use more pictures
```
/datasets
    /data
```

2. start visdom(if need)
```
python -m visdom.server
```

3. start train
```
python main.py \
--jpg_dir ./datasets/data/JPEGImages \
--png_dir ./datasets/data/SegmentationClassAug \
--list_dir ./datasets/data/Segmentation \
--total_itrs 1 \
--num_classes 21 \
--crop_val --crop_size 513 \
--checkpoints ./results/checkpoints \
--save_prediction_dir ./results/result \
--val_interval 1 --save_val_results \
--enable_vis
```



#### Test:

```
--use_ckpt ./results/checkpoints/best_deeplabv3plus_mobilenet_os16.pth \
--save_prediction_dir ./results/predict_result \
--test_only --test_dir ./datasets/data/JPEGImages \
--pretrained_backbone_dir ./models \
--crop_val --crop_size 513
```

