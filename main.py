
from tqdm import tqdm
import os
import random
import argparse
import numpy as np
from torch.utils import data
import torch
import torch.nn as nn
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

import network
import utils
from datasets import DataSegmentation, DataSegmentationTest
from utils import ext_transforms as et
from metrics import StreamSegMetrics
from utils.visualizer import Visualizer



def get_argparser():
    parser = argparse.ArgumentParser()

    # Deeplab Options
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=['deeplabv3_resnet50',  'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=30e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--use_ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")

    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")
    parser.add_argument("--jpg_dir", type=str, default='',
                        help='jpg dir')
    parser.add_argument("--png_dir", type=str, default='',
                        help='png dir')
    parser.add_argument("--list_dir", type=str, default='',
                        help='train.txt and val.txt folder path')
    parser.add_argument("--checkpoints", type=str, default='./checkpoints',
                        help='checkpoints save dir. this dir will save checkpoints when training')
    parser.add_argument("--save_prediction_dir", type=str, default='./results',
                        help='save result dir')

    # Test Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--test_dir", type=str, default='',
                        help='test jpg dir')

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='8097',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    parser.add_argument("--pretrained_backbone_dir", type=str, default='',
                        help='path to save pretrained dir')

    return parser


def is_test(opts):
    if utils.is_null_string(opts.test_dir):
        return False
    return True

def get_test_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.crop_val:
        val_transform = et.ExtCompose([
            et.ExtResize(opts.crop_size),
            et.ExtCenterCrop(opts.crop_size),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    else:
        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    # 生成test_dir
    test_dst = DataSegmentationTest(transform=val_transform, test_dir=opts.test_dir, png_dir=opts.png_dir)
    return test_dst

def get_dataset(opts):
    """ Dataset And Augmentation
    """
    train_transform = et.ExtCompose([
        #et.ExtResize(size=opts.crop_size),
        et.ExtRandomScale((0.5, 2.0)),
        et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    if opts.crop_val:
        val_transform = et.ExtCompose([
            et.ExtResize(opts.crop_size),
            et.ExtCenterCrop(opts.crop_size),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    else:
        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    train_dst = DataSegmentation(image_set='train', transform=train_transform, jpg_dir=opts.jpg_dir, png_dir=opts.png_dir, list_dir=opts.list_dir)
    val_dst = DataSegmentation(image_set='val', transform=val_transform, jpg_dir=opts.jpg_dir, png_dir=opts.png_dir, list_dir=opts.list_dir)
    return train_dst, val_dst


def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []

    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
    img_id = 0

    if opts.save_val_results:
        if os.path.exists(opts.save_prediction_dir):
            import shutil
            shutil.rmtree(opts.save_prediction_dir)
        os.mkdir(opts.save_prediction_dir)


    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):
            
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(image).save(os.path.join(opts.save_prediction_dir, str(img_id)+"_image.png"))
                    Image.fromarray(target).save(os.path.join(opts.save_prediction_dir, str(img_id)+"_target.png"))
                    Image.fromarray(pred).save(os.path.join(opts.save_prediction_dir, str(img_id)+"_pred.png"))

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())

                    plt.savefig(os.path.join(opts.save_prediction_dir, str(img_id) + "_overlay.png"), bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score, ret_samples


def test(opts, model, loader, device, metrics=None, ret_samples_ids=None):
    """Do validation and return specified samples"""
    # metrics.reset()
    ret_samples = []

    if os.path.exists(opts.save_prediction_dir):
        import shutil
        shutil.rmtree(opts.save_prediction_dir)
    os.makedirs(opts.save_prediction_dir)

    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
    img_id = 0
    score = 0

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            # metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            for i in range(len(images)):
                image = images[i].detach().cpu().numpy()
                target = targets[i]
                pred = preds[i]

                image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                target = loader.dataset.decode_target(target).astype(np.uint8)
                pred = loader.dataset.decode_target(pred).astype(np.uint8)

                Image.fromarray(image).save(os.path.join(opts.save_prediction_dir, str(img_id) + "_image.png"))
                Image.fromarray(target).save(os.path.join(opts.save_prediction_dir, str(img_id) + "_target.png"))
                Image.fromarray(pred).save(os.path.join(opts.save_prediction_dir, str(img_id) + "_pred.png"))

                fig = plt.figure()
                plt.imshow(image)
                plt.axis('off')
                plt.imshow(pred, alpha=0.7)
                ax = plt.gca()
                ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())

                plt.savefig(os.path.join(opts.save_prediction_dir, str(img_id) + "_overlay.png"), bbox_inches='tight',
                            pad_inches=0)
                plt.close()
                img_id += 1

        # score = metrics.get_results()
    return score, ret_samples


def main(opts=None):
    if opts is None:
        opts = get_argparser().parse_args()
    print("---> opts: ", opts)

    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    test_check = is_test(opts)
    if test_check:
        test_dst = get_test_dataset(opts)
        test_loader = data.DataLoader(
            test_dst, batch_size=opts.batch_size, shuffle=True, num_workers=0)
        print("Test Dataset set: %d" % (len(test_dst)))
    else:
        train_dst, val_dst = get_dataset(opts)
        # num_workers在mac上必须要为0
        train_loader = data.DataLoader(
            train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=0)
        val_loader = data.DataLoader(
            val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=0)
        print("Dataset check. Train set length: %d, Val set length: %d" %
              (len(train_dst), len(val_dst)))

    # Set up model
    model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
    }

    if opts.use_ckpt is not None and os.path.isfile(opts.use_ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.use_ckpt, map_location=torch.device('cpu'))
        opts.num_classes = checkpoint["num_classes"]

    if not utils.is_null_string(opts.pretrained_backbone_dir):
        print("try to load pretrained model from: ", opts.pretrained_backbone_dir)
        model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride, pretrained_backbone=True, model_dir=opts.pretrained_backbone_dir)
        print("load pretrained model success: ", opts.pretrained_backbone_dir)
    else:
        print("don't use pretrained model")
        model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride, pretrained_backbone=False, model_dir=None)


    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1*opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)∂
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)

    if opts.lr_policy=='poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy=='step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    # criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    def save_use_ckpt(path):
        """ save current model
        """
        save_path = os.path.join(opts.checkpoints, path)
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
            "num_classes": opts.num_classes
        }, save_path)
        # torch.save(model, save_path+".orig")
        print("Model saved as %s" % save_path)

    utils.mkdir(opts.checkpoints)
    # Restore
    best_score = -1
    cur_itrs = 0
    cur_epochs = 0
    if opts.use_ckpt is not None and os.path.isfile(opts.use_ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        # checkpoint = torch.load(opts.use_ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.use_ckpt)
        print("Model restored from %s" % opts.use_ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    #==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if test_check:
        model.eval()
        test_score, ret_samples = test(opts=opts, model=model, loader=test_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        # print(metrics.to_str(test_score))
        return

    if opts.test_only:
        model.eval()
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        return

    interval_loss = 0
    while True: #cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        for (images_orig, labels_orig) in train_loader:
            cur_itrs += 1

            images = images_orig.to(device, dtype=torch.float32)
            labels = labels_orig.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss
            if vis is not None:
                vis.vis_scalar('Loss', cur_itrs, np_loss)

            output_interval = 1
            if (cur_itrs) % output_interval == 0:
                interval_loss = interval_loss/output_interval
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                interval_loss = 0.0

            if (cur_itrs) % opts.val_interval == 0:
            # if (cur_itrs) % opts.total_itrs == 0:
                # save_use_ckpt('latest_%s_os%d.pth' % (opts.model, opts.output_stride))
                print("validation...")
                model.eval()
                val_score, ret_samples = validate(
                    opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
                print(metrics.to_str(val_score))
                if val_score['Mean IoU'] > best_score:  # save best model
                    print("save best pth. current best_core: ", best_score, " new score:", val_score['Mean IoU'])
                    best_score = val_score['Mean IoU']
                    save_use_ckpt('best_%s_os%d.pth' % (opts.model, opts.output_stride))
                else:
                    print("don't save best pth. current best_core: ", val_score['Mean IoU'], " new score:", best_score)

                if vis is not None:  # visualize validation score and samples
                    vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                    vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                    vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

                    for k, (img, target, lbl) in enumerate(ret_samples):
                        img = (denorm(img) * 255).astype(np.uint8)
                        target = train_dst.decode_target_colorful(target).transpose(2, 0, 1).astype(np.uint8)
                        lbl = train_dst.decode_target_colorful(lbl).transpose(2, 0, 1).astype(np.uint8)
                        concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                        vis.vis_image('Sample %d' % k, concat_img)
                model.train()
            scheduler.step()

            if cur_itrs >= opts.total_itrs:
                return

        
if __name__ == '__main__':
    main()
    print("finish task")
