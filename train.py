import torch
import torch.nn.functional as F
from torch.autograd import Variable
from PVT_Model.pvtmodel import PvtNet
import numpy as np
import pdb, os, argparse
from datetime import datetime
from data import get_loader, test_dataset
from utils import clip_gradient, adjust_lr
from pamr import BinaryPamr
import os
import logging
from scipy import misc
from fast_slic import Slic
import smoothness
from tools import *
import imageio
from lscloss import *
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=300, help='epoch number')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=200, help='every n epochs decay learning rate')
parser.add_argument('--sm_loss_weight', type=float, default=0.3, help='weight for smoothness loss')
parser.add_argument('--edge_loss_weight', type=float, default=1.0, help='weight for edge loss')
parser.add_argument('--save_path', type=str, default='', help='the path to save models and logs')
parser.add_argument('--load', type=str, default='', help='train from checkpoints')
opt = parser.parse_args()

print('Learning Rate: {}'.format(opt.lr))
# build models
model = PvtNet(opt)
model.encoder_rgb.load_state_dict(torch.load(""), strict=True)
model.encoder_depth.load_state_dict(torch.load(""), strict=True)
# if(opt.load is not None):
#     model.pvtb2.init_weights(opt.load)
model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)


image_root = ''
depth_root = ''
gt_root = ''
mask_root = ''
grayimg_root = ''
edge_root = ''
test_image_root = ''
test_gt_root =''
test_depth_root =''
train_loader = get_loader(image_root, depth_root, gt_root, mask_root, grayimg_root, edge_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
test_loader = test_dataset(test_image_root, test_gt_root, test_depth_root, opt.trainsize)
total_step = len(train_loader)

CE = torch.nn.BCELoss()
smooth_loss = smoothness.smoothness_loss(size_average=True)

best_mae = 1
best_epoch = 0

loss_lsc = LocalSaliencyCoherence().cuda()
loss_lsc_kernels_desc_defaults = [{"weight": 1, "xy": 6, "rgb": 0.1}]

loss_lsc_radius = 5
save_path = opt.save_path

logging.basicConfig(filename=save_path + 'log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("scribbleNet-Train")
logging.info("Config")
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.load, save_path,
        opt.decay_epoch))

def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2, 3))/weit.sum(dim=(2, 3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()

# def visualize_prediction1(pred,name):
#     for kk in range(pred.shape[0]):
#         pred_edge_kk = pred[kk, :, :, :]
#         pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
#         pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
#         pred_edge_kk *= 255.0
#         pred_edge_kk = pred_edge_kk.astype(np.uint8)
#         save_path = './label_gt/'
#         if not os.path.exists(save_path):
#             os.makedirs(save_path)
#         # name = '{:02d}_sal1.png'.format(kk)
#         imageio.imsave(save_path + name[kk], pred_edge_kk)


def run_pamr(img, sal):
    lbl_self = BinaryPamr(img, sal.clone().detach(), binary=0.4)
    return lbl_self


def train(train_loader, model, optimizer, epoch):
    # global step
    model.train()
    loss_all = 0
    epoch_step = 0
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        images, depths, gts, masks, grays, edges, label_gt, label_gt_depth, name = pack
        images = Variable(images)
        depths = Variable(depths)
        gts = Variable(gts)
        masks = Variable(masks)
        grays = Variable(grays)
        # edges = Variable(edges)
        label_gt = Variable(label_gt)
        label_gt_depth = Variable(label_gt_depth)
        images = images.cuda()
        # depths = depths.repeat(1, 3, 1, 1, ).cuda()
        depths = depths.cuda()
        gts = gts.cuda()
        masks = masks.cuda()
        grays = grays.cuda()
        # edges = edges.cuda()
        label_gt = label_gt.cuda()
        label_gt_depth = label_gt_depth.cuda()


        img_size = images.size(2) * images.size(3) * images.size(0)
        ratio = img_size / torch.sum(masks)

        result_final, mask4, mask3, mask2, sal1, sal2 = model(images, depths)

        # BCEloss for the 1st DF
        sal1_loss = CE(sal1, label_gt)

        # BCEloss for the 2nd DF
        sal2_loss = CE(sal2, label_gt_depth)

        # visualize_prediction1(sal1, name)
        # The self-supervision term between 1st DF and 2nd DF


        # Guidance loss for the final saliency decoder
        lbl_tea = run_pamr(images, (sal1 + sal2) / 2)
        # visualize_prediction1(lbl_tea, name)
        loss_reult_final = structure_loss(torch.sigmoid(result_final), lbl_tea)
        loss_reult_mask4 = structure_loss(torch.sigmoid(mask4), lbl_tea)
        loss_reult_mask3 = structure_loss(torch.sigmoid(mask3), lbl_tea)
        loss_reult_mask2 = structure_loss(torch.sigmoid(mask2), lbl_tea)

        image_scale = F.interpolate(images, scale_factor=0.25, mode='bilinear', align_corners=True)
        depth_scale = F.interpolate(depths, scale_factor=0.25, mode='bilinear', align_corners=True)
        #
        result_final_scale, mask4_s, mask3_s, mask2_s, sal1_s, sal2_s = model(image_scale, depth_scale)
        result_out_scale = F.interpolate(torch.sigmoid(result_final), scale_factor=0.25, mode='bilinear', align_corners=True)
        loss_ssc = SaliencyStructureConsistency(torch.sigmoid(result_final_scale), result_out_scale, 0.85)

        images_ = F.interpolate(images, scale_factor=0.25, mode="bilinear", align_corners=True)
        sample_rgb = {'rgb': images_}

        #
        final_prob = torch.sigmoid(result_final)
        final_prob = final_prob * masks
        smoothLoss_cur_final = opt.sm_loss_weight * smooth_loss(torch.sigmoid(result_final), grays)
        sal_loss_final = ratio * CE(final_prob, gts * masks) + smoothLoss_cur_final

        result_final_ = F.interpolate(torch.sigmoid(result_final), scale_factor=0.25, mode="bilinear", align_corners=True)
        lossfinal_lsc = loss_lsc(result_final_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample_rgb, images_.shape[2],images_.shape[3])['loss']
        lossfinal = sal_loss_final + lossfinal_lsc + loss_ssc + loss_reult_final

        mask4_prob = torch.sigmoid(mask4)
        mask4_prob = mask4_prob * masks
        smoothLoss_cur_mask4 = opt.sm_loss_weight * smooth_loss(torch.sigmoid(mask4), grays)
        sal_loss_mask4 = ratio * CE(mask4_prob, gts * masks) + smoothLoss_cur_mask4


        mask4_ = F.interpolate(torch.sigmoid(mask4), scale_factor=0.25, mode="bilinear", align_corners=True)
        lossmask4_lsc = loss_lsc(mask4_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample_rgb, images_.shape[2],images_.shape[3])['loss']
        lossmask4 = sal_loss_mask4 + lossmask4_lsc +loss_reult_mask4


        mask3_prob = torch.sigmoid(mask3)
        mask3_prob = mask3_prob * masks
        smoothLoss_cur_mask3 = opt.sm_loss_weight * smooth_loss(torch.sigmoid(mask3), grays)
        sal_loss_mask3 = ratio * CE(mask3_prob, gts * masks) + smoothLoss_cur_mask3


        mask3_ = F.interpolate(torch.sigmoid(mask3), scale_factor=0.25, mode="bilinear", align_corners=True)
        lossmask3_lsc = loss_lsc(mask3_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample_rgb, images_.shape[2],images_.shape[3])['loss']
        lossmask3 = sal_loss_mask3 + lossmask3_lsc +loss_reult_mask3

        mask2_prob = torch.sigmoid(mask2)
        mask2_prob = mask2_prob * masks
        smoothLoss_cur_mask2 = opt.sm_loss_weight * smooth_loss(torch.sigmoid(mask2), grays)
        sal_loss_mask2 = ratio * CE(mask2_prob, gts * masks) + smoothLoss_cur_mask2


        mask2_ = F.interpolate(torch.sigmoid(mask2), scale_factor=0.25, mode="bilinear", align_corners=True)
        lossmask2_lsc = loss_lsc(mask2_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample_rgb, images_.shape[2], images_.shape[3])['loss']
        lossmask2 = sal_loss_mask2 + lossmask2_lsc +loss_reult_mask2
        loss = lossfinal * 1 + lossmask2 * 0.8 + lossmask3 * 0.6 + lossmask4 * 0.4 + sal1_loss + sal2_loss

        loss.backward()

        clip_gradient(optimizer, opt.clip)
        optimizer.step()
        # if i % 10 == 0 or i == total_step:
        #     print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], sal1_loss: {:0.4f}, loss: {:0.4f}, sal2_loss: {:0.4f}'.
        #           format(datetime.now(), epoch, opt.epoch, i, total_step, loss1.data, loss.data, loss2.data))
        if i % 100 == 0 or i == total_step or i == 1:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Lossfinal: {:.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data))
            logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Lossfinal: {:.4f}'.
                         format(epoch, opt.epoch, i, total_step, loss.data))

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if epoch % 30 == 0:
        torch.save(model.state_dict(), save_path + 'scribble' + '_%d'  % epoch  + '.pth')

def test(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    # 神经网络沿用batch normalization的值，并不使用drop out
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, depth, name, img_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            # depth = depth.repeat(1, 3, 1, 1, ).cuda()
            depth = depth.cuda()
            res, _, _, _ = model(image, depth)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae = mae_sum / test_loader.size
        # writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'scribble_epoch_best.pth')
                print('best epoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))

print("Starting!")
for epoch in range(1, opt.epoch+1):
    adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
    train(train_loader, model, optimizer, epoch)
    test(test_loader, model, epoch, save_path)
