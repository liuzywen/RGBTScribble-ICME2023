import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
import os , argparse
import cv2
from PVT_Model.pvtmodel import PvtNet
from data import test_dataset



parser = argparse.ArgumentParser()
# parser.add_argument('--trainsize', type=int, default=256, help='testing size')
parser.add_argument('--testsize', type=int, default=256, help='testing size')
# parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path', type=str, default='', help='test dataset path')
# parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
opt = parser.parse_args()

dataset_path = opt.test_path

#set device for test
# if opt.gpu_id=='0':
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#     print('USE GPU 0')
# elif opt.gpu_id=='1':
#     os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#     print('USE GPU 1')

#load the model
model = PvtNet(None)
#Large epoch size may not generalize well. You can choose a good model to load according to the log file and pth files saved in ('./BBSNet_cpts/') when training.
model.load_state_dict(torch.load(''))
model.cuda()
model.eval()

#test
test_datasets = ['VT821', 'VT1000']
# test_datasets = ['VT5000']
# test_datasets = ['DES']
for dataset in test_datasets:
    save_path = './test_maps/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path +dataset+'/'+'RGB'+'/'
    gt_root = dataset_path +dataset+'/'+'GT'+'/'
    depth_root = dataset_path +dataset+'/'+'T'+'/'
    test_loader = test_dataset(image_root, gt_root, depth_root, opt.testsize)
    for i in range(test_loader.size):
        image, gt, depth, name, image_for_post = test_loader.load_data()#image_for_post有什么用？
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        depth = depth.cuda()
        res, _, _, _ = model(image, depth)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('save img to: ', save_path+name)
        cv2.imwrite(save_path+name, res*255)
    print('Test Done!')
