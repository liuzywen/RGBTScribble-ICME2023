import os
from PIL import Image
import torch.utils.data as data
import numpy as np
import torch
import torchvision.transforms as transforms
import cv2
from fast_slic import Slic
class SalObjDataset(data.Dataset):
    def __init__(self, image_root,depth_root, gt_root, mask_root, gray_root,edge_root,trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.png') or f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.masks = [mask_root + f for f in os.listdir(mask_root) if f.endswith('.png')]
        self.grays = [gray_root + f for f in os.listdir(gray_root) if f.endswith('.png')]
        self.edges = [edge_root + f for f in os.listdir(edge_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.masks = sorted(self.masks)
        self.grays = sorted(self.grays)
        self.edges = sorted(self.edges)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.depth_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.mask_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.gray_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.edge_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        depth = self.rgb_loader(self.depths[index])
        gt = self.binary_loader(self.gts[index])
        mask = self.binary_loader(self.masks[index])
        gray = self.binary_loader(self.grays[index])
        edge = self.binary_loader(self.edges[index])
        name = self.images[index].split('/')[-1]
        np_img = np.array(image)
        np_img = cv2.resize(np_img, dsize=(self.trainsize, self.trainsize), interpolation=cv2.INTER_LINEAR)

        np_depth = np.array(depth)
        np_depth = cv2.resize(np_depth, dsize=(self.trainsize, self.trainsize), interpolation=cv2.INTER_LINEAR)

        np_gt = np.array(gt)
        np_gt = cv2.resize(np_gt, dsize=(self.trainsize, self.trainsize), interpolation=cv2.INTER_LINEAR) / 255

        np_mask = np.array(mask)
        np_mask = cv2.resize(np_mask, dsize=(self.trainsize, self.trainsize), interpolation=cv2.INTER_LINEAR) / 255

        slic = Slic(num_components=40, compactness=10)
        SS_map = slic.iterate(np_img)
        SS_map_depth = slic.iterate(np_depth)

        SS_map = SS_map + 1
        SS_map_depth = SS_map_depth + 1

        SS_maps_label = []
        SS_maps = []

        # SS_maps_label_mask = []
        # SS_maps_mask = []

        SS_maps_label_depth = []
        SS_maps_depth = []

        # SS_maps_label_mask_depth = []
        # SS_maps_mask_depth = []

        label_gt = np.zeros((1, self.trainsize, self.trainsize))
        # label_mask = np.zeros((1, self.trainsize, self.trainsize))

        label_gt_depth = np.zeros((1, self.trainsize, self.trainsize))
        # label_mask_depth = np.zeros((1, self.trainsize, self.trainsize))

        for i in range(1, 40+ 1):
            buffer = np.copy(SS_map)
            buffer[buffer != i] = 0
            buffer[buffer == i] = 1

            if np.sum(buffer) != 0:
                if np.sum(buffer * np_gt) > 1:
                    label_gt = label_gt+buffer
                    SS_maps_label.append(1)

                else:
                    SS_maps_label.append(0)
            else:
                SS_maps_label.append(0)
            SS_maps.append(buffer)
        label_gt = torch.tensor(label_gt)
        label_gt = label_gt.to(torch.float32)



        for i in range(1, 40 + 1):
            buffer = np.copy(SS_map_depth)
            buffer[buffer != i] = 0
            buffer[buffer == i] = 1

            if np.sum(buffer) != 0:
                if np.sum(buffer * np_gt) > 1:
                    label_gt_depth = label_gt_depth+buffer
                    SS_maps_label_depth.append(1)

                else:
                    SS_maps_label_depth.append(0)
            else:
                SS_maps_label_depth.append(0)
            SS_maps_depth.append(buffer)
        label_gt_depth = torch.tensor(label_gt_depth)
        label_gt_depth = label_gt_depth.to(torch.float32)



        image = self.img_transform(image)
        depth = self.depth_transform(depth)
        gt = self.gt_transform(gt)
        mask = self.mask_transform(mask)
        gray = self.gray_transform(gray)
        edge = self.edge_transform(edge)
        return image, depth, gt, mask, gray, edge, label_gt, label_gt_depth,name

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        depths = []
        gts = []
        masks = []
        grays = []
        edges = []
        for img_path,depth_path, gt_path, mask_path, gray_path,edge_path in zip(self.images, self.depths, self.gts, self.masks, self.grays,self.edges):
            img = Image.open(img_path)
            depth = Image.open(depth_path)
            gt = Image.open(gt_path)
            mask = Image.open(mask_path)
            gray = Image.open(gray_path)
            edge = Image.open(edge_path)
            if img.size == gt.size:
                images.append(img_path)
                depths.append(depth_path)
                gts.append(gt_path)
                masks.append(mask_path)
                grays.append(gray_path)
                edges.append(edge_path)
        self.images = images
        self.depths = depths
        self.gts = gts
        self.masks = masks
        self.grays = grays
        self.edges = edges

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_loader(image_root,depth_root, gt_root, mask_root, gray_root,edge_root, batchsize, trainsize, shuffle=True, num_workers=0, pin_memory=True):

    dataset = SalObjDataset(image_root, depth_root, gt_root, mask_root, gray_root, edge_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

class test_dataset:
    def __init__(self, image_root, gt_root,depth_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.depths=[depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp')
                    or f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths=sorted(self.depths)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        # self.gt_transform = transforms.Compose([
        #     transforms.Resize((self.trainsize, self.trainsize)),
        #     transforms.ToTensor()])
        self.depths_transform = transforms.Compose([transforms.Resize((self.testsize, self.testsize)),transforms.ToTensor()])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        depth=self.rgb_loader(self.depths[self.index])
        depth=self.depths_transform(depth).unsqueeze(0)
        name = self.images[self.index].split('/')[-1]
        image_for_post=self.rgb_loader(self.images[self.index])
        image_for_post=image_for_post.resize(gt.size)
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.jpg'
        self.index += 1
        self.index = self.index % self.size
        return image, gt,depth, name,np.array(image_for_post)


    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

