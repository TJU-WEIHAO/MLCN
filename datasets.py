import json
from os import path as osp

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils import data
from torchvision import transforms


class VITONDataset(data.Dataset):
    def __init__(self, opt):
        super(VITONDataset, self).__init__()
        self.load_height = opt.load_height  # 输入图片的长
        self.load_width = opt.load_width  # 输入图片的宽
        self.semantic_nc = opt.semantic_nc  # 输入语义分割的通道数=13
        self.data_path = osp.join(opt.dataset_dir, opt.dataset_mode)  # dataset里面包括train，val，test
        self.stage = opt.stage  # 根据不同训练阶段加载不同的数据
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # load data list
        img_names = []
        c_names = []

        with open(osp.join(opt.dataset_dir, opt.dataset_list), 'r') as f:
            for line in f.readlines():
                img_name, c_name = line.strip().split()
                img_names.append(img_name)
                c_names.append(c_name)

        self.img_names = img_names
        self.c_names = c_names

    def get_parse_agnostic(self, parse, pose_data):
        parse_array = np.array(parse)
        parse_upper = ((parse_array == 5).astype(np.float32) +
                       (parse_array == 6).astype(np.float32) +
                       (parse_array == 7).astype(np.float32))

        parse_neck = (parse_array == 10).astype(np.float32)

        r = 10
        agnostic = parse.copy()

        # mask arms
        for parse_id, pose_ids in [(14, [2, 5, 6, 7]), (15, [5, 2, 3, 4])]:
            mask_arm = Image.new('L', (192, 256), 'black')
            mask_arm_draw = ImageDraw.Draw(mask_arm)
            i_prev = pose_ids[0]
            for i in pose_ids[1:]:
                if (pose_data[i_prev, 0] == 0.0 and pose_data[i_prev, 1] == 0.0) or (
                        pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):  # 避免对被遮挡的点（预测为0）
                    continue
                mask_arm_draw.line([tuple(pose_data[j]) for j in [i_prev, i]], 'white', width=r * 10)
                pointx, pointy = pose_data[i]
                radius = r * 4 if i == pose_ids[-1] else r * 15
                mask_arm_draw.ellipse((pointx - radius, pointy - radius, pointx + radius, pointy + radius), 'white',
                                      'white')
                i_prev = i
            parse_arm = (np.array(mask_arm) / 255) * (parse_array == parse_id).astype(np.float32)
            agnostic.paste(0, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))

        # mask torso & neck
        agnostic.paste(0, None, Image.fromarray(np.uint8(parse_upper * 255), 'L'))
        agnostic.paste(0, None, Image.fromarray(np.uint8(parse_neck * 255), 'L'))

        return agnostic

    def get_img_agnostic(self, img, parse, pose_data):
        parse_array = np.array(parse)
        parse_head = ((parse_array == 4).astype(np.float32) +
                      (parse_array == 13).astype(np.float32))
        parse_lower = ((parse_array == 9).astype(np.float32) +
                       (parse_array == 12).astype(np.float32) +
                       (parse_array == 16).astype(np.float32) +
                       (parse_array == 17).astype(np.float32) +
                       (parse_array == 18).astype(np.float32) +
                       (parse_array == 19).astype(np.float32))

        r = 5
        agnostic = img.copy()
        agnostic_draw = ImageDraw.Draw(agnostic)

        '''length_a = np.linalg.norm(pose_data[5] - pose_data[2])  #求范数
        length_b = np.linalg.norm(pose_data[12] - pose_data[9])
        point = (pose_data[9] + pose_data[12]) / 2
        pose_data[9] = point + (pose_data[9] - point) / length_b * length_a
        pose_data[12] = point + (pose_data[12] - point) / length_b * length_a'''

        # mask arms
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'gray', width=r * 10)
        for i in [2, 5]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx - r * 5, pointy - r * 5, pointx + r * 5, pointy + r * 5), 'gray', 'gray')
        for i in [3, 4, 6, 7]:
            if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (
                    pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                continue
            agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'gray', width=r * 10)
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx - r * 5, pointy - r * 5, pointx + r * 5, pointy + r * 5), 'gray', 'gray')

        # mask torso
        for i in [8, 11]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx - r * 3, pointy - r * 6, pointx + r * 3, pointy + r * 6), 'gray', 'gray')
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 8]], 'gray', width=r * 6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [5, 11]], 'gray', width=r * 6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [8, 11]], 'gray', width=r * 12)
        agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 11, 8]], 'gray', 'gray')

        # mask neck
        pointx, pointy = pose_data[1]
        agnostic_draw.rectangle((pointx - r * 7, pointy - r * 7, pointx + r * 7, pointy + r * 7), 'gray', 'gray')
        agnostic.paste(img, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
        agnostic.paste(img, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))

        return agnostic

    def __getitem__(self, index):
        img_name = self.img_names[index]
        cloth_name = self.c_names[index]
        c = Image.open(osp.join(self.data_path, 'cloth', cloth_name)).convert('RGB')
        # c = transforms.Resize(self.load_width, interpolation=2)(c)
        cm = Image.open(osp.join(self.data_path, 'cloth_mask', cloth_name)).convert('L')
        # cm = transforms.Resize(self.load_width, interpolation=0)(cm)

        c = self.transform(c)  # [-1,1]
        cm_array = np.array(cm)
        cm_array = (cm_array >= 128).astype(np.float32)
        cm = torch.from_numpy(cm_array)  # [0,1]
        cm.unsqueeze_(0)
  
        change_shape =  Image.open('/home/weihao/桌面/model_idea/datasets/test/cloth/003434_1.jpg').convert('RGB')
        change_shape = self.transform(change_shape)  # [-1,1]
        # load pose image姿势图
        pose_name = img_name.replace('.jpg', '.png')
        pose_rgb = Image.open(osp.join(self.data_path, 'pose_img', pose_name))
        # pose_rgb = transforms.Resize(self.load_width, interpolation=2)(pose_rgb)
        pose_rgb = self.transform(pose_rgb)  # [-1,1]

        pose_name = img_name.replace('.jpg', '_keypoints.json')
        with open(osp.join(self.data_path, 'pose_json', pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))[:, :2]  # 只获取前两个坐标

        # load parsing image分割图
        parse_name = img_name.replace('.jpg', '.png')
        parse = Image.open(osp.join(self.data_path, 'parse_img', parse_name))
        # parse = transforms.Resize(self.load_width, interpolation=0)(parse)


        parse_agnostic = self.get_parse_agnostic(parse, pose_data)
        parse_agnostic = torch.from_numpy(np.array(parse_agnostic)[None]).long()

        labels = {
            0: ['background', [0, 10]],
            1: ['hair', [1, 2]],
            2: ['face', [4, 13]],
            3: ['upper', [5, 6, 7]],
            4: ['bottom', [9, 12]],
            5: ['left_arm', [14]],
            6: ['right_arm', [15]],
            7: ['left_leg', [16]],
            8: ['right_leg', [17]],
            9: ['left_shoe', [18]],
            10: ['right_shoe', [19]],
            11: ['socks', [8]],
            12: ['noise', [3, 11]]
        }
        parse_agnostic_map = torch.zeros(20, self.load_height, self.load_width, dtype=torch.float)
        parse_agnostic_map.scatter_(0, parse_agnostic, 1.0)
        new_parse_agnostic_map = torch.zeros(self.semantic_nc, self.load_height, self.load_width, dtype=torch.float)
        for i in range(len(labels)):
            for label in labels[i][1]:
                new_parse_agnostic_map[i] += parse_agnostic_map[label]

        # -------------  load person image  -------------
        img = Image.open(osp.join(self.data_path, 'image', img_name))
        img_agnostic = self.get_img_agnostic(img, parse, pose_data)
        img = self.transform(img)  # [-1,1]
        img_agnostic = self.transform(img_agnostic)  # [-1,1]

        # ------------  load cloth label  ---------
        if self.stage == 'gmm' or self.stage == 'tom' or self.stage == 'flow':
            im_parse = np.array(parse)
            parse_cloth = ((im_parse == 5).astype(np.float32) + (im_parse == 6).astype(np.float32)+ (im_parse == 7).astype(np.float32))
            pcm = torch.from_numpy(parse_cloth)  # [0,1]
            cloth_label = img * pcm + (1 - pcm)
        else:
            cloth_label = ''

        # -------------  transform parse to one-hot  ---------------
        parse = torch.from_numpy(np.array(parse)[None]).long()
        parse_map = torch.zeros(20, self.load_height, self.load_width, dtype=torch.float)
        parse_map.scatter_(0, parse, 1.0)
        new_parse_map = torch.zeros(self.semantic_nc, self.load_height, self.load_width, dtype=torch.float)
        for i in range(len(labels)):  # return parse one-hot
            for label in labels[i][1]:
                if i == 0:
                    new_parse_map[i] += parse_map[label]
                else:
                    new_parse_map[i] += parse_map[label] / i

        # loading predited image parse
        if self.stage == 'gmm' or self.stage == 'tom'or self.stage == 'flow':
            pred_parse = Image.open(osp.join(self.data_path, 'pred_parse', img_name))
            pred_parse = torch.from_numpy(np.array(pred_parse)[None]).long()

        else:
            pred_parse = ''

        if self.stage == 'tom'or self.stage == 'flow':
            warped_cloth = Image.open(osp.join(self.data_path, 'wraped_c', img_name))
            warped_cloth = self.transform(warped_cloth)
        else:
            warped_cloth = ''

        result = {
            'img_name': img_name,
            'cloth_name': cloth_name,
            'change_shape':change_shape,
            'img': img,
            'img_parse': new_parse_map,
            'img_agnostic': img_agnostic,
            'parse_agnostic': new_parse_agnostic_map,
            'pose': pose_rgb,
            'cloth': c,
            'cloth_mask': cm,
            'pred_parse': pred_parse,
            'warped_cloth': warped_cloth,
            'cloth_label':cloth_label,
        }
        return result

    def __len__(self):
        return len(self.img_names)


class VITONDataLoader:
    def __init__(self, opt, dataset):
        super(VITONDataLoader, self).__init__()

        if opt.shuffle:
            train_sampler = data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = data.DataLoader(
            dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
            num_workers=opt.workers, pin_memory=True, drop_last=True, sampler=train_sampler
        )
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch
