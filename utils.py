import os

import cv2
import numpy as np
from PIL import Image
import torch
import prettytable as pt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_input(inputs):
    img_names = inputs['img_name']
    c_names = inputs['cloth_name']
    change_shape = inputs['change_shape']
    img = inputs['img']
    parse = inputs['img_parse']
    img_agnostic = inputs['img_agnostic']
    parse_agnostic = inputs['parse_agnostic']
    pose = inputs['pose']
    c = inputs['cloth']
    cm = inputs['cloth_mask']
    pred_parse = inputs['pred_parse']
    warped_cloth = inputs['warped_cloth']
    cloth_label = inputs['cloth_label']

    return img_names, c_names, change_shape, img, parse, img_agnostic, parse_agnostic, pose, c, cm, pred_parse, warped_cloth, cloth_label


def gen_noise(shape):
    noise = np.zeros(shape, dtype=np.uint8)
    ### noise
    noise = cv2.randn(noise, 0, 255)
    noise = np.asarray(noise / 255, dtype=np.uint8)
    noise = torch.tensor(noise, dtype=torch.float32)
    return noise


def save_images(img_tensors, img_names, save_dir):
    for img_tensor, img_name in zip(img_tensors, img_names):
        tensor = (img_tensor.clone() + 1) * 0.5 * 255
        tensor = tensor.cpu().clamp(0, 255)

        try:
            array = tensor.numpy().astype('uint8')
        except:
            array = tensor.detach().numpy().astype('uint8')

        if array.shape[0] == 1:
            array = array.squeeze(0)
        elif array.shape[0] == 3:
            array = array.swapaxes(0, 1).swapaxes(1, 2)

        im = Image.fromarray(array)
        im.save(os.path.join(save_dir, img_name), format='JPEG')


def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise ValueError("'{}' is not a valid checkpoint path".format(checkpoint_path))
    model.load_state_dict(torch.load(checkpoint_path))


class Seg_Evalution():
    def __init__(self, pred, label):
        self.metrcis = {}
        self.pred = pred
        self.label = label

    def ConfusionMatrix(self, numClass, imgPredict, Label):
        #  返回混淆矩阵
        mask = (Label >= 0) & (Label < numClass)
        label = numClass * Label[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=numClass ** 2)
        confusionMatrix = count.reshape(numClass, numClass)
        return confusionMatrix

    def OverallAccuracy(self, confusionMatrix):
        #  返回所有类的整体像素精度OA
        # acc = (TP + TN) / (TP + TN + FP + TN)
        OA = np.diag(confusionMatrix).sum() / confusionMatrix.sum()
        return OA

    def Precision(self, confusionMatrix):
        #  返回所有类别的精确率precision
        precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis=0)
        return precision

    def Recall(self, confusionMatrix):
        #  返回所有类别的召回率recall
        recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis=1)
        return recall

    def F1Score(self, confusionMatrix):
        precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis=0)
        recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis=1)
        f1score = 2 * precision * recall / (precision + recall)
        return f1score

    def IntersectionOverUnion(self, confusionMatrix):
        #  返回交并比IoU
        intersection = np.diag(confusionMatrix)
        union = np.sum(confusionMatrix, axis=1) + np.sum(confusionMatrix, axis=0) - np.diag(confusionMatrix)
        IoU = intersection / union
        return IoU

    def MeanIntersectionOverUnion(self, confusionMatrix):
        #  返回平均交并比mIoU
        intersection = np.diag(confusionMatrix)
        union = np.sum(confusionMatrix, axis=1) + np.sum(confusionMatrix, axis=0) - np.diag(confusionMatrix)
        IoU = intersection / union
        mIoU = np.nanmean(IoU)
        return mIoU

    def Frequency_Weighted_Intersection_over_Union(self, confusionMatrix):
        #  返回频权交并比FWIoU
        freq = np.sum(confusionMatrix, axis=1) / np.sum(confusionMatrix)
        iu = np.diag(confusionMatrix) / (
                np.sum(confusionMatrix, axis=1) +
                np.sum(confusionMatrix, axis=0) -
                np.diag(confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def get_metrcis(self):
        confusionMatrix = self.ConfusionMatrix(13, self.pred, self.label)
        precision = self.Precision(confusionMatrix)
        recall = self.Recall(confusionMatrix)
        OA = self.OverallAccuracy(confusionMatrix)
        IoU = self.IntersectionOverUnion(confusionMatrix)
        mIOU = self.MeanIntersectionOverUnion(confusionMatrix)
        f1ccore = self.F1Score(confusionMatrix)
        self.metrcis = {'Precision': precision, 'Recall': recall, 'OA': OA, 'IoU': IoU,
                        'mIoU': mIOU, 'F1': f1ccore}

        return self.metrcis


class Info_txt():
    # 定义txt类，用于储存打印信息
    def __init__(self, opt):
        self.opt = opt
        self.mode = opt.dataset_mode
        if os.path.exists(opt.save_dir + self.mode) == False:
            os.mkdir(opt.save_dir + self.mode)
        self.model_info = opt.save_dir + self.mode + '/model_info.txt'
        self.loss_info = opt.save_dir + self.mode + '/loss_info.txt'
        self.metrcis_info = opt.save_dir + self.mode + '/metrcis_info.txt'
        self.tb = pt.PrettyTable()

    def loss_write(self, context):
        # 写入打印内容
        with open(self.loss_info, 'a') as f:
            f.write(context + '\n')
        f.close()

    def metric_write(self, context):
        # 写入打印内容
        with open(self.metrcis_info, 'a') as f:
            f.write(context + '\n')
        f.close()

    def model_write(self):
        file = open(self.model_info, 'a')
        file.write('Model name : ' + self.opt.name + '\n')
        file.write('Iter_step : ' + str(self.opt.keep_step + self.opt.decay_step) + '\n')
        file.write('Batch_size : ' + str(self.opt.batch_size) + '\n')
        file.write('Save_root  : ' + str(self.opt.save_dir) + '\n')
        file.close()

def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def labelcolormap(N):
    if N == 35: # cityscape
        cmap = np.array([(  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (111, 74,  0), ( 81,  0, 81),
                     (128, 64,128), (244, 35,232), (250,170,160), (230,150,140), ( 70, 70, 70), (102,102,156), (190,153,153),
                     (180,165,180), (150,100,100), (150,120, 90), (153,153,153), (153,153,153), (250,170, 30), (220,220,  0),
                     (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
                     (  0, 60,100), (  0,  0, 90), (  0,  0,110), (  0, 80,100), (  0,  0,230), (119, 11, 32), (  0,  0,142)],
                     dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7-j))
                g = g ^ (np.uint8(str_id[-2]) << (7-j))
                b = b ^ (np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap

class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image

def tensor2label(label_tensor, n_label, imtype=np.uint8):
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    #label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    label_numpy = label_tensor.numpy()
    label_numpy = label_numpy / 255.0

    return label_numpy

def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy

    image_numpy = image_tensor.cpu().float().numpy()
    #if normalize:
    #    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    #else:
    #    image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = (image_numpy + 1) / 2.0
    image_numpy = np.clip(image_numpy, 0, 1)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:
        image_numpy = image_numpy[:,:,0]

    return image_numpy
