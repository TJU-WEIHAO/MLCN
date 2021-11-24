import argparse
import os
import torch
from torch import nn
from torch.nn import functional as F
import torchgeometry as tgm
import numpy as np
import warnings
import cv2
from PIL import Image
from tensorboardX import SummaryWriter
import lpips
from tqdm import tqdm
import scipy.misc
from visualization import board_add_image, board_add_images
from datasets import VITONDataset, VITONDataLoader
from model import SegModule, GMM, Tryon_module, VGGLoss
from utils import gen_noise, Seg_Evalution, Info_txt, get_input, tensor2label
from options import get_opt

warnings.filterwarnings("ignore")
torch.set_printoptions(profile="full")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_label_plain(inputs):
    size = inputs.size()
    pred_batch = []
    for input in inputs:
        input = input.view(1, 13, 256, 192)
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
        pred_batch.append(pred)

    pred_batch = np.array(pred_batch)
    pred_batch = torch.from_numpy(pred_batch)
    label_batch = pred_batch.view(size[0], 1, 256, 192)

    return label_batch


def inference(opt, data, board):
    gauss = tgm.image.GaussianBlur((15, 15), (3, 3))
    gauss.cuda()
    seg_model = SegModule(opt, input_nc=opt.semantic_nc + 7)
    seg_root = os.path.join(opt.checkpoint_dir, opt.seg_checkpoint)
    print('Segmodule loading....', seg_root)
    seg_model.load_network(seg_model, seg_root)
    seg_model.to(device)
    seg_model.eval()
    gmm_model = GMM(opt, inputA_nc=7, inputB_nc=3)
    gmm_root = os.path.join(opt.checkpoint_dir, opt.gmm_checkpoint)
    print('GMM loading....', gmm_root)
    gmm_model.load_network(gmm_model, gmm_root)
    gmm_model.to(device)
    gmm_model.eval()
    tom_model = Tryon_module(input_nc=22)
    tom_root = os.path.join(opt.checkpoint_dir, opt.tom_checkpoint)
    print('TOM loading....', tom_root)
    tom_model.load_network(tom_model, tom_root)
    tom_model.to(device)
    tom_model.eval()
    for i_test, inputs in tqdm(enumerate(data.data_loader)):
        img_names, c_names,change_shape, img, parse, img_agnostic, parse_agnostic, pose, c, cm, pred_parse, warped_cloth, cloth_label = get_input(inputs)
        nosie_size = [1, 1, 256, 192]
        seg_input = torch.cat(
            (c, parse_agnostic, pose, gen_noise(nosie_size)), dim=1)

        seg_output, ce_loss, gan_loss = seg_model(seg_input.to(device), parse.to(device),
                                                  generate_label_plain(parse).to(device))

        parse_cloth_gmm = seg_output[:, 3:4]

        gmm_input = torch.cat((parse_cloth_gmm.to(device), pose.to(device), img_agnostic.to(device)), dim=1)
        warped_grid, warped_refine = gmm_model(gmm_input, c.to(device))
        warped_c = F.grid_sample(c.to(device),  warped_refine, padding_mode='border')
        warped_cm = F.grid_sample(cm.to(device),  warped_refine, padding_mode='border')

        tom_input = torch.cat((img_agnostic.to(device), seg_output, warped_c, pose.to(device)), dim=1).to(device)

        tom_img =  tom_model(tom_input, c.to(device))

        tensor = (tom_img.clone() + 1) * 0.5 * 255
        tensor = tensor.cpu().clamp(0, 255)

        array = tensor.detach().numpy().astype('uint8')
        if array.shape[0] == 1:
            array = array.squeeze(0)
            array = array.swapaxes(0, 1).swapaxes(1, 2)
        pred_root = opt.test_dir + 'pred_root'
        if not os.path.exists(pred_root):
            os.makedirs(pred_root)
        root = pred_root + '/' + c_names[0]
        array = Image.fromarray(array)
        array.save(root)

        img = (img.clone() + 1) * 0.5 * 255
        img = img.cpu().clamp(0, 255)
        img_agnostic = (img_agnostic.clone() + 1) * 0.5 * 255
        img_agnostic = img_agnostic.cpu().clamp(0, 255)
        c = (c.clone() + 1) * 0.5 * 255
        c = c.cpu().clamp(0, 255)
        warped_c = (warped_c.clone() + 1) * 0.5 * 255
        warped_c = warped_c.cpu().clamp(0, 255)
        canvas = torch.FloatTensor(
            1, 3, 256, 192 * 5).fill_(0.5)
        canvas[:, :, 0: 256, 0: 192].copy_(img)
        canvas[:, :, 0: 256, 192: 384].copy_(img_agnostic)
        canvas[:, :, 0: 256, 384: 576].copy_(c)
        canvas[:, :, 0: 256, 576: 768].copy_(warped_c)
        canvas[:, :, 0: 256, 768: 960].copy_(tensor)
        canvas = torch.squeeze(canvas)

        array = canvas.detach().numpy().astype('uint8')
        array = array.swapaxes(0, 1).swapaxes(1, 2)
        pred_root = opt.test_dir + 'vis_root'
        if not os.path.exists(pred_root):
            os.makedirs(pred_root)
        root = pred_root + '/' + c_names[0]
        array = Image.fromarray(array)
        array.save(root)


def main():
    opt = get_opt()
    print('-----------  args  ----------')
    for k in list(vars(opt).keys()):
        print('%s: %s' % (k, vars(opt)[k]))
    print('----------   args  ----------\n')
    print("Start to train stage: %s, named: %s!" % (opt.stage, opt.name))

    #  --------  loading dataset  --------

    train_dataset = VITONDataset(opt)
    train_loader = VITONDataLoader(opt, train_dataset)

    #  --------  visualization  ---------
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.name))

    # create model & train & save the final checkpoint

    if opt.stage == 'inf':
        inference(opt, train_loader, board)

    print('Finished testing %s, nameed: %s!' % (opt.stage, opt.name))


if __name__ == '__main__':
    main()
