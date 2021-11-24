import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn.utils.spectral_norm import spectral_norm
from .Basemodel import BaseNetwork
from .modules import DeformConv
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DT(nn.Module):
    def __init__(self):
        super(DT, self).__init__()

    def forward(self, x1, x2):
        dt = torch.abs(x1 - x2)
        return dt


class DT2(nn.Module):
    def __init__(self):
        super(DT, self).__init__()

    def forward(self, x1, y1, x2, y2):
        dt = torch.sqrt(torch.mul(x1 - x2, x1 - x2) +
                        torch.mul(y1 - y2, y1 - y2))
        return dt


class GicLoss(nn.Module):
    def __init__(self, opt):
        super(GicLoss, self).__init__()
        self.dT = DT()
        self.opt = opt

    def forward(self, grid):
        Gx = grid[:, :, :, 0]
        Gy = grid[:, :, :, 1]
        Gxcenter = Gx[:, 1:256 - 1, 1:192 - 1]
        Gxup = Gx[:, 0:256 - 2, 1:192 - 1]
        Gxdown = Gx[:, 2:256, 1:192 - 1]
        Gxleft = Gx[:, 1:256 - 1, 0:192 - 2]
        Gxright = Gx[:, 1:256 - 1, 2:192]

        Gycenter = Gy[:, 1:256 - 1, 1:192 - 1]
        Gyup = Gy[:, 0:256 - 2, 1:192 - 1]
        Gydown = Gy[:, 2:256, 1:192 - 1]
        Gyleft = Gy[:, 1:256 - 1, 0:192 - 2]
        Gyright = Gy[:, 1:256 - 1, 2:192]

        dtleft = self.dT(Gxleft, Gxcenter)
        dtright = self.dT(Gxright, Gxcenter)
        dtup = self.dT(Gyup, Gycenter)
        dtdown = self.dT(Gydown, Gycenter)

        return torch.sum(torch.abs(dtleft - dtright) + torch.abs(dtup - dtdown))


# ----------------------------------------------------------------------------------------------------------------------
#                                                  GMM-related classes
# ----------------------------------------------------------------------------------------------------------------------
class FeatureExtraction(BaseNetwork):
    def __init__(self, input_nc, ngf=64, num_layers=4, norm_layer=nn.BatchNorm2d):
        super(FeatureExtraction, self).__init__()

        nf = ngf
        self.layer_01 = nn.Sequential(
            *[nn.Conv2d(input_nc, nf, kernel_size=4, stride=2, padding=1), nn.ReLU(), norm_layer(nf)])
        self.layer_02 = nn.Sequential(
            *[nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.ReLU(), norm_layer(128)])
        self.layer_03 = nn.Sequential(
            *[nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), nn.ReLU(), norm_layer(256)])
        self.layer_04 = nn.Sequential(
            *[nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), nn.ReLU(), norm_layer(512)])
        self.layer_05 = nn.Sequential(
            *[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(), norm_layer(512)])
        self.layer_06 = nn.Sequential(*[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU()])

        self.init_weights()

    def forward(self, x):
        layer1 = self.layer_01(x)
        layer2 = self.layer_02(layer1)
        layer3 = self.layer_03(layer2)
        layer4 = self.layer_04(layer3)
        output = self.layer_06(self.layer_05(layer4))
        return output, layer4, layer3, layer2, layer1


class FeatureCorrelation(nn.Module):
    def __init__(self):
        super(FeatureCorrelation, self).__init__()

    def forward(self, featureA, featureB):
        # Reshape features for matrix multiplication.
        b, c, h, w = featureA.size()
        featureA = featureA.permute(0, 3, 2, 1).reshape(b, w * h, c)
        featureB = featureB.reshape(b, c, h * w)

        # Perform matrix multiplication.
        corr = torch.bmm(featureA, featureB).reshape(b, w * h, h, w)
        return corr


class FeatureRegression(nn.Module):
    def __init__(self, input_nc=512, output_size=6, norm_layer=nn.BatchNorm2d):
        super(FeatureRegression, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_nc, 512, kernel_size=4, stride=2, padding=1), norm_layer(512), nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=4, stride=2, padding=1), norm_layer(256), nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1), norm_layer(128), nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1), norm_layer(64), nn.ReLU()
        )
        self.linear = nn.Linear(64 * 4 * 3, output_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv(x)
        x = self.linear(x.view(x.size(0), -1))
        theta = self.tanh(x)

        return theta


class TpsGridGen(nn.Module):
    def __init__(self, opt, load_width, load_height, dtype=torch.float):
        super(TpsGridGen, self).__init__()

        # Create a grid in numpy.
        # TODO: set an appropriate interval ([-1, 1] in CP-VTON, [-0.9, 0.9] in the current version of VITON-HD)
        grid_X, grid_Y = np.meshgrid(np.linspace(-0.9, 0.9, load_width), np.linspace(-0.9, 0.9, load_height))
        # grid_X, grid_Y = np.meshgrid(np.linspace(-0.9, 0.9, 100), np.linspace(-0.9, 0.9, 100))
        grid_X = torch.tensor(grid_X, dtype=dtype).unsqueeze(0).unsqueeze(3)  # size: (1, h, w, 1)
        grid_Y = torch.tensor(grid_Y, dtype=dtype).unsqueeze(0).unsqueeze(3)  # size: (1, h, w, 1)

        # Initialize the regular grid for control points P.
        self.N = opt.grid_size * opt.grid_size
        coords = np.linspace(-0.9, 0.9, opt.grid_size)
        # FIXME: why P_Y and P_X are swapped?
        P_Y, P_X = np.meshgrid(coords, coords)
        P_X = torch.tensor(P_X, dtype=dtype).reshape(self.N, 1)
        P_Y = torch.tensor(P_Y, dtype=dtype).reshape(self.N, 1)
        P_X_base = P_X.clone()
        P_Y_base = P_Y.clone()

        Li = self.compute_L_inverse(P_X, P_Y).unsqueeze(0)
        P_X = P_X.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)  # size: (1, 1, 1, 1, self.N)
        P_Y = P_Y.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)  # size: (1, 1, 1, 1, self.N)

        self.register_buffer('grid_X', grid_X, False)
        self.register_buffer('grid_Y', grid_Y, False)
        self.register_buffer('P_X_base', P_X_base, False)
        self.register_buffer('P_Y_base', P_Y_base, False)
        self.register_buffer('Li', Li, False)
        self.register_buffer('P_X', P_X, False)
        self.register_buffer('P_Y', P_Y, False)

    # TODO: refactor
    def compute_L_inverse(self, X, Y):
        N = X.size()[0]  # num of points (along dim 0)
        # construct matrix K
        Xmat = X.expand(N, N)
        Ymat = Y.expand(N, N)
        P_dist_squared = torch.pow(Xmat - Xmat.transpose(0, 1), 2) + torch.pow(Ymat - Ymat.transpose(0, 1), 2)
        P_dist_squared[P_dist_squared == 0] = 1  # make diagonal 1 to avoid NaN in log computation
        K = torch.mul(P_dist_squared, torch.log(P_dist_squared))
        # construct matrix L
        O = torch.FloatTensor(N, 1).fill_(1)
        Z = torch.FloatTensor(3, 3).fill_(0)
        P = torch.cat((O, X, Y), 1)
        L = torch.cat((torch.cat((K, P), 1), torch.cat((P.transpose(0, 1), Z), 1)), 0)
        Li = torch.inverse(L)
        return Li

    # TODO: refactor
    def apply_transformation(self, theta, points):
        if theta.dim() == 2:
            theta = theta.unsqueeze(2).unsqueeze(3)
        # points should be in the [B,H,W,2] format,
        # where points[:,:,:,0] are the X coords
        # and points[:,:,:,1] are the Y coords

        # input are the corresponding control points P_i
        batch_size = theta.size()[0]
        # split theta into point coordinates
        Q_X = theta[:, :self.N, :, :].squeeze(3)
        Q_Y = theta[:, self.N:, :, :].squeeze(3)
        Q_X = Q_X + self.P_X_base.expand_as(Q_X)
        Q_Y = Q_Y + self.P_Y_base.expand_as(Q_Y)

        # get spatial dimensions of points
        points_b = points.size()[0]
        points_h = points.size()[1]
        points_w = points.size()[2]

        # repeat pre-defined control points along spatial dimensions of points to be transformed
        P_X = self.P_X.expand((1, points_h, points_w, 1, self.N))
        P_Y = self.P_Y.expand((1, points_h, points_w, 1, self.N))

        # compute weigths for non-linear part
        W_X = torch.bmm(self.Li[:, :self.N, :self.N].expand((batch_size, self.N, self.N)), Q_X)
        W_Y = torch.bmm(self.Li[:, :self.N, :self.N].expand((batch_size, self.N, self.N)), Q_Y)
        # reshape
        # W_X,W,Y: size [B,H,W,1,N]
        W_X = W_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)
        W_Y = W_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)
        # compute weights for affine part
        A_X = torch.bmm(self.Li[:, self.N:, :self.N].expand((batch_size, 3, self.N)), Q_X)
        A_Y = torch.bmm(self.Li[:, self.N:, :self.N].expand((batch_size, 3, self.N)), Q_Y)
        # reshape
        # A_X,A,Y: size [B,H,W,1,3]
        A_X = A_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)
        A_Y = A_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)

        # compute distance P_i - (grid_X,grid_Y)
        # grid is expanded in point dim 4, but not in batch dim 0, as points P_X,P_Y are fixed for all batch
        points_X_for_summation = points[:, :, :, 0].unsqueeze(3).unsqueeze(4).expand(
            points[:, :, :, 0].size() + (1, self.N))
        points_Y_for_summation = points[:, :, :, 1].unsqueeze(3).unsqueeze(4).expand(
            points[:, :, :, 1].size() + (1, self.N))

        if points_b == 1:
            delta_X = points_X_for_summation - P_X
            delta_Y = points_Y_for_summation - P_Y
        else:
            # use expanded P_X,P_Y in batch dimension
            delta_X = points_X_for_summation - P_X.expand_as(points_X_for_summation)
            delta_Y = points_Y_for_summation - P_Y.expand_as(points_Y_for_summation)

        dist_squared = torch.pow(delta_X, 2) + torch.pow(delta_Y, 2)
        # U: size [1,H,W,1,N]
        dist_squared[dist_squared == 0] = 1  # avoid NaN in log computation
        U = torch.mul(dist_squared, torch.log(dist_squared))

        # expand grid in batch dimension if necessary
        points_X_batch = points[:, :, :, 0].unsqueeze(3)
        points_Y_batch = points[:, :, :, 1].unsqueeze(3)
        if points_b == 1:
            points_X_batch = points_X_batch.expand((batch_size,) + points_X_batch.size()[1:])
            points_Y_batch = points_Y_batch.expand((batch_size,) + points_Y_batch.size()[1:])

        points_X_prime = A_X[:, :, :, :, 0] + \
                         torch.mul(A_X[:, :, :, :, 1], points_X_batch) + \
                         torch.mul(A_X[:, :, :, :, 2], points_Y_batch) + \
                         torch.sum(torch.mul(W_X, U.expand_as(W_X)), 4)

        points_Y_prime = A_Y[:, :, :, :, 0] + \
                         torch.mul(A_Y[:, :, :, :, 1], points_X_batch) + \
                         torch.mul(A_Y[:, :, :, :, 2], points_Y_batch) + \
                         torch.sum(torch.mul(W_Y, U.expand_as(W_Y)), 4)

        return torch.cat((points_X_prime, points_Y_prime), 3)

    def forward(self, theta):
        warped_grid = self.apply_transformation(theta, torch.cat((self.grid_X, self.grid_Y), 3))
        return warped_grid


class Feature_encoder(nn.Module):
    def __init__(self, input_fea=1, out_c=128, input_cor=2):
        super(Feature_encoder, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.conv0 = nn.Sequential(nn.Conv2d(input_fea, out_c, kernel_size=3, padding=1), norm_layer(out_c), nn.ReLU())
        self.conv1 = nn.Sequential(nn.Conv2d(input_cor, out_c, kernel_size=3, padding=1), norm_layer(out_c), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(2 * out_c, out_c, kernel_size=3, padding=1), norm_layer(out_c), nn.ReLU())

    def forward(self, feature, cor):
        cor = cor.permute(0, 3, 1, 2)
        conv0 = self.conv0(feature)
        conv1 = self.conv1(cor)
        output = self.conv2(torch.cat((conv0, conv1), 1))
        return output


class Pred_head(nn.Module):
    def __init__(self, input_c=1, output_c=2):
        super(Pred_head, self).__init__()
        self.conv0 = nn.Sequential(nn.Conv2d(input_c, output_c, kernel_size=3, padding=1))
        self.conv1 = nn.Sequential(nn.Conv2d(input_c, 2, kernel_size=1))

    def forward(self, feature):
        d_offset = self.conv0(feature)
        f_offset = self.conv1(feature)
        return d_offset, f_offset


class GMM(BaseNetwork):
    def __init__(self, opt, inputA_nc, inputB_nc):
        super(GMM, self).__init__()

        self.extractionA = FeatureExtraction(inputA_nc, ngf=64, num_layers=4)
        self.extractionB = FeatureExtraction(inputB_nc, ngf=64, num_layers=4)
        self.correlation = FeatureCorrelation()
        self.regression = FeatureRegression(input_nc=(opt.load_width // 16) * (opt.load_height // 16),
                                            output_size=2 * opt.grid_size ** 2)
        self.gridGen_0 = TpsGridGen(opt, 24, 32)
        self.gridGen_1 = TpsGridGen(opt, 192, 256)

        norm_layer = nn.BatchNorm2d
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'))


        self.fea_de_03 = Feature_encoder(512, 128, 2)
        self.pred_head_03 = Pred_head(128, 8 * 18)
        self.dconv_3 = DeformConv(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1, groups=2,
                                  deformable_groups=8).cuda()

        self.fea_de_02 = Feature_encoder(256, 64, 2)
        self.pred_head_02 = Pred_head(64, 4 * 18)
        self.dconv_2 = DeformConv(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1, groups=2,
                                  deformable_groups=4).cuda()

        self.fea_de_01 = Feature_encoder(128, 64, 2)
        self.pred_conv = nn.Sequential(nn.Conv2d(64, 2, kernel_size=1))

        self.print_network()

    def forward(self, inputA, inputB):
        featureA, outA4, outA3, outA2, outA1 = self.extractionA(inputA)
        featureA = F.normalize(featureA, dim=1)
        featureB, outB4, outB3, outB2, outB1 = self.extractionB(inputB)
        featureB = F.normalize(featureB, dim=1)

        corr = self.correlation(featureA, featureB)
        theta = self.regression(corr)
        warped_grid = self.gridGen_0(theta)
        warped_grid_01 = self.gridGen_1(theta)

        warped_3 = F.grid_sample(outB3, warped_grid, padding_mode='border')
        stage_3 = torch.cat((warped_3, outA3), 1)
        fea_3 = self.fea_de_03(stage_3, warped_grid)
        fea_3 = self.up(fea_3)
        d_off_03, f_off_03 = self.pred_head_03(fea_3)

        warped_2 = self.dconv_3(outB2, d_off_03)

        flow_2 = f_off_03.permute(0, 2, 3, 1)
        stage_2 = torch.cat((warped_2, outA2), 1)
        fea_2 = self.fea_de_02(stage_2, flow_2)
        fea_2 = self.up(fea_2)
        d_off_02, f_off_02 = self.pred_head_02(fea_2)

        warped_1 = self.dconv_2(outB1, d_off_02)

        flow_1 = f_off_02.permute(0, 2, 3, 1)
        stage_1 = torch.cat((warped_1, outA1), 1)
        flow_1 = self.fea_de_01(stage_1, flow_1)

        flow = self.pred_conv(flow_1)

        warped_refine = self.up(flow).permute(0, 2, 3, 1) + warped_grid_01

        return warped_grid_01, warped_refine
