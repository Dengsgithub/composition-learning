import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import cv2
import argparse
import numpy as np
import logging
import time
import torch
from torch.nn import MSELoss
from torch.nn import L1Loss
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from math import log10
from Net import parameterNet, parameterNet_mlp, parameterNet_linear, parameterNet_pure, parameterNet_conv3
from dataset import TrainDataset, TestDataset
from cal_ssim import SSIM
from data_generate import data_g_iter, data_g_no_iter
from skimage.metrics import peak_signal_noise_ratio as cal_psnr
import winsound



def get_args():
    parser = argparse.ArgumentParser(description="train derain model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train_dir", type=str, default='./dataset/train',
                        help="train image dir")
    parser.add_argument("--test_dir", type=str, default='./dataset/test',
                        help="test image dir")
    parser.add_argument("--log_dir", type=str, default='./logdir',
                        help="log_dir")
    parser.add_argument("--model_dir", type=str, default='./models',
                        help="test image dir")
    parser.add_argument("--image_size", type=int, default=300,
                        help="training patch size")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="batch size")
    parser.add_argument("--epochs", type=int, default=140,
                        help="number of epochs")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="numworks in dataloader")
    parser.add_argument("--aug_data", type=bool, default=False,
                        help="whether to augment data")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate")
    parser.add_argument("--loss", type=str, default="MSE",
                        help="loss; MSE', 'L1Loss', or 'MyLoss' is expected")
    parser.add_argument("--opt", type=str, default="Adam",
                        help="Optimizer for updating the network parameters")
    parser.add_argument("--checkpoint", type=str, default="the_end",
                        help="model architecture ('Similarity')")

    ### usually adjust####
    parser.add_argument("--sessname", type=str, default='4basis_0.3',#"30biggerbasis_linear_0.01",
                        help="different session names for parameter modification")
    parser.add_argument("--noise_var", type=float, default=0.05,
                        help="var = arg.noise_var,for random_noise(mode='gaussian')")
    parser.add_argument("--Net", type=str, default="parameterNet_linear",
                        help="choice of Network: parameterNet_mlp, parameterNet_linear ")

    ####判断是否使用特定参数， 默认为使用特定参数
    parser.add_argument("--If_sp", type=bool, default=True, help="If normalizing the image")

    ####使用迭代的方法的参数
    parser.add_argument("--sigma_c", type=list, default=[0.1, 0.6, 1.1],#[0.1, 0.4, 0.7, 1, 2],
                        help="sigma_c = [0.1, 0.3, 0.5, 0.7, 1],[0.1, 0.3, 0.4, 0.5, 0.7, 1] ,[0.1, 0.4, 0.7, 1, 2],[0.1, 0.2, 0.3, 0.4, 0.5]")
    parser.add_argument("--sigma_s", type=list, default=[0.5, 2, 3.5],
                        help=" sigma_s = [0.5, 1, 2],[0.5, 2, 3.5] ")
    parser.add_argument("--size", type=list, default=[15],#[5, 9],
                        help=" size = [5] ,[3, 5, 7],[5, 9],[7],[5, 9, 13]")

    ####指定参数的方法 参数规则[sigma_c, sigma_s, size]

    parser.add_argument("--specified_para", type=list, default=[

       [3,3], [5,5],[7,7],[9,9]
    ], help="[sigma_c, sigma_s, size]")
    #[[0.5, 1.5, 15],[0.1, 2, 15],[1, 3, 15],[0.1, 0.5,15],[0.3,2,15],[0.3,3,15],[0.5,3,15],[2,4,15],[0.1,1,15]
    # [0.05,0.5,15], [0.1,0.5,15],[0.1,0.7,15],[0.1,2,15],[0.1,3,15], [0.1,3,15], [0.3,1,15], [0.3,2,15], [0.3,5,15], [0.5,2,15],[0.5,5,15],[1,1.5,15], [1,3,15],[1,5,15],[3,5,15]

    args = parser.parse_args()
    return args


def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

class Session:
    def __init__(self, args, number):
        self.log_dir = args.log_dir
        self.model_dir = args.model_dir
        ensure_dir(self.log_dir)
        ensure_dir(self.model_dir)
        logger.info('set log dir as %s' % self.log_dir)
        logger.info('set model dir as %s' % self.model_dir)

        if args.Net=="parameterNet_linear":
            # self.net = parameterNet_linear(in_channel=number+3, out_channel=3).cuda()
            self.net = parameterNet_linear(in_channel=number, out_channel=3).cuda()
        elif args.Net=='parameterNet_mlp':
            self.net = parameterNet_mlp(in_channel=number, out_channel=3).cuda()
        elif args.Net=='parameterNet_conv3':
            self.net = parameterNet_conv3(in_channel=number, out_channel=3).cuda()
        self.ssim = SSIM().cuda()
        self.image_size = args.image_size
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        self.step = 0
        self.epoch = args.epochs
        self.now_epoch = 0
        self.start_epoch = 0
        self.writers = {}
        self.total_step = 0

        self.sessname = args.sessname

        self.crit = MSELoss().cuda()

        if args.opt == "SGD":
            self.opt = SGD(self.net.parameters(), lr=args.lr)
        else:
            self.opt = Adam(self.net.parameters(), lr=args.lr)

        # self.sche = MultiStepLR(self.opt, milestones=[100, 200, 300, 400, 500], gamma=0.5)

    def tensorboard(self, name):
        path = os.path.join(self.log_dir, self.sessname)
        ensure_dir(path)
        self.writers[name] = SummaryWriter(os.path.join(path, name + '.events'))
        return self.writers[name]

    def write(self, name, loss, ssim, psnr, epoch):
        lr = self.opt.param_groups[0]['lr']
        self.writers[name].add_scalar("lr", lr, epoch)
        self.writers[name].add_scalars(
            "train_loss", {"loss1": loss[0][0], "loss2": loss[0][1], "loss3": loss[0][2]},

            epoch
        )
        self.writers[name].add_scalars(
            "test_loss", {"loss1": loss[1][0], "loss2": loss[1][1], "loss3": loss[1][2]},
            epoch
        )
        self.writers[name].add_scalars("ssim", {"train": ssim[0], "test": ssim[1]}, epoch)
        self.writers[name].add_scalars("psnr", {"train": psnr[0], "test": psnr[1]}, epoch)


    def write_close(self, name):
        self.writers[name].close()

    def get_dataloader(self, dir, name):
        if name == "train":
            dataset = TrainDataset(dir, self.image_size)
            a = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                           num_workers=self.num_workers, drop_last=True)
            self.total_step = len(a)
            return a
        elif name == "val":
            dataset = TestDataset(dir)
            return DataLoader(dataset, batch_size=1, shuffle=False,
                              num_workers=self.num_workers, drop_last=True)
        else:
            print("Incorrect Name for Dataloader!!!")
            return 0

    def save_checkpoints(self, name):
        dir = os.path.join(self.model_dir, self.sessname)
        ensure_dir(dir)
        ckp_path = os.path.join(dir, name)
        obj = {
            'net': self.net.state_dict(),
            'now_epoch': self.now_epoch + 1,
            'opt': self.opt.state_dict(),
        }
        torch.save(obj, ckp_path)

    def load_checkpoints(self, dir):
        ckp_path = dir
        try:
            obj = torch.load(ckp_path)
            logger.info('Load checkpoint %s' % ckp_path)
        except FileNotFoundError:
            logger.info('No checkpoint %s!!' % ckp_path)
            return
        self.net.load_state_dict(obj['net'])
        self.opt.load_state_dict(obj['opt'])
        self.start_epoch = obj['now_epoch']

    def inf_batch(self, name, batch):

        bilater, RES = batch['bilater'].type(torch.FloatTensor).cuda(), batch['RES'].type(torch.FloatTensor).cuda()
        N, GT = batch['NOISE'].type(torch.FloatTensor).cuda(), batch['GT'].type(torch.FloatTensor).cuda()
        if name == "train":
            # background = self.net(N, bilater, RES)
            residual, background, gt = self.net(N, bilater, RES)
            loss1 = self.crit(residual, N - GT)
            loss2 = self.crit(background, GT)
            loss3 = self.crit(gt, GT)
            # loss = loss2
            # loss = loss1 + loss2 + loss3
            loss = 0.1*loss1 + 0.1*loss2 + loss3

            ssim = self.ssim(gt, GT)
            # ssim = self.ssim(background, GT)
            # mse = MSELoss().cuda()(background, GT)
            mse = MSELoss().cuda()(gt, GT)
            psnr = 10 * log10(1 / mse.item())
            self.net.zero_grad()
            loss.backward()
            self.opt.step()
            lr_now = self.opt.param_groups[0]["lr"]
            logger.info("epoch %d/%d, step %d/%d, loss1 %f loss2 %f loss3 %f psnr: %f, ssim : %f, lr is %f"
                        % (self.now_epoch, self.epoch, self.step, self.total_step, loss1, loss2, loss3, psnr, ssim,lr_now))

            # logger.info("epoch %d/%d, step %d/%d, loss2 %f,  psnr: %f, ssim : %f, lr is %f"
            #             % (self.now_epoch, self.epoch, self.step, self.total_step,  loss2, psnr, ssim, lr_now))
            self.step += 1
            return gt, [loss1.item(), loss2.item(), loss3.item()], psnr, ssim.item()
            # return background, loss2.item(), psnr, ssim.item()

        else:
            residual, background, gt = self.net(N, bilater, RES)
            loss1 = self.crit(residual, N - GT)
            loss2 = self.crit(background, GT)
            loss3 = self.crit(gt, GT)

            ssim = self.ssim(gt, GT)
            mse = MSELoss().cuda()(gt, GT)
            psnr = 10 * log10(1 / mse.item())

            return gt, [loss1.item(), loss2.item(), loss3.item()], psnr, ssim.item()

    def change_lr(self, epoch):
        if epoch < 30:
            self.opt.param_groups[0]["lr"] = 0.01
        elif epoch < 60:
            self.opt.param_groups[0]["lr"] = 0.005
        elif epoch < 90:
            self.opt.param_groups[0]["lr"] = 0.0025
        elif epoch < 120:
            self.opt.param_groups[0]["lr"] = 0.00125
        elif epoch < 150:
            self.opt.param_groups[0]["lr"] = 0.001
        elif epoch < 180:
            self.opt.param_groups[0]["lr"] = 0.0005
        else:
            self.opt.param_groups[0]["lr"] = 0.00001

    def epoch_out(self):
        self.step = 0

    def train_net(self):
        self.net.train()

    def val_net(self):
        self.net.eval()

def run_train_val(args):

    if args.If_sp:
        para = args.specified_para
        number = int(len(para))
    else:
        sigma_c = args.sigma_c
        sigma_s = args.sigma_s
        size = args.size
        para = [sigma_c, sigma_s, size]
        number = int(len(sigma_c) * len(sigma_s) * len(size))

    k_number = 1
    sess = Session(args, number*3)
    # sess.load_checkpoints("../models/fr_64_16_run3/epoch 9_ssim 0.419491")
    sess.tensorboard('parameter')
    psnr_m = 0.0
    sess.now_epoch = sess.start_epoch
    path_h5_train = os.path.join('./h5_file', args.sessname, 'train')
    path_h5_test = os.path.join('./h5_file', args.sessname, 'test')
    ensure_dir(path_h5_train)
    ensure_dir(path_h5_test)
    if args.If_sp: #input_dir, output_dir,  base_number, para, k_number, noise_var
        train_h5_dir = data_g_no_iter(
            input_dir=args.train_dir,
            output_dir=path_h5_train,
            base_number=number,
            para=para,
            k_number=k_number,
            noise_var=args.noise_var)
        test_h5_dir = data_g_no_iter(
            input_dir=args.test_dir,
            output_dir=path_h5_test,
            base_number=number,
            para=para,
            k_number=k_number,
            noise_var=args.noise_var)
    else:
        train_h5_dir = data_g_iter(
            input_dir=args.train_dir,
            output_dir=path_h5_train,
            base_number=number,
            para=para,
            k_number=k_number,
            noise_var=args.noise_var)
        test_h5_dir = data_g_iter(
            input_dir=args.test_dir,
            output_dir=path_h5_test,
            base_number=number,
            para=para,
            k_number=k_number,
            noise_var=args.noise_var
        )
    for epoch in range(int(sess.epoch - sess.start_epoch)):
        epoch = epoch + sess.start_epoch
        sess.change_lr(epoch)
        dt_train = sess.get_dataloader(dir=train_h5_dir, name='train')
        dt_val = sess.get_dataloader(dir=test_h5_dir, name='val')
        sess.train_net()
        loss0_train = []
        loss1_train = []
        loss2_train = []
        ssim_train = []
        psnr_train = []
        for batch in dt_train:
            result_train, loss, psnr, ssim = sess.inf_batch("train", batch)
            loss0_train.append(loss[0])
            loss1_train.append(loss[1])
            loss2_train.append(loss[2])
            ssim_train.append(ssim)
            psnr_train.append(psnr)
        sess.epoch_out()
        loss_train = [
            np.mean(loss0_train),
            np.mean(loss1_train),
            np.mean(loss2_train)
        ]
        logger.info('Train: loss1 %f loss2 %f loss3 %f' % (loss_train[0], loss_train[1], loss_train[2]))
        loss0_test = []
        loss1_test = []
        loss2_test = []
        ssim_test = []
        psnr_test = []
        sess.val_net()
        with torch.no_grad():
            for batch in dt_val:
                result_val, loss,  psnr, ssim = sess.inf_batch("val", batch)
                loss0_test.append(loss[0])
                loss1_test.append(loss[1])
                loss2_test.append(loss[2])
                ssim_test.append(ssim)
                psnr_test.append(psnr)
            loss_test = [
                np.mean(loss0_test),
                np.mean(loss1_test),
                np.mean(loss2_test)]
            logger.info('Test: loss1 %f loss2 %f loss3 %f' % (loss_test[0], loss_test[1], loss_test[2]))
            sess.write(
                name="parameter",
                loss=[loss_train, loss_test],
                ssim=[np.mean(ssim_train), np.mean(ssim_test)],
                psnr=[np.mean(psnr_train), np.mean(psnr_test)],
                epoch=epoch)
        if np.mean(psnr_test) > psnr_m:
            logger.info('psnr increase from %f to %f now' % (psnr_m, np.mean(psnr_test)))
            psnr_m = np.mean(psnr_test)
            sess.save_checkpoints("epoch %d psnr %f " % (epoch, psnr_m))
            logger.info('save model as epoch %d psnr %f' % (epoch, psnr_m))
        else:
            logger.info("psnr now is %f, not increase from %f" % (np.mean(psnr_test), psnr_m))
        sess.now_epoch += 1
        # sess.sche.step(epoch=epoch)
    # sess.write_close("parameter")

if __name__ == '__main__':
    log_level = 'info'
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    args = get_args()
    run_train_val(args=args)