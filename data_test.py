import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
import torch
from torch.nn import MSELoss
from cal_ssim import SSIM
from pathlib import Path
from tqdm import tqdm
import h5py
import os
import numpy as np
from train import get_args, ensure_dir
from Net import parameterNet_mlp, parameterNet_linear
import math
from skimage.util import random_noise
from scipy.ndimage import median_filter

# from data_generate import gaussian_noise

args = get_args()
modelPath = "./models/"+args.sessname+"/" + "epoch 139 psnr 24.590886"
# input_dir = "./clean-noise0.01"
input_dir = "./dataset/test"
k = 0.4


phase = str(k)


def cal_psnr(img1, img2):
   mse = np.mean((img1/1.0 - img2/1.0) ** 2)
   if mse < 1.0e-10:
      return 100
   return 10 * math.log10(1.0**2/mse)


def get_image(image):
    image = image*[255]
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def load_checkpoints(dir):
    ckp_path = dir
    try:
        obj = torch.load(ckp_path)
        print('Load checkpoint %s' % ckp_path)
        return obj
    except FileNotFoundError:
        print('No checkpoint %s!!' % ckp_path)
        return
    # self.net.load_state_dict(obj['net'])

    #     # self.opt.load_state_dict(obj['opt'])
    #     # self.start_epoch = obj['now_epoch']


def run_test():

    ssim = SSIM().cuda
    crit = MSELoss().cuda()
    k_number = 1

    if args.If_sp:
        para = args.specified_para
        base_number = int(len(para))
    else:
        sigma_c = args.sigma_c
        sigma_s = args.sigma_s
        size = args.size
        para = [sigma_c, sigma_s, size]
        base_number = int(len(sigma_c) * len(sigma_s) * len(size))

    if args.Net == "parameterNet_linear":
        net = parameterNet_linear(in_channel=base_number*3, out_channel=3).cuda()
    else:
        net = parameterNet_mlp(in_channel=base_number*3, out_channel=3).cuda()

    obj = load_checkpoints(modelPath)
    net.load_state_dict(obj['net'])

    image_files = list(Path(input_dir).glob("*.*"))
    outout_dir = os.path.join("./result", args.sessname + phase)
    ensure_dir(outout_dir)

    psnr_o_all = []
    psnr_all = []
    loss1_all = []
    loss2_all = []
    loss3_all = []
    f = open(outout_dir + "/psnr.txt", 'a')
    for image_file in image_files:
        image_name = str(image_file).split("\\")[-1]
        image_o = (cv2.imread(str(image_file))/255.0).astype(np.float32)
        # image_n = (gaussian_noise(image_o, mean=0, var=args.noise_var)).astype(np.float32)
        # k = np.random.randint(low=1, high=7)
        # k = k / 10.0
        image_n = (random_noise(image_o, mode='s&p', amount=k)).astype(np.float32)
        h, w, c = image_o.shape
        bilater_out = np.zeros((h, w, c * base_number), dtype=np.float)
        N = np.zeros((h, w, c * base_number), dtype=np.float)
        for i in range(base_number):
            N[:, :, i * 3:i * 3 + 3] = image_n
        if args.If_sp:
            for b_sample in range(len(para)):
                img = np.zeros((h, w, c))
                k1, k2 = para[b_sample]
                for i in range(c):
                    img[:, :, i] = median_filter(image_n[:, :, i], (k1, k2))
                bilater_out[:, :, c * b_sample:c * b_sample + 3] = img
        else:
            b_sample = 0
            for gama_c in para[0]:
                for gama_s in para[1]:
                    for size in para[2]:
                        img = image_n
                        for times in range(k_number):
                            img = cv2.bilateralFilter(img, size, gama_c, gama_s)
                        bilater_out[:, :, c * b_sample:c * b_sample + 3] = img
                        b_sample += 1

        RES = N - bilater_out
        image_o = np.transpose(image_o, (2, 0, 1))
        bilater_out = np.transpose(bilater_out, (2, 0, 1))
        RES = np.transpose(RES, (2, 0, 1))
        image_n = np.transpose(image_n, (2, 0, 1))
        bilater_out = torch.from_numpy(np.expand_dims(bilater_out, axis=0)).type(torch.FloatTensor).cuda()
        image_n = torch.from_numpy(np.expand_dims(image_n, axis=0)).type(torch.FloatTensor).cuda()
        image_o = torch.from_numpy(np.expand_dims(image_o, axis=0)).type(torch.FloatTensor).cuda()
        RES = torch.from_numpy(np.expand_dims(RES, axis=0)).type(torch.FloatTensor).cuda()

        residual, background, result = net(image_n, bilater_out, RES)
        loss1 = crit(residual, image_n - image_o).item()
        loss2 = crit(background, image_o).item()
        loss3 = crit(result, image_o).item()

        image_o = image_o.cpu().detach().numpy()
        image_n = image_n.cpu().detach().numpy()
        result = result.cpu().detach().numpy()
        psnr_o = cal_psnr(image_o, image_n)
        psnr = cal_psnr(image_o, result)

        psnr_o_all.append(psnr_o)
        psnr_all.append(psnr)
        loss1_all.append(loss1)
        loss2_all.append(loss2)
        loss3_all.append(loss3)

        f.write("Test image %s psnr_original: %f, psnr: %f, loss1: %f loss2: %f loss3: %f\n" %
                ( image_name, psnr_o, psnr, loss1, loss2, loss3))

        result = np.transpose(result[0], (1, 2, 0))
        result = get_image(result)

        ####save noise image###
        path_noise = './noise_image_' + str(args.noise_var)
        ensure_dir(path_noise)
        image_n = np.transpose(image_n[0], (1, 2, 0))
        image_n = get_image(image_n)
        cv2.imwrite(path_noise+ "/%s" % image_name, image_n)
        ###############

        cv2.imwrite(outout_dir + "/%s" % image_name, result)

        print("Process %s"%image_name)
    f.write("平均为 %f ,loss1: %f, loss2: %f, loss3: %f" % (
        np.mean(psnr_all), np.mean(loss1_all), np.mean(loss2_all), np.mean(loss3_all)))
    f.close()


if __name__ == '__main__':
    run_test()
