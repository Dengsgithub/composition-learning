from pathlib import Path
from tqdm import tqdm
import h5py
import os
import numpy as np
import cv2



def save_name(para, k_number):
    out ="k"+str(k_number)+"_"
    for i in range(len(para)):
        for j in range(len(para[i])):
            out = out+str(para[i][j])+"_"
    return out+".h5"


def gaussian_noise(img, mean, var):       #####要求是0到1之间的图片
    noise_img = img.astype(np.float)
    np.random.seed(10)
    noise = np.random.normal(mean, var ** 0.5, img.shape)
    noise_img += noise
    noise_img = np.clip(noise_img, 0, 1).astype(np.float)
    return noise_img


def data_g_iter(input_dir, output_dir, base_number, para, k_number, noise_var):
    paths = list(Path(input_dir).glob("*.*"))
    out_name = save_name(para, k_number)
    out_file = os.path.join(output_dir, out_name)
    if not os.path.exists(out_file):
        f = h5py.File(out_file, 'w')
        n = f.create_group("noise")
        x = f.create_group("input")
        y = f.create_group("label")
        for i in tqdm(range(len(paths))):
            image_file = str(paths[i])
            image_name = image_file.split("\\")[-1].split(".")[0]
            image_o = (cv2.imread(image_file)/255.0).astype(np.float32)
            image_n = (gaussian_noise(image_o, mean=0, var=noise_var)).astype(np.float32)
            h, w, c = image_o.shape
            n.create_dataset(image_name, data=image_n)
            bilater_out = np.zeros((h, w, c*base_number), dtype=np.float)
            b_sample = 0
            for gama_c in para[0]:
                for gama_s in para[1]:
                    for size in para[2]:
                        img = image_n
                        for times in range(k_number):
                            img = cv2.bilateralFilter(img, size, gama_c, gama_s)
                        bilater_out[:, :, c*b_sample:c*b_sample+3] = img
                        b_sample += 1
            x.create_dataset(image_name, data=bilater_out)
            y.create_dataset(image_name, data=image_o)
        f.close()
        print("Images process successful !!!")
        return out_file
    else:
        print("Images had been processed !!!")
        return out_file


def data_g_no_iter(input_dir, output_dir,  base_number, para, k_number, noise_var):
    paths = list(Path(input_dir).glob("*.*"))
    out_name = save_name(para, k_number)
    out_file = os.path.join(output_dir, out_name)
    if not os.path.exists(out_file):
        f = h5py.File(out_file, 'w')
        n = f.create_group("noise")
        x = f.create_group("input")
        y = f.create_group("label")
        for i in tqdm(range(len(paths))):
            image_file = str(paths[i])
            image_name = image_file.split("\\")[-1].split(".")[0]
            image_o = (cv2.imread(image_file)/255.0).astype(np.float32)
            image_n = (gaussian_noise(image_o, mean=0, var=noise_var)).astype(np.float32)
            h, w, c = image_o.shape
            n.create_dataset(image_name, data=image_n)
            bilater_out = np.zeros((h, w, c*base_number), dtype=np.float)
            for b_sample in range(len(para)):
                img = image_n
                gama_c, gama_s, size = para[b_sample]
                for time in range(k_number):
                    img = cv2.bilateralFilter(img, size, gama_c, gama_s)
                bilater_out[:, :, c * b_sample:c * b_sample + 3] = img
            x.create_dataset(image_name, data=bilater_out)
            y.create_dataset(image_name, data=image_o)
        f.close()
        print("Images process successful !!!")
        return out_file
    else:
        print("Images had been processed !!!")
        return out_file