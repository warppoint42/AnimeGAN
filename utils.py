import csv
import subprocess
import math
import json
import os
import shlex
import cv2 
import shutil
from tqdm import tqdm
import numpy as np
import itertools, imageio, torch, random
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import datasets
from scipy.misc import imresize
from torch.autograd import Variable
import VideosDataset

#load frame video datasets
def data_load(path, subfolder, transform, batch_size, shuffle=False, drop_last=True):
    dset = VideosDataset(path, batch_size, 3, None)
    return torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

#from pytorch-CartoonGAN, print network structure
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

#TODO - add 3D    
def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

#get FPS of video
def getFps(vidname):    
    video = cv2.VideoCapture(vidname);

    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver)  < 3 :
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else :
        fps = video.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    video.release(); 
    return fps

#create the directory if it doesn't exist, empty it if it does
def cleardir(dirpath):
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
    try:
        os.mkdir(dirpath)
    except OSError:
        print ("Creation of the directory %s failed" % dirpath)
        print(OSError)
    else:
        print ("Successfully created the directory %s " % dirpath)
        
        
# https://github.com/c0decracker/video-splitter/blob/master/ffmpeg-split.py

def get_video_length(filename):

    output = subprocess.check_output(("ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", filename)).strip()
    video_length = int(float(output))
    print ("Video length in seconds: "+str(video_length))

    return video_length

def ceildiv(a, b):
    return int(math.ceil(a / float(b)))

def split_by_seconds(filename, split_length, vcodec="copy", acodec="copy",
                     extra="", video_length=None, **kwargs):
    if split_length and split_length <= 0:
        print("Split length can't be 0")
        raise SystemExit

    if not video_length:
        video_length = get_video_length(filename)
    split_count = ceildiv(video_length, split_length)
    if(split_count == 1):
        print("Video length is less then the target split length.")
        raise SystemExit

#     split_cmd = ["ffmpeg", "-i", filename, "-vcodec", vcodec, "-acodec", acodec] + shlex.split(extra)
    split_cmd = ["ffmpeg"] + shlex.split(extra)
    try:
        filebase = ".".join(filename.split(".")[:-1])
        shortname = filebase.split('/')[-1]
        fileext = filename.split(".")[-1]
    except IndexError as e:
        raise IndexError("No . in filename. Error: " + str(e))
    
    path = "vid_short/" + shortname
    cleardir(path)
        
    for n in range(0, split_count):
        split_args = []
        if n == 0:
            split_start = 0
        else:
            split_start = split_length * n

#         split_args += ["-ss", str(split_start), "-t", str(split_length),
#                        filebase + "-" + str(n+1) + "-of-" + \
#                         str(split_count) + "." + fileext]
        split_args += ["-ss", str(split_start), "-t", str(split_length), "-i", filename, 
                       "-vcodec", vcodec, "-acodec", acodec,
                       "vid_short/" + shortname + "/" + shortname + str(n+1) + "." + fileext]
        print ("About to run: "+" ".join(split_cmd+split_args))
        subprocess.check_output(split_cmd+split_args)
        
# https://www.geeksforgeeks.org/python-program-extract-frames-using-opencv/
# Program To Read video 
# and Extract Frames 
  
# Function to extract frames 
def FrameCapture(inpath, outfolder, prefix = "", div = 24, startFrame = 0, clearoutfolder = True): 
      
    if clearoutfolder:
        cleardir(outfolder)
    else: 
        try: #make dest folder if doesn't exist
            os.mkdir(outfolder)
        except OSError:
            pass
    
    # Path to video file 
    vidObj = cv2.VideoCapture(inpath) 
  
    # Used as counter variable 
    count = 0
    
    frame = startFrame
  
    # checks whether frames were extracted 
    success = 1
  
    while success: 
        vidObj.set(cv2.CAP_PROP_POS_MSEC,(frame*1000/div))
        # vidObj object calls read 
        # function extract frames 
        success, image = vidObj.read() 
        if not success:
            break
        # Saves the frames with frame-count 
        cv2.imwrite((outfolder + "/" + prefix + "frame%d.jpg") % count, image) 
  
        count += 1
        frame += 1
    
#rejoin all images in folder according to pattern into video at fps
def rejoin(folder, pattern, outfile, fps):
    cmd = "ffmpeg -r " + str(fps) + " -i " + folder + '/' + pattern + " -vcodec mpeg4 -y -c:v libx264 -pix_fmt yuv420p " + outfile
    
    print(cmd)
    os.system(cmd)
    
#https://github.com/znxlwm/pytorch-CartoonGAN/blob/master/edge_promoting.py    
def edge_promoting_no_resize_inplace(root, numframes):
#     file_list = os.listdir(root)
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gauss = cv2.getGaussianKernel(kernel_size, 0)
    gauss = gauss * gauss.transpose(1, 0)
    n = 1
    for n in range(numframes):
        f = "frame" + str(n) + ".jpg"
#         if not f.lower().endswith(('.jpg')):
#             continue
        rgb_img = cv2.imread(os.path.join(root, f))
        gray_img = cv2.imread(os.path.join(root, f), 0)
#         rgb_img = cv2.resize(rgb_img, (256, 256))
        pad_img = np.pad(rgb_img, ((2,2), (2,2), (0,0)), mode='reflect')
#         gray_img = cv2.resize(gray_img, (256, 256))
        edges = cv2.Canny(gray_img, 100, 200)
        dilation = cv2.dilate(edges, kernel)

        gauss_img = np.copy(rgb_img)
        idx = np.where(dilation != 0)
        for i in range(np.sum(dilation != 0)):
            gauss_img[idx[0][i], idx[1][i], 0] = np.sum(np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 0], gauss))
            gauss_img[idx[0][i], idx[1][i], 1] = np.sum(np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 1], gauss))
            gauss_img[idx[0][i], idx[1][i], 2] = np.sum(np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 2], gauss))

        result = np.concatenate((rgb_img, gauss_img), 1)

        cv2.imwrite(os.path.join(root, str(n) + '.png'), result)
        os.system("rm " + os.path.join(root, f))
        n += 1

#from pytorch-CartoonGAN
def edge_promoting(root, save):
    file_list = os.listdir(root)
    if not os.path.isdir(save):
        os.makedirs(save)
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gauss = cv2.getGaussianKernel(kernel_size, 0)
    gauss = gauss * gauss.transpose(1, 0)
    n = 1
    for f in tqdm(file_list):
        rgb_img = cv2.imread(os.path.join(root, f))
        gray_img = cv2.imread(os.path.join(root, f), 0)
        rgb_img = cv2.resize(rgb_img, (256, 256))
        pad_img = np.pad(rgb_img, ((2,2), (2,2), (0,0)), mode='reflect')
        gray_img = cv2.resize(gray_img, (256, 256))
        edges = cv2.Canny(gray_img, 100, 200)
        dilation = cv2.dilate(edges, kernel)

        gauss_img = np.copy(rgb_img)
        idx = np.where(dilation != 0)
        for i in range(np.sum(dilation != 0)):
            gauss_img[idx[0][i], idx[1][i], 0] = np.sum(np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 0], gauss))
            gauss_img[idx[0][i], idx[1][i], 1] = np.sum(np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 1], gauss))
            gauss_img[idx[0][i], idx[1][i], 2] = np.sum(np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 2], gauss))

        result = np.concatenate((rgb_img, gauss_img), 1)

        cv2.imwrite(os.path.join(save, str(n) + '.png'), result)
        n += 1
    
