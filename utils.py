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
def FrameCapture(inpath, outfolder, div = 24, startFrame = 0, clearoutfolder = True): 
      
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
        cv2.imwrite((outfolder + "/" + "frame%d.jpg") % count, image) 
  
        count += 1
        frame += 1
    
#rejoin all images in folder according to pattern into video at fps
def rejoin(folder, pattern, outfile, fps):
    cmd = "ffmpeg -r " + str(fps) + " -i " + folder + '/' + pattern + " -vcodec mpeg4 -y -c:v libx264 -pix_fmt yuv420p " + outfile
    
    print(cmd)
    os.system(cmd)
    