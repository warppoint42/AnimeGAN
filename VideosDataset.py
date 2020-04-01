from torch.utils.data import Dataset, DataLoader
import os 
import torch
from PIL import Image
import imagesize
from torchvision import transforms

class VideosDataset(Dataset):
    "Dataset Class for Loading Videos"
    def __init__(self, root_dir, batch_sz, num_channels=3, transform=None):
        """
        Args:
            root_dir: root directory containing all videos
            batch_sz: number of videos in a batch
            num_channels: number of channels in each frame
            transform: transform to be applied to all videos
        """
        self.video_folders = os.listdir(root_dir)
        self.root_dir = root_dir
        self.batch_sz = batch_sz
        self.num_channels = num_channels
        first_folder = os.path.join(root_dir, self.video_folders[0])
        first_file = os.path.join(first_folder, os.listdir(first_folder)[0])
        self.width, self.height = imagesize.get(first_file)
        self.transform = transform
    
    def __len__(self):
        return len(self.video_folders)
    
    def getVideo(self, video_folder):
        cur_folder = os.path.join(self.root_dir, video_folder)
        frame_fns = os.listdir(video_folder)
        frames = torch.FloatTensor(self.num_channels, len(frame_fns), self.height, self.width)
        for index in range(len(frame_fns)):
            frame = transforms.ToTensor()(Image.open(os.path.join(cur_folder, frame_fns[index])))
            frames[:, index, :, :] = frame
        return frames

    def __getitem__(self, idx):
        video_folder = os.path.join(self.root_dir, self.video_folders[idx])
        videoFrames = self.getVideo(video_folder)
        if self.transform:
            videoFrames = self.transform(videoFrames)
        return videoFrames