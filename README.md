# AnimeGAN

Autumn 2019 Stanford CS236 project. An implementation of a GAN for video style transfer, with specific improvements for cartoon-like styles.

Before running the dataset processor or the baseline generator, ffmpeg must be installed, as well as the requirements from requirements.txt.

### Folder structure
The following shows basic folder structure, which should be created before running the dataset processor.
Place all target dataset videos grouped by dataset in folders in /data/videos/tgts/.
Place all videos (avi files) from Hollywood2 in /data/videos/src.

```
├── data (not included in this repo)
│   ├── AnimeGAN_data 
│   │   ├── src_processed # subset of src_data, resized and limited to two seconds each
│   │   │   ├── train 
│   │   │   └── test 
│   │   ├── paired_targets # target_datasets but resized and with edge_promoting
│   │   ├── src_data # folders containing scenes split by frames 
│   │   └── target_datasets # one folder per target, each containing folders containing scenes split by frames
│   ├── img_datasets # 
│   │   ├── src_img # 1 fps frame samples from src videos
│   │   └── target_img_datasets # one folder per target, each containing 1 fps samples from target videos
|   └── videos
│       ├── other #various videos
│       ├── src # hollywood2 database videos, either in original avi form or trimmed mp4 form
│       ├── src_scenes # hollywood2 videos split by scene
│       ├── tgts # target database folders each containing their video sources
│       └── tgts_scenes # target database folders each containing their video sources split by scene
├── noodles # training code
├── noodles
├── utils.py
├── noodles
└── noodles # results to be saved here
```

Preprocessing:
1. Splice all videos into scenes using PySceneDetect. 
    1. For videos from Hollywood2, cut out the first 15 seconds to eliminate watermarks.
    1. Remove credits scenes from anime sets by hand.
1. Sample frames from all subscenes at 24 fps.
    1. Only retain the first 24 (or other desired number of frames) from each Hollywood2 subscene to reduce dataset size.
1. Letterbox all frames to a 16:9 ratio and resize to 480:270.
    1. For anime training data frames, create paired frames using the following process:
    1. detect edge pixels using a standard Canny edge detector
    1. dilate the edge regions
    1. apply a Gaussian smoothing in the dilated edge regions.
1. (PyTorch DataLoader) Combine chosen consecutive 24 frames into 4d arrays (channel, depth, height, width). Sample every 24 frames in a subscene, or more frequently to augment data.

    
    
