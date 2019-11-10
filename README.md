# AnimeGAN
Preprocessing:
1. Splice all videos into (x) second long subvideos. 
    1. Remove all subvideos that involve a scene change.
1. Sample frames from all subvideos at 24 fps.
    1. For anime training data frames, create paired frames using the following process:
    1. detect edge pixels using a standard Canny edge detector
    1.  dilate the edge regions
    1. apply a Gaussian smoothing in the dilated edge regions
1. Somehow crop-resample all frames to a 16:9 ratio with standardized resolution
    
    
