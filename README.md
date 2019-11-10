# AnimeGAN
Preprocessing:
1. Splice all videos into (x) second long subvideos. 
1.5. Remove all subvideos that involve a scene change.
2. Sample frames from all subvideos at 24 fps.
2.5. For anime training data frames, create paired frames using the following process:
2.5.1.  detect edge pixels using a standard Canny edge detector
2.5.2.  dilate the edge regions
2.5.3. apply a Gaussian smoothing in the dilated edge regions
    
    
