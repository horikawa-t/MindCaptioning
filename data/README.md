## Preprocessed fMRI data and features
- Preprcessed fMRI data and features should be placed in this directory.
- They are available from figshare (<a href="https://figshare.com/ndownloader/files/46336531">fmri.zip</a>, <a href="https://figshare.com/ndownloader/files/46336429">feature.zip</a>).

### fMRI data
- fMRI data contain preprocessed fMRI data and spaceDefine files for all subjects.
- ./fMRI directory should have the following files:
```plaintext
data/
├── preprocessed/
│   ├── trainPerception_S1.mat (training data; braindat:2180 samples x nVoxels)
│   ├── testPerception_S1.mat (test perception data; braindat:[72 samples x 5 repetitions] x nVoxels)
│   ├── trainImagery_S1.mat (test perception data; braindat:[72 samples x 5 repetitions] x nVoxels)
│   ├── trainPerception_S2.mat
│   └── ...
└── spaceDefine/
    ├── spaceDefine_S1.nii
    ├── spaceDefine_S2.nii
    └── ...　       
```
### features
- Feature data contain semantic features of deberta-large (for each video and caption) and visual features of timesformer (for each video).
- This directory also contains normalization parameters (mu, std) for all layers.
```plaintext
feature/
├── deberta-large/
│   ├── caption/
│   │   └── (layer01.mat/.../layer24.mat) (caption-wise features; feat:[nVideos x 20 captions] x nUnits)
│   ├── video/
│   │   └── (layer01.mat/.../layer24.mat) (video-wise features; feat:nVideos x nUnits)
│   └── norm_param/
│       └── (layer01.mat/.../layer24.mat)  (mu, sd; 1 x nUnits)
└── timesformer/
    ├── video/
    │   └── (layer01.mat/.../layer12.mat) (video-wise features; feat:nVideos x nUnits)
    └── norm_param/
        └── (layer01.mat/.../layer12.mat)  (mu, sd; 1 x nUnits)
```
### Note
-
