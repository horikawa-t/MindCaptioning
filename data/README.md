## Preprocessed fMRI data, features, and captions
- Preprcessed fMRI data and features should be placed in this directory.
- They are available from figshare (<a href="https://figshare.com/ndownloader/files/46336531">fmri.zip</a>, <a href="https://figshare.com/ndownloader/files/46336429">feature.zip</a>).
- Video captions: [./caption/caption_ck20.csv](./caption/caption_ck20.csv)

### fMRI data
- fMRI data contain preprocessed fMRI data and spaceDefine files for all subjects.
- ./fMRI directory should have the following files:
```plaintext
data/
├── preprocessed/
│   ├── trainPerception_S1.mat (training data; braindat:2180 samples x nVoxels)
│   ├── testPerception_S1.mat (test perception data; braindat:[72 samples x 5 repetitions] x nVoxels)
│   ├── testImagery_S1.mat (test perception data; braindat:[72 samples x 5 repetitions] x nVoxels)
│   ├── trainPerception_S2.mat
│   └── ...
└── spaceDefine/
    ├── spaceDefine_S1.nii
    ├── spaceDefine_S2.nii
    └── ...　       
```
### feature
- Feature data contain semantic features of deberta-large (for each video and caption) and visual features of timesformer (for each video).
- This directory also contains normalization parameters (mu, std) for all layers of 43 models (42 language models and timesformer).
```plaintext
feature/
├── caption/ (caption-wise features; feat:[nVideos x 20 captions] x nUnits)
│   └── deberta-large/
│       └── (layer01.mat/.../layer24.mat) 
├── video/ (video-wise features; feat:nVideos x nUnits)
│   ├── deberta-large/
│   │   └── (layer01.mat/.../layer24.mat) 
│   └── timesformer/
│       └── (layer01.mat/.../layer12.mat)
└── norm_param/ (normalization parameters; mu, sd; 1 x nUnits)
    ├── deberta-large/
    │   └── (layer01.mat/.../layer24.mat) 
    ├── timesformer/
    │   └── (layer01.mat/.../layer12.mat)
    └── ...
```
### Video caption
- The video caption data were collected via a crowdsourcing experiment.
- 20 captions were annotated for each of the original 2196 videos, resulting in a total of 43920 captions.

### misc
- The misc directory contains data used for summarizing results. 
