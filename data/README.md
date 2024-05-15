## Preprocessed fMRI data and features
- Preprcessed fMRI data and features should be placed in this directory.
- They are available from figshare (<a href="https://figshare.com/ndownloader/files/46336531">fmri</a>, <a href="https://figshare.com/ndownloader/files/46336429">feature</a>).

### fMRI data
- fMRI data contain preprocessed fMRI data and spaceDefine files for all subjects.
- ./fMRI directory should have the following files:
```plaintext
data/ --+-- preprocessed/ --+-- trainPerception_S1.mat
        |                   |
        |                 --+-- testPerception_S1.mat
        |                   |
        |                 --+-- trainImagery_S1.mat
        |                   |
        |                 --+-- trainPerception_S2.mat
        |                   |
        |                    
        +-- spaceDefine/ --+-- spaceDefine_S1.nii
        |                  |
        |                --+-- spaceDefine_S2.mat
        |                  |
　       
```
### features
- Feature data contain semantic features of deberta-large (for each video and caption) and visual features of timesformer (for each video).
- This directory also contains normalization parameters (mu, std) for all layers.
```plaintext
feature/ --+-- deberta-large/ --+-- caption/ --+-- layer01.mat (caption-wise features: [nVideo x 20 captions] x nUnit)
        |                       |
        |                     --+-- video/ --+-- layer01.mat (video-wise features: nVideo x nUnit)
        |                       |
        |                     --+-- norm_param/ --+-- layer01.mat (mu, sd; 1 x nUnit)
        |                    
         --+-- timesformer/ --+-- video/ --+-- layer01.mat (video-wise features: nVideo x nUnit)
        |                     |
        |                   --+-- norm_param/ --+-- layer01.mat (mu, sd; 1 x nUnit)
```
### Note
-
