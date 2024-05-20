### fMRI data
- fMRI data contain preprocessed fMRI data and spaceDefine files for all subjects.
- [../fmri](../fmri/) directory should have the following files:
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
