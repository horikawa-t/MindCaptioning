### fMRI data
- fMRI data contain preprocessed fMRI data and spaceDefine files for all subjects.
- ./fMRI directory should have the following files:
```plaintext
data/
├── preprocessed/
│   ├── trainPerception_S1.mat (training data; 2180 samples x nVoxels)
│   ├── testPerception_S1.mat (test perception data; [72 samples x 5 repetitions] x nVoxels)
│   ├── trainImagery_S1.mat (test perception data; [72 samples x 5 repetitions] x nVoxels)
│   ├── trainPerception_S2.mat
│   └── ...
└── spaceDefine/
    ├── spaceDefine_S1.nii
    ├── spaceDefine_S2.mat
    └── ...　       
```
