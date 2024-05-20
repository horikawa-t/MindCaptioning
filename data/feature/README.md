### feature
- Feature data contain semantic features of deberta-large and visual features of timesformer computed for all videos.
- This directory also contains normalization parameters (mu, std) for all layers of 43 models (42 language models and timesformer).
```plaintext
feature/
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
