### feature
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
