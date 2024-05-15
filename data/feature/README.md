### features
- Feature data contain semantic features of deberta-large (for each video and caption) and visual features of timesformer (for each video).
- This directory also contains normalization parameters (mu, std) for all layers.
```plaintext
feature/
├── deberta-large/
│   ├── caption/
│   │   ├── layer01.mat (caption-wise features: [nVideos x 20 captions] x nUnits)
│   │   └── ...
│   ├── video/
│   │   ├── layer01.mat (video-wise features: nVideos x nUnits)
│   │   └── ...
│   └── norm_param/
│       ├── layer01.mat (mu, sd; 1 x nUnits)
│       └── ...
└── timesformer/
    ├── video/
    │   ├── layer01.mat (video-wise features: nVideo x nUnit)
    │   └── ...
    └── norm_param/
        ├── layer01.mat (mu, sd; 1 x nUnit)
        └── ...
```
