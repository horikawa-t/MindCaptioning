## Results
- Results will be saved in this directory.
- The decoding results can be downloaded from figshare without performing the analysis by yourself: figshare (res_featdec.zip)
- The decoding results will be used in the text generation analysis implemented by Python.
- ./res directory should have the following files:
```plaintext
res/
├── decoding/
│   ├── trainPerception/
│   │   └── deberta-large/
│   │   │   ├── S1/
│   │   │   │   ├── layer01.mat
│   │   │   │   └── ...
│   │   │   └── S2/
│   │   │   └── ...
│   │   └── ...
│   ├── testPerception/
│   │   └── deberta-large/
│   │   │   ├── S1/
│   │   │   │   ├── layer01.mat
│   │   │   │   └── ...
│   │   │   └── S2/
│   │   │   └── ...
│   │   └── ...
│   └── testImagery/
│       └── deberta-large/
│       │   ├── S1/
│       │   │   ├── layer01.mat
│       │   │   └── ...
│       │   └── S2/
│       │   └── ...
│       └── ...
└── encoding/
    ├──(same with the decoding directory)
```
