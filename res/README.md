## Results
- Results will be saved in this directory.
- The decoding results can be downloaded from figshare without performing the analysis by yourself: figshare (res_featdec.zip)
- The decoding results will be used in the text generation analysis implemented by Python.
- ./res directory should have the following files:
```plaintext
res/
├── decoding/
│   └── (trainPerception/testPerception/testImagery)/
│       └── deberta-large/
│           └──(S1/.../S6)/
│               └──(WB/WBnoVis/WBnoSem/WBnoLang/Lang)/ 
│                   └──(layer01/.../layer24).mat
├── encoding/
│   └── (trainPerception/testPerception/testImagery)/
│       └── (deberta-large/timesformer)/
│           └──(S1/.../S6)/
│              └──(layer01/.../layer24).mat
└── text_generation/
    └── (trainPerception/testPerception/testImagery)/
        └── (mlm_roberta-large)/
            └── (lm_deberta-large)/
                └──(S1/.../S6)/
                    └──(WB/WBnoVis/WBnoSem/WBnoLang/Lang)/ 
                       └──(res_samp0001/...).mat
```
