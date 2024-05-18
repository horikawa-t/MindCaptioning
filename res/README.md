## Results
- Results will be saved in this directory.
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
- The decoding results will be used in the text generation analysis implemented by [../code/python/](Python).
- The decoding results can be downloaded from figshare without performing the analysis by yourself: <a href="https://doi.org/10.6084/m9.figshare.25808179">figshare</a> (<a href="https://figshare.com/ndownloader/files/46387381">res_featdec_wb.zip</a>)
