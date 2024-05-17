## Decoding and encoding analyses
- We implemented the decoding and encoding analyses using MATLAB (tested by MATLAB version 9.7, R2019b).
- Preprocessed fMRI data and features should be located in specified directory in MindCaptioning/data/{[fmri](../../data/fmri)/[feature](../../data/feature)}. See [README.md](../../data/README.md) in the data directory.

### Analysis
- Run the following script in MATLAB.
```plaintext
>> mcap_encoding_analysis
>> mcap_decoding_analysis
```
- Each of the encoding and decoding analysis can be performed with multiple CPUs in parallel.
- The encoding results are required for the decoding analysis (for voxel selection).
- Using multiple CPUs is strongly recommended for the entire analysis, as the encoding analysis will take about 1 week and the decoding analysis about 50 weeks to complete if using a single CPU.
- All results will be saved in [MindCaptioning/res/](../../res) directory.
- The resultant decoded features will be used in the text generation analysis implemented by [Python](../python).
- Whole brain decoding results can be downloaded from figshare without performing the analysis by yourself:
 <a href="https://doi.org/10.6084/m9.figshare.25808179">figshare</a>
 (<a href="https://figshare.com/ndownloader/files/46347475">res_featdec_wb.zip</a>)

### 
