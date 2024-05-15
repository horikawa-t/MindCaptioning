## Decoding and encoding analyses (MATLAB version 9.7, R2019b)
- We implemented the decoding and encoding analyses using MATLAB.
- Preprocessed fMRI data and features should be located in specified directory MindCaptioning/data/{fmri/feature}. See [README.md](../../data/README.md) in the data directory.

### Analysis
- Run the following script in MATLAB.
```plaintext
>> mcap_encoding_analysis
>> mcap_decoding_analysis
```
- Each of the encoding and decoding analysis can be performed with multiple cpus in parallel.
- The encoding results are required for the decoding analysis (for voxel selection).
- The use of multiple cpus is strongly recommended to run the whole analysis, as the encoding and decoding analysis will take about 1 and 20 weeks, respectivly, using a single cpu to complete all the computations.
 
- All the results will be saved in MindCaptioning/res/ directory.
- The resultant decoded features will be used in the text generation analysis implemented by Python.
- The decoding results can be downloaded from figshare without performing the analysis by yourself:
 <a href="https://doi.org/10.6084/m9.figshare.25808179">figshare</a>
 (<a href="https://figshare.com/ndownloader/files/46338292">res_featdec.zip</a>)
## 
