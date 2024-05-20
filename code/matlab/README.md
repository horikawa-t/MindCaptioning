## Decoding and encoding analyses
- We implemented the decoding and encoding analyses using MATLAB (tested by MATLAB version 9.7, R2019b).
- Preprocessed fMRI data and features should be located in specified directory in MindCaptioning/data/{fmri/feature}.
- See [README.md](../../data/README.md) in the data directory for the detailes of data structures.

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
 (<a href="https://figshare.com/ndownloader/files/46420294">decfeat_wb.zip</a>). After downloading, unzip the file and place it in the [MindCaptioning/res/decoding](../../res/decoding) directory.

### Evaluation
- Run the following script in MATLAB after completing the encoding analysis.
```plaintext
>> mcap_summary_encoding
```
- This script generates result figures in Fig. 3.
- You can download subsets of the encoding results from figshare without performing the analysis by yourself:
 <a href="https://doi.org/10.6084/m9.figshare.25808179">figshare</a>
 (<a href="https://figshare.com/ndownloader/files/46420405">res_encoding.zip</a>). After downloading, unzip the file and place it in the [MindCaptioning/res/encoding](../../res/encoding) directory.

### Summary of text generation results
- Run the following script in MATLAB after completing the text generation analysis.
```plaintext
>> mcap_summary_decoding
```
- This script generates result figures in Fig. 2 and 4.
- You can download subsets of the text generation results from <a href="https://doi.org/10.6084/m9.figshare.25808179">figshare</a> (<a href="https://figshare.com/ndownloader/files/46422523">res_textgen.zip</a>) without having to perform the analysis yourself. After downloading, unzip the file and place it in the [MindCaptioning/res/text_generation](../../res/text_generation) directory.
