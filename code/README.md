# Code for reproducing Mind Captioning results
- Decoding and encoding analyses were implemented by matlab.
- Text generation analysis was implemented by python.

## Decoding and encoding analyses (matlab, R2019b was used)
- We implemented the decoding and encoding analyses using MATLAB.
- Preprocessed fMRI data and features should be located in specified directory MindCaptioning/data/{fmri/feature}. See [README.md](../data/README.md) in the data directory.

### Analysis
- Run the following script in MATLAB.
```plaintext
>> mcap_encoding_analysis
>> mcap_decoding_analysis
```
- Each of the encoding and decoding analysis can be performed with multiple cpus in parallel.
- The encoding results are required for the decoding analysis (for voxel selection).
- The use of multiple cpus is strongly recommended to run the whole analysis, as the encoding and decoding analysis will take about 1 and 50 weeks, respectivly, using a single cpu to complete all the computations.
 
- All the results will be saved in MindCaptioning/res/ directory.
- The resultant decoded features will be used in the text generation analysis implemented by Python.
- Whole brain decoding results can be downloaded from figshare without performing the analysis by yourself:
 <a href="https://doi.org/10.6084/m9.figshare.25808179">figshare</a>
 (<a href="https://figshare.com/ndownloader/files/46347475">res_featdec_wb.zip</a>)

## Text generation analysis
- We implemented the text generation analysis using Python.
- Results of feature decoding are required for generating descriptions from the brain.

### Setup
- Run the following command to setup and activate the conda enviroment (named mcap_demo).
- For Linux, we tested Ubuntu 20.04.6 and 22.04.1 with nvidia GPU.
- For Mac, we tested macOS Monterey version 1.12.1.
```plaintext
>> sh ./setup.sh
>> source activate mcap_demo
```
### Demo
- You can also use our method for reconstucting arbitrary word sequences from semantic features (e.g., "May the Force be with you."). See mcap_demo.ipynb for demo. 

### Analysis
- To reproduce the main results in our manuscript, run the following script after activating the environment (mcap_demo).
```plaintext
>> source activate mcap_demo
>> python mcap_analysis.py
```
