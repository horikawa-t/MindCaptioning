# Code for reproducing Mind Captioning results
- Decoding and encoding analyses were implemented by matlab.
- Text generation analysis was implemented by python.

## Decoding and encoding analyses (matlab, R2019b was used)
- We implemented the decoding and encoding analyses using MATLAB.
- Preprocessed fMRI data and features should be located in specified directory MindCaptioning/data/{fmri/feature}. See [README.md](../data/README.md) in data directory.

### Analysis
- Run the following script in MATLAB.
```plaintext
>> mcap_encoding_analysis
>> mcap_decoding_analysis
```
- Each of the encoding and decoding analysis can be performed with multiple cpu in parallel.
- But, the encoding results are required for the decoding analysis (for voxel selection).
- We strongly recomend to use multiple cpus to run the whole analysis, as the encoding and decoding analysis will take about 1 or 20 weeks, respectivly, using a single cpu to complete all the computations.
 
- The all results will be saved in MindCaptioning/res/ directory.
- The resultant decoded features will be used in the text generation analysis implemented by Python.
- The decoding results can be downloaded from figshare without performing the analysis by yourself:
 <a href="https://doi.org/10.6084/m9.figshare.25808179">figshare</a>
 (<a href="https://figshare.com/ndownloader/files/46338292">res_featdec.zip</a>)

## Text generation analysis
- We implemented the text generation analysis using Python.
- Results of feature decoding are required for generating descriptions from the brain.
- You can also test our method for arbitrary word sequences (e.g., "Five apples are on the table."). See mcap_demo.ipynb. 

### Setup
- Run the following command to setup the conda enviroment (named mcap).
```plaintext
For Linux (we tested Ubuntu 20.04.6 or 22.04.1 with nvidia GPU)
>> conda env create -f environment.yml
>> source activate mcap
>> pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117

For Mac (we tested macOS Monterey version 1.12.1)
>> CONDA_SUBDIR=osx-64 conda env create -f environment.yml
>> source activate mcap
>> pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0
```


### Analysis
- Run the following script in the environment.
```plaintext
>> python mcap_analysis.py
```
