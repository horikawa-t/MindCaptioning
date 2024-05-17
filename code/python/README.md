## Text generation analysis
- We implemented the text generation analysis using Python.
- Results of feature decoding are required for generating descriptions from the brain.

### Setup instructions
- Run the following command to set up and activate the conda enviroment (named 'mcap_demo').
```plaintext
>> sh ./setup.sh
>> source activate mcap_demo
```
- For Linux: Tested Ubuntu 20.04.6 and 22.04.1 with NVIDIA GPU.
- For Mac: Tested macOS Monterey version 1.12.1.
- For a list of installed library versions on Linux, see [./version_info.txt](./version_info.txt).
- The installation process will take a few minutes.

### Demo
- You can use our method to reconstruct arbitrary word sequences from semantic features (e.g., "May the Force be with you.").
- For a quick demo, see [./mcap_demo.ipynb](./mcap_demo.ipynb)．

### Analysis
- To reproduce the main results in our manuscript, run the following script after activating the environment ('mcap_demo').
- While this analysis required decoding results generated using our [matlab code](../matlab), you can download whole brain decoding results from figshare without performing the analysis by yourself: <a href="https://doi.org/10.6084/m9.figshare.25808179">figshare</a> (<a href="https://figshare.com/ndownloader/files/46347475">res_featdec_wb.zip</a>)
- 
```plaintext
>> source activate mcap_demo
>> python mcap_analysis.py
```
- Using multiple GPUs is strongly recommended for the entire analysis, as the text generation analysis will take about 30 days to complete a single ROI analysis if using a single GPU.

### Note
- Although the script can run without a GPU, it will take an extremely long time for computation.
