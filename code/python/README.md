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
- You can also use our method for reconstucting arbitrary word sequences from semantic features (e.g., "May the Force be with you.").
- See [./python/mcap_demo.ipynb](./python/mcap_demo.ipynb) for a quick demo. 

### Analysis
- To reproduce the main results in our manuscript, run the following script after activating the environment (mcap_demo).
```plaintext
>> source activate mcap_demo
>> python mcap_analysis.py
```