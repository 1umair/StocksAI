# Deep Learning - Sentinel Stocks

# Description

This repository contains scripts for training and tuning the NLP model BERT, Longformer, FinBERT, and ULMFit for the goal of sentiment analysis on stock transcript data.
This is a comparative study of three deep learn-
ing models—BERT, FinBERT, and ULMFiT—in
sentiment analysis of earnings call transcripts. The
goal is to find out how leveraging Natural Lan-
guage Processing (NLP) to extract sentiments con-
veyed in large financial transcripts can help in
better-informed investment decisions and risk man-
agement strategies. We explore the advantages and
disadvantages each of the deep learning model of-
fers in the context of our task. We will also assess
the different requirements of each model in regard
to the data pre-processing as well as the optimiza-
tion of their computational resources. Through rig-
orous experimentation and analysis, we evaluate
and perform a comparative study of the models us-
ing various performance metrics such as accuracy,
precision, recall, and F1-score. Finally, we will
discuss what possible improvements can help make
these models more suitable for sentiment analysis
in financial contexts


# Table of Contents
- [Description](#description)
- [Cloning](#cloning)
- [Usage](#usage)
- [Installation](#installation)
- [Contributers](#contributers)



# Cloning

Follow the following instructions to clone the repository. Make sure you have an SSH key setup with gitlab for SSH access. Reach out to POC if having trouble accessing the repository.

For SSH:
```bash
git clone git@github.com:1umair/StocksAI.git
```

For HTTPS:
```bash
git clone https://github.com/1umair/StocksAI.git
```



# Usage

Activate the enviornment with: 
```bash
conda activate nlp
```

You can deactivate the environment with:
```bash
conda deactivate
```

Run jupyter with:
```bash
jupyter notebook
```

# Installation
These installation steps have been tested on trainBERT and trainLongformer. 
Additional libraries might be required for the other two models.

## For CUDA-capable GPU
For standard GPU:
```bash
conda create -n nlp python=3.9
conda activate nlp
conda install cudatoolkit cudnn
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install conda-forge::transformers
conda install anaconda::seaborn 
conda install anaconda::scikit-learn 
conda install anaconda::jupyter
pip install seaborn --upgrade # must have seaborn be 0.13.0, there is a bug in 0.12.2
```

For Icehammer:
```bash
module load anaconda3
module load jupyterhub
module load cuda/10.1
module load gcc/5
conda create -n nlp -c defaults -c conda-forge pip jupyterhub=2.0 batchspawner jupyterlab
conda install cudatoolkit cudnn
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install conda-forge::transformers
conda install anaconda::seaborn 
conda install anaconda::scikit-learn 
conda install anaconda::jupyter
pip install seaborn --upgrade # must have seaborn be 0.13.0, there is a bug in 0.12.2
```

## For CPU only
Will likely experience memory problems and errors when running on the CPU. 
```bash
conda create -n nlp python=3.6
conda activate nlp
conda install conda-forge::transformers
conda install anaconda::seaborn 
conda install anaconda::scikit-learn 
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install anaconda::jupyter 
```


## Install Transcript Database

You will need this repository to run any of these scripts. This holds the custom datset utilized.

Go to the following link to view: `https://github.gatech.edu/edaykin3/dl-7643-finalproj-DatasetCuration`

To clone:
```bash
git clone https://github.gatech.edu/edaykin3/dl-7643-finalproj-DatasetCuration
```


# Contributors
- Umair Zakir Abowath (uabowath3@gatech.edu)
- Evan Daykin (edaykin3@gatech.edu)
- Amssatou Diagne (adiagne7@gatech.edu)
- Jacob Faile (jfaile3@gatech.edu)
