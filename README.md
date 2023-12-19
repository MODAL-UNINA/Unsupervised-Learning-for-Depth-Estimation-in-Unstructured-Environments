# Unsupervised Learning for Depth Estimation in Unstructured Environments
This is a Pytorch implementation of the following paper:

Pian Qi, Fabio Giampaolo, Edoardo Prezioso, Francesco Piccialli, Unsupervised Learning for Depth Estimation in Unstructured Environments. [Paper](link)

# Requirements
python>=3.6

pytorch>=0.4

# Note
Follow the official website of the Mid-Air dataset [here](https://midair.ulg.ac.be/) to download and place the files into the folder named "Dataset".

Example data folder structure
'''
├── Dataset
│   ├── train
│   │   ├── trajectory_0000
│   │   │   ├─ image_left
│   │   │   │   ├── 00001.png
│   │   │   │   └── ...
│   │   │   ├─ image_right
│   │   │   │   ├── 00001.png
│   │   │   │   └── ...
│   │   ├── trajectory_0001
│   │   │   ├─ ...
│   ├── test
│   │   ├── trajectory
│   │   │   ├─ image_left
│   │   │   │   ├── 00001.png
│   │   │   │   └── ...
│   │   │   ├─ image_right
│   │   │   │   ├── 00001.png
│   │   │   │   └── ...
'''
