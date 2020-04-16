# CSE481N: NLP-Capstone Project

[Project Blog](https://estberg.github.io/CSE481N/)

## Set Up

1. Install [Conda](https://developers.google.com/earth-engine/python_install-conda) if there's no existing one in your machine.

2. Set up the conda environment in environment.yml by following these instructions:

    2.1 Create the environment from the `environment.yml` file
    ```
    conda env create -f environment.yml
    ```
   
   2.2 Activate the environment called `slim`
    ```
    conda activate slim
    ```
   At this point, your Conda should be using `slim` environment. To verify the installation of the environment, run `conda env list` or `conda info --envs`

## Download

Run the following

```
python src/download_videos.py \
-s data/MS-ASL/meta/MSASL_tiny.json \
-o data/MS-ASL/videos
```