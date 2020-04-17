# CSE481N: NLP-Capstone Project

[Project Blog](https://estberg.github.io/CSE481N/)

## Set Up

1. Install [Conda](https://developers.google.com/earth-engine/python_install-conda) if there's no existing one on your machine.

2. Set up the conda environment in `environment.yml` by following these instructions:

    2.1. Create the environment from the `environment.yml` file
    ```
    conda env create -f environment.yml
    ```
   
    2.2. Activate the environment called `slim`
    ```
    conda activate slim
    ```
   At this point, your Conda should be using `slim` environment. To verify the installation of the environment, run `conda env list` or `conda info --envs`

## Download and Set Up

Run the following to download some videos

```
python src/data_processing/download_videos.py \
-s data/MS-ASL/meta/MSASL_tiny.json \
-o data/MS-ASL/videos
```

Run the following to extract frames from the videos

```
python src/data_processing/extract_frames.py \
  -s data/MS-ASL/meta/MSASL_tiny.json \
  -v data/MS-ASL/videos \
  -o data/MS-ASL/frames
```