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

3. Install all required packages
```
pip install -r requirements.txt
```

## Download and Set Up

To understand `MS-ASL` dataset, read [here](data/MS-ASL/meta/README.md).

Below shows the step on how to download and extract frames on **tiny** dataset:

1. Run the following to download some videos

    ```
    python src/data_processing/download_videos.py \
    -s data/MS-ASL/meta/MSASL_tiny.json \
    -o data/MS-ASL/videos
    ```

    Now, the videos should be downloaded into `data/MS-ASL`

2. Run the following to extract frames from the videos

    ```
    python src/data_processing/extract_frames.py \
      -s data/MS-ASL/meta/MSASL_tiny.json \
      -v data/MS-ASL/videos \
      -o data/MS-ASL/frames
    ```

    At this point, picture frames of each video should be under `data/MS-ASL/frames/global_crops`. Information of each video (such as its class label, start frame, end frame, total number of frames, and fps) will be in under `data/MS-ASL/frames/tiny.txt`