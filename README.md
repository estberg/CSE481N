# CSE481N: NLP-Capstone Project

[Project Blog](https://estberg.github.io/CSE481N/)

## Set Up

[Set up the conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) in environment.yml

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