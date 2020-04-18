---
layout: post
title:  "Blog Post 4: Project Setup"
date:   2020-04-17 21:00:06 -0700
---

## ASL Recognition and Translation

### Approach
We are using the top 3 labels (“hello”, “nice”, “teacher”) of the MS-ASL dataset, downloaded and parsed. We are running a simple sequential model using Tensorflow that predicts the label for the videos. Note that, there were already 95 videos from just the top 3 labels. For quick naive experimental purposes mostly just with the goal of getting the data through a model, we have 4 Lambda layers for our model, defined to just take the sum of all the pixels in a frame, and then the maximum frame, and then mod with the number of labels to essentially randomly map from each video to a label. 

### Experiment

1. Download videos of ASL signing from [MS-ASL](https://www.microsoft.com/en-us/research/publication/ms-asl-a-large-scale-data-set-and-benchmark-for-understanding-american-sign-language/) using python script ([download_videos.py](https://github.com/estberg/CSE481N/blob/master/src/data_processing/download_videos.py))
2. Extract those videos into frames and map each video into a label class representing a word using python script ([extract_frames.py](https://github.com/estberg/CSE481N/blob/master/src/data_processing/extract_frames.py))
3. We run the core of our approach in another python script ([main.py](https://github.com/estberg/CSE481N/blob/master/src/main.py)):
- We load the MS-ASL video frames.
- Using the shape of the data, we create a model.
- We use the model to predict the labels to test a small sample of the test videos.

### Evaluation Framework

MS-ASL has provided us with train, validation, and test sets. Each video is labeled to a class (listing from 0-999), and each class corresponds to a specific word. For example, class 0 corresponds to “hello”, class 1 corresponds to “nice”, and so on.

The evaluation metric is to calculate the accuracy: what percentage of the labels were correctly predicted by the model. (accuracy of recognition)

### Note on Progressing

We have the potential to work on another project with some researchers who are addressing the slightly more complex continuous translation task. We will update in the next post, but there is a chance we will pivot to work with them on one of the following components of their task:
- Segmentation: recognizing the start and end of a single sign
- Translate the signing motion in the captured video range into intermediate representations
- Translate the intermediate representation to written text (currently just English)

We are hoping this can work out as it could be used to help in their project. 
