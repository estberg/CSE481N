---
layout: post
title:  "Blog Post 5: Second Baseline Approach"
date:   2020-04-23 22:49:43 -0700
---

## ASL Recognition and Translation

### Updates on Naive Approach
We have expanded our dataset to look at the top 100 labels in the dataset, and have made a minor change in our existing model to reflect the number of classes (% 100, instead of % 1000). This took some setup to download the 20~ GB of video from YouTube (without getting 429 banned) and make sure it was set up properly on `nlpg02`. 

After having run 5 epochs on the top-100, we have achieved a training accuracy of 0.012474589. 

This is slightly better, compared to the accuracy of 0.008686 that we have achieved previously with 5 epochs on the top-3. This could be because we have access to a lot more data to feed into our model, and the fix helped our model. However, it is essentially equivalent to the random guessing in the 100 classes, which makes sense as this is close to the extremely naive approach of modulus with a sum of the pixel values. We just were using this to check our data set up.

### I3D Set Up Progress
The authors of the two papers ([MS-ASL](http://export.arxiv.org/pdf/1812.01053#page=7), [WLASL](https://www.groundai.com/project/word-level-deep-sign-language-recognition-from-video-a-new-large-scale-dataset-and-methods-comparison/1)) we are interested in both cited this paper on [I3D](https://arxiv.org/pdf/1705.07750.pdf). It turns out that we don’t have to worry about the temporal issue of the videos being different lengths if we are recreating I3D in this manner. They trained the model on a hyperparameter of chosen number of frames and for each sample chose a random start place, so we modified our data loader to do this as well. However, it might be interesting to explore if there is a way to use the entire clip of each sign in training as the full sign would be useful to include. This could be done with some padding and mask, but would not work with a 3D convolutional neural network training in batches with different sample lengths (it would be analogous to training a 2D convolution neural network on images of different sizes in the batches). It may be a short falling of the 3D convolutional neural network approach that could be addressed with other approaches. 

### Collaborating for project YOLO
YOLO, i.e. “You Only Look Once”, is an algorithm that will “only look once” to recognize signing that is being worked on by a data scientist at Microsoft. It segments a continuous signing into individual signs, translates the signing motions into intermediate representations into English. We hope that in this collaboration for YOLO, that we label the videos with intermediate forms, or translate intermediate form into text.

YOLO was developed by Joseph Redmon, a UW graduate student, to detect an object in the video in real-time. Here are the [paper](https://pjreddie.com/media/files/papers/yolo.pdf) and the [project page](https://pjreddie.com/darknet/yolo/).

#### Updates on Collaboration
We have been provided links to the [codebase](https://github.com/pjreddie/darknet), and a [Keras implementation of YOLO](https://github.com/qqwweee/keras-yolo3). We are also tentatively planning to meet with the people working on this project next Monday.


### Next Steps
We made good progress this week in terms of figuring out how to get a GPU equipped system (`nlpg02`) set up for training our model, and further examining I3D. We also continued our contact with the potential team to collaborate with and scheduled a tentative meeting. 

We need to get a working version of I3D for our model, but there are complications in setting this up and we will do our best to have that in the next week. This will give us some weeks to experiment, and then hopefully move on to translation. Ultimately, we’d hope to work on the collaboration project instead though, and have started to prepare for this transition as well. 
