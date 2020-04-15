---
layout: post
title:  "Blog Post 3: Project Proposal"
date:   2020-04-14 21:59:06 -0700
---

## ASL Recognition and Translation

### Minimum Viable Action Plan

The first step of our minimum viable action plan is to address the task of recognition on an available dataset. There are two easily accessible datasets that are already set up for the tasks of ASL recognition. The first is the Microsoft American Sign Language Dataset (MS-ASL), which has four sets of varying size, ranging from 100 to 2000 classes (words) and 189 to 222 signers, the largest of which contains 25,000 samples of signing. The second is the Word Level American Sign Language dataset which contains a similar number of samples and classes, with slightly fewer signers. At a minimum, we would use one of these existing datasets and try a series of different models to do word recognition (classification) from videos of signing. 

To start, we have chosen to work with the MS-ASL dataset which has over 24 hours of video, and at least 11 samples (one signer’s video) for each class. We will work to recreate some of the models tested in this paper. The authors take a variety of approaches but we will try to recreate their top performing approach using I3D, using a 3D CNN. At a minimum, we will get one this model performing and can look into ways to improve its performance. 

### Stretch Goals

The first stretch goal would be to try a completely different approach. This could involve (1) body key points or (2) 2D CNNs and LSTM or GRU. These were the other approaches that researchers found to be somewhat successful for this task. 

One goal would be to incorporate larger datasets than those which were previously used in many recognition projects. We are exploring the option of collecting or combining data from various sources and have contacted researchers at Microsoft and Gallaudet University. We could also consider trying to combine some of the similar datasets that are available online. Another option that could be explored as a related stretch goal would be adding generated videos to the training dataset. There is some work exploring this for sign language that has demonstrated some performance improvements. 

Finally, another project consideration we are excited about is translation where videos are continuous sequences of signs communicating entire sentences, rather than just individual words. This is another, related task. There are good amounts of work on recognition and translation, but seems to be limited work relating these two closely related tasks, and translation is the lesser explored of the two tasks. It would be interesting to explore how training a model for word recognition and continuous translation compares and how these can be used together. Further, most of the work on continuous translation of sign languages uses European sign language data sets. If we can find and do similar work with ASL, it would be very exciting!

### Motivation and Context

American Sign Language (ASL)  incorporates a variety of articulators which makes it a highly challenging recognition task. As signers communicate through their handshape, orientation, movement, upper body, and the face there are many components of the language that need to be captured by recognition models. Further, traditional language models operate on the assumption that language occurs chronologically, but there are cases in which signs are related asynchronously on multiple streams.

There is a significant population of around 500,000 people communicating with ASL, so recognition, translation, interpretation, and generation are all relevant tasks to serve this community. Computer aided translation, interpretation, and generation are all dependent on the performance of recognition. 

### Related Works
 
#### Main Goal: Recognition
- [**MS-ASL: A Large-Scale Data Set and Benchmark for Understanding American Sign Language**](http://export.arxiv.org/pdf/1812.01053#page=7)
- [**Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison**](https://www.groundai.com/project/word-level-deep-sign-language-recognition-from-video-a-new-large-scale-dataset-and-methods-comparison/1)
 
#### Stretch Goal: Body Key Points
- [**Co-occurrence Feature Learning from Skeleton Data for Action Recognition and
Detection with Hierarchical Aggregation**](https://www.ijcai.org/Proceedings/2018/0109.pdf)
- [**Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields**](https://arxiv.org/pdf/1611.08050.pdf)
- [**Hand Keypoint Detection in Single Images using Multiview Bootstrapping**](https://arxiv.org/pdf/1704.07809.pdf)
 
#### Stretch Goal: Translation
- [**Video-based Sign Language Recognition without Temporal Segmentation**](https://arxiv.org/abs/1801.10111)
- [**DeepASL: Enabling Ubiquitous and Non-Intrusive Word and Sentence-Level Sign Language Translation**](https://arxiv.org/abs/1802.07584)

### Project Objectives
As discussed above, our main objective is to use computer vision techniques, particularly neural networks to train and recognize ASL words. If possible, we would love to extend this to eventually be able to translate continuous sequences as well and incorporate new data. 

### Proposed Methodologies
We are planning to do preprocessing on our dataset that would involve using an optical flow algorithm to label the direction of the movements of signings. Then we will use a standard 3D CNN architecture, I3D, implemented [**here**](https://github.com/deepmind/kinetics-i3d). MS-ASL also has pretrained weights that we could use that we use in tandem with the datasets to more easily reproduce results.

### Available Resources
#### Main Goal: Recognition -- Datasets
- [**Microsoft American Sign Language Dataset**](https://www.microsoft.com/en-us/research/project/ms-asl/#!downloads)
- [**ASLU (American Sign Language University)**](https://www.lifeprint.com/)
- [**National Center for Sign Language and Gesture Resources (NCSLGR) Corpus**](http://secrets.rutgers.edu/dai/queryPages/querySelection.php)
- We have also reached out to the Department of Linguistics at Gallaudet University for more clean ASL corpus.

#### Main Goal: Recognition -- Code Bases
- [**I3D**](https://github.com/deepmind/kinetics-i3d)

#### Stretch Goal: Body Key Points -- Code Bases
- [**OpenPose**](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

#### Stretch Goal: Translation -- Datasets
- RWTH-PHOENIX-Weather (German Sign Langauge) 
- [**Chinese Sign Language Recognition Dataset**](http://home.ustc.edu.cn/~pjh/openresources/cslr-dataset-2015/index.html)
- [**How 2 Sign**](https://imatge.upc.edu/web/sites/default/files/pub/cDuarteb.pdf)

### Evaluation Plan
MS-ASL is sorted into train, test, and dev sets. Much like the paper we are examining, we plan to evaluate our model on the test set, examining the top-1 and top-5 accuracy. It may be interesting to examine the top-10 accuracy as well as we compare different approaches that we try. 
We will also do a qualitative analysis, where we analyze the test results, and find patterns that the model gets wrong between different signings that are similar. This will thus help in tuning future implementations specific to ASL recognition, or for comparing the type of errors against other models/architectures we might test.