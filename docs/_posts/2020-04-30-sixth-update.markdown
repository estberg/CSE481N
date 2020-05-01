---
layout: post
title:  "Blog Post 6: Working with I3D and YOLO"
date:   2020-04-30 23:55:43 -0700
---

## Updates from Team SLIM on ASL Recognition

### Progress with I3D
We spent some time this week continuing to work on getting I3D working and continued to run into some roadblocks. The first was image size, which we were able to address by resizing the images. The second issue was loading the weights of the pretrained models, which were resolved by uninstalling and downgrading various Tensorflow packages. Another issue was with mismatched versions of CUDA stopping Tensorflow from working properly. 

We finally got I3D working so that we can pass a video sample through it and make a prediction! This was very exciting, but there is of course still work to be done to get this approach working for our task. We were able to load the pretrained weights which are for 400 action classes trained on the kinetics 400 dataset. We need to learn about using these pretrained weights to start training for the 100 classes we want to initially focus on. 

We also realized that the I3D implementation that we forked and have been working to duplicate is written for a very outdated version of Tensorflow. We’ve started work on a version in the most up to date version of Tensorflow to address some compatibility issues.

### Meeting with Microsoft Researchers
On Tuesday night, we met with researchers at Microsoft who have worked on using a YOLO based approach to build a detector for signs of letters. They trained the model to detect the signs for ‘C’, ‘O’, ‘F’, and ‘E’ (to spell COFFEE) and set further goals to take this approach and apply it to word recognition, which would be novel as far as they know. To train their model, they started with the pretrained weights and used around 400 or 500 sample videos for the 4 signs. They preprocessed these in a number of ways to yield around 3,500 augmented samples. After this meeting we have decided to consider using an approach based on YOLO for word recognition as well  (we may return to I3D if we run into insurmountable problems with YOLO). 

### Progress with a YOLO Based Approach
Since the meeting on Tuesday, we mostly focused on setting up YOLO to see if we can get things running and did a simple search to see if there were any other cases of YOLO being used for video classification before. 

In terms of getting YOLO running, we were able to clone the researchers’ repo, and download the YOLOv3 weights from [the website](https://pjreddie.com/darknet/yolo/). Then we will be able to convert the DarkNet YOLO weights to Keras model and this will let us run the detection for the pre trained images detection that the Microsoft researchers have prepared. 

However, the signed letters detection performed by the researchers is a little different than our task of word recognition. This is because the letters they trained their model for are all static signs. They do not involve a temporal aspect or motion. So, we knew that as we took their approach and progressed into word recognition, we would need to consider how we would handle this temporal aspect. In our simple search for other work related to this, we found [a paper](http://cs231n.stanford.edu/reports/2017/pdfs/707.pdf) that used YOLO in video classification. YOLO was used to collect a region of interest (in this case the person or people) of each frame and then these extracted crops could be processed in any standard approach for handling video classification (they used two-stream 2D CNN). There was also [one paper](https://ieeexplore-ieee-org.offcampus.lib.washington.edu/document/8329933/references#references) that took an approach similar to that discussed above for sign language recognition. They used YOLO to remove all parts of frames of signers, clothing, and the background out of the frames, such that each frame only contained a small region of interest of the hands on a black background. Then they stitched together the frames of a sign into a “wide image” .This does not seem like the best approach towards the temporal aspect, but the use of YOLO here does seem like an interesting approach. 

### Goals, Immediate Steps, and Failing Fast

Our ultimate goal at this time will be to do word recognition on a single ASL word. Currently, the YOLO solution we have is to recognize a static sign (fingerspelling for ‘C’, ‘F’, ‘O’, ‘E’). We discussed one option of the first thing we could do would be add a few more letters, such as `S`, `U` to have the word `FOCUS`. This will ensure we understand the current state of the YOLO code [we have from the researchers](https://github.com/prabaskrishnan/asl_finger). However, we are still inquiring about data to do this. 

To do's are a lot. We are still trying to work on framing a fail fast goal first of all. 
 
After that, our immediate milestone would be to define what we are going to do for a word model, which includes the labeling. One option here is to use (or modify) the labeling in the MS-ASL dataset. Currently we do not have access to code that was used to produce the related papers, which used I3D. This labeling assumes the video contains just one word (the videos are only a couple of seconds long).
