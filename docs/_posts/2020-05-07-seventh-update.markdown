---
layout: post
title:  "Blog Post 7: Training with I3D and Updates On YOLO"
date:   2020-05-07 23:55:43 -0700
---

## Updates from Team SLIM on ASL Recognition

### Progress with I3D
We do plan to continue with the I3D approach in addition to our exploration of employing YOLO in an algorithm because it could be an approach used alongside YOLO (to handle the temporal aspect) and since we feel we are fairly close in recreation of the papers that used I3D on our classification task. 

In addition from the previous blog post, we start training MS-ASL dataset with the I3D model. The weights for each layer were initialized from the pre-trained weights of the Kinetics-I3D model except the last layer in which we modify the shape of the layer to output one of 100 classes instead of 400 classes. Note that we are focusing on the top-100 classes of MS-ASL dataset for now.

The model was trained using Gradient Descent Optimizer with the learning rate of 0.9 and cross-entropy loss function. Since the model is complex and each input video was extracted into 64 frames, we only got a chance to run it for one epoch. We used a batch size of 8 samples to start, but want to explore what resources we have to try different batch sizes. A single epoch took 12:46 to run. We recognize that that learning rate of 0.9 is probably far too large, but just started with this for the first run with a small number of epochs and will have to adjust this hyperparameter as we continue. We also want to try with different optimizers and loss functions, time permitting as a way to improve performance.

After the first epoch, the initial loss was 4.96, initial train accuracy 0.0% was, and initial validation accuracy was 0.0%. This was still in the stages of setting up for training, so we look forward to running more epochs and tests and seeing how we can train and improve this approach. Also, before we added accuracy checks, we did see the loss go down over epochs. The loss after the second epoch for example was 3.10.

### Progress with a YOLO Based Approach
The Microsoft researchers used YOLO for character detection (fingerspelling, where the signer is signing static characters) with a good deal of success. We thought about trying to work on fingerspelling too, which could incorporate an aspect of translation. This week we set up their code base and got it ready to run. Unfortunately, though there are limited data sets for this. This week we tried contacting some people about fingerspelling datasets we could use as a first project with YOLO but without time to collect more data, we don’t plan to continue with that specific work of fingerspelling, unless data becomes available.

However, it has been a goal of the researchers to expand their approach to word recognition which is what we would like to do. We are still debating different ways we can incorporate YOLO into an approach for word classification.

### Goals, Immediate Steps, and Failing Fast
* We are planning to continue training the I3D model for more epochs. 
* Also, we may need to experiment with different loss functions and optimizers with different learning rates to train the model.
* We will discuss our approach of incorporating YOLO into our work and get a working plan for how to do this. 
