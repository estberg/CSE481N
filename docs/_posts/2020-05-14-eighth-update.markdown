---
layout: post
title:  "Blog Post 8: Optimizing I3D"
date:   2020-05-14 23:40:06 -0700
---

## Updates from Team SLIM on ASL Recognition

### Progress with I3D
We corrected our accuracy calculation and have improved results to report. This was with a cross entropy loss function, momentum optimizer, batch size of 8, learning rate of 0.01, momentum of 0.9, and sample frame size of 64. Running the model for 23 epochs, we found that validation accuracy was at its lowest after 11 epochs at 34.2%. Each epoch runs for around 12 minutes still, but with our updated accuracy calculation, evaluation is much faster. From this run we report the following results.

![Graph](https://estberg.github.io/CSE481N/assets/training.jpg)

We then updated our preprocessing to include samples that are of shorter lengths than our desired frame size. We do this by randomly stretching with the first or last frame, duplicating this frame until we have the desired number of frames. This increases the number of samples that can be used for training, so each epoch takes longer. Using the same model as above, we were able to reach training accuracy of 22.7% and validation accuracy of 14.5% after 11 epochs with this approach of preprocessing. We then stopped running the model to try with hyperparameters suggested by researchers.

We contacted two research teams that used I3D for sign language classification to ask about their hyperparameters used in training. One quickly shared their code base and we have tried training on it. 

### Progress with a YOLO Based Approach
We focused on I3D this week, as we get closer to duplication of prior researchers' work.

### Goals, Immediate Steps, and Failing Fast
* We are planning to continue exploring the hyperparameters to use with I3D for successful training. We will try with the hyperparameters we received from researchers as well others that might make sense. Note that this might take us a while to explore because the training takes a significant amount of time.
* We want to add accuracy checks for top5 and top10 classification. Although this has not been reported on the MS-ASL dataset we are using, it can be useful in evaluating our implementation of I3D. 
* We also intend to add random horizontal flipping to our preprocessing, as this was cited by researchers and relates to the lack of handedness in ASL. 
