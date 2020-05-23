---
layout: post
title:  "Blog Post 9: Training I3D and Reflecting on YOLO"
date:   2020-05-22 22:55:00 -0700
---

## Updates from Team SLIM on ASL Recognition

### Progress Training I3D
Based on the hyperparameters that we received from the researchers we tried many different versions of our models. However, none led to significant improvements in performance unfortunately, and our validation accuracy remained around 30% with the best accuracy of 34.2% as in the last blog post. In addition to these hyperparameters we added image flipping, because the right and left hands of an image will not matter and a horizontal flip can occur randomly with probability of 0.5 in each sample.

![Graph](https://docs.google.com/spreadsheets/d/e/2PACX-1vTxJH97upHjrTlpwo7EDn3b5laO267pFYKXLTE1QiSI-K5io2FGyJm8ST7iIbNLOKQA7OCoPP_5qMlo/pubchart?oid=360710499&amp;format=image)

All of the trials in the above graph have a batch size of 8, initial learning rate of 0.1, and momentum of 0.9, and a dropout rate of 0.7 unless otherwise noted.

MMTDropout0.5 uses standard momentum and has a dropout rate of 0.5. Samples are flipped horizontally with a probability of 0.5.

DecayMMTShortSamplesFlipImages uses exponential decay on the learning rate and trains on shortened samples (clipped to the specified time frame). There are 5 decay steps and a decay rate of 0.95. Samples are flipped horizontally with a probability of 0.5.

DecayMMTFullSamplesFlipImages uses exponential decay on the learning rate and trains on full samples (clipped to the specified time frame). There are 5 decay steps and a decay rate of 0.95. Samples are flipped horizontally with a probability of 0.5. 

DecayMMTFullSamples uses exponential decay on the learning rate and trains on full samples (clipped to the specified time frame). There are 5 decay steps and a decay rate of 0.95. 

DecayMMT uses exponential decay on the learning rate and trains on shortened samples (stretched to the specified time frame).

Momentum does not use any decay on the learning rate, and uses the full samples (clipped to the frame count).

See the appendix for the detailed results. 

We have a few ideas for further improvements too, as discussed below in goals, immediate steps, and failing fast.

### Reflections on a YOLO Based Approach
YOLO only does object recognition and sets bounding boxes on the videos. We can take some time to run it on multiple stacked frames to extract only specific objects to obtain cleaned temporal streams to use in other models, like I3D. Possibilities related to YOLO work include:
- isolating the hands/skeleton from the backgrounds
- recognizing certain movements such as pedestrians (network structure YOLO-R) which might be the next thing to look at
- combine YOLO (pre-processor) and I3D into 1 pipeline
We will investigate these further while we continue with the I3D implementation. Our ultimate goal for this quarter will be documenting our findings regarding YOLO in ASL recognition and so we can prepare the future researchers. Some of us might continue to work on proof of concept regarding this in the summer.

### Goals, Immediate Steps, and Failing Fast
* As with last week, we are planning to continue exploring the hyperparameters to use with I3D for successful training. Now that we have Note that this might take us a while to explore because the training takes a significant amount of time.
* Although we believe that we are using the same preprocessing techniques described in the papers, we believe that there could be some bug in our data loading or preprocessing that is causing our accuracy in training the model.
* We want to add accuracy checks for top5 and top10 classification. Although this has not been reported on the MS-ASL dataset we are using, it can be useful in evaluating our implementation of I3D. 
* We also intend to add random horizontal flipping to our preprocessing, as this was cited by researchers and relates to the lack of handedness in ASL.  
* Error analysis would also give us and future researchers some insights on what word classes needed to be focused on and what could be improved in the way we trained the model.

### Appendix 

Below are the results of each training session of the model. All of the trials have a batch size of 8, initial learning rate of 0.1, and momentum of 0.9, and a dropout rate of 0.7 unless otherwise noted.

MMTDropout0.5 uses standard momentum and has a dropout rate of 0.5. Samples are flipped horizontally with a probability of 0.5.

| Epochs | Loss       | Train Accuracy | Validation Accuracy |
|--------|------------|----------------|---------------------|
| 1      | 3.823176   | 0.13283063     | 0.09297521          |
| 2      | 2.1673198  |                | 0.22933884          |
| 3      | 1.5391798  |                | 0.24173554          |
| 4      | 1.3765696  |                | 0.15599174          |
| 5      | 1.0094106  |                | 0.21177686          |
| 6      | 0.7180179  |                | 0.30578512          |
| 7      | 0.558643   |                | 0.28822314          |
| 8      | 0.38285276 |                | 0.24173554          |
| 9      | 0.9703902  |                | 0.30061983          |
| 10     | 0.6003172  |                | 0.25619835          |
| 11     | 0.34020698 | 0.48955916     | 0.28409091          |
| 12     | 0.32361126 |                | 0.20557851          |
| 13     | 0.5639672  |                | 0.24896694          |
| 14     | 0.00370924 |                | 0.23863636          |
| 15     | 0.1468808  |                | 0.25                |
| 16     | 0.0807769  |                | 0.24070248          |
| 17     | 0.00424493 |                | 0.25413223          |
| 18     | 0.01996285 |                | 0.2696281           |
| 19     | 0.01309926 |                | 0.21590909          |
| 20     | 0.2087551  |                | 0.25619835          |
| 21     | 0.5788877  | 0.54930394     | 0.28719008          |
| 22     | 0.02746165 |                | 0.27892562          |
| 23     | 0.02144082 |                | 0.22933884          |
| 24     | 0.03246434 |                | 0.2231405           |
| 25     | 0.02324902 |                | 0.22727273          |
| 26     | 0.2511135  |                | 0.22933884          |
| 27     | 0.01381906 |                | 0.23863636          |
| 28     | 0.24776675 |                | 0.18595041          |
| 29     | 0.05134327 |                | 0.24896694          |
| 30     | 0.00297015 |                | 0.2107438           |
| 31     | 0.02500087 | 0.36658933     | 0.19421488          |

DecayMMTShortSamplesFlipImages uses exponential decay on the learning rate and trains on shortened samples (clipped to the specified time frame). There are 5 decay steps and a decay rate of 0.95. Samples are flipped horizontally with a probability of 0.5.

| Epochs | Loss        | Train Accuracy | Validation Accuracy |
|--------|-------------|----------------|---------------------|
| 1      | 4.1009827   | 0.09599768     | 0.069214876         |
| 2      | 3.835241    |                | 0.179752066         |
| 3      | 1.9513866   |                | 0.224173554         |
| 4      | 1.1688004   |                | 0.224173554         |
| 5      | 1.0772955   |                | 0.199380165         |
| 6      | 1.3350482   |                | 0.209710744         |
| 7      | 0.4147607   |                | 0.246900826         |
| 8      | 1.6197646   |                | 0.178719008         |
| 9      | 0.33156908  |                | 0.190082645         |
| 10     | 0.48407125  |                | 0.132231405         |
| 11     | 0.87069017  | 0.369489559    | 0.175619835         |
| 12     | 0.93436277  |                | 0.123966942         |
| 13     | 0.25887847  |                | 0.150826446         |
| 14     | 0.08142643  |                | 0.17768595          |
| 15     | 0.23576011  |                | 0.146694215         |
| 16     | 0.27154505  |                | 0.161157025         |
| 17     | 0.38070866  |                | 0.134297521         |
| 18     | 0.029678682 |                | 0.152892562         |
| 19     | 0.24533346  |                | 0.19214876          |
| 20     | 0.10368898  |                | 0.144628099         |
| 21     | 0.052036025 | 0.332656613    | 0.147727273         |
| 22     | 0.04666648  |                | 0.16838843          |
| 23     | 0.042864107 |                | 0.132231405         |
| 24     | 0.037424497 |                | 0.129132231         |
| 25     | 0.011143027 |                | 0.149793388         |
| 26     | 0.023084812 |                | 0.142561983         |
| 27     | 0.0758645   |                | 0.126033058         |
| 28     | 0.051862907 |                | 0.123966942         |
| 29     | 0.006350528 |                | 0.141528926         |
| 30     | 0.040007688 |                | 0.142561983         |
| 31     | 0.005015978 | 0.260150812    | 0.115702479         |
| 32     | 0.008238992 |                | 0.122933884         |
| 33     | 0.010086702 |                | 0.129132231         |
| 34     | 0.03130296  |                | 0.145661157         |
| 35     | 0.008890851 |                | 0.148760331         |
| 36     | 0.025609817 |                | 0.134297521         |
| 37     | 0.04106762  |                | 0.116735537         |
| 38     | 0.01059778  |                | 0.138429752         |
| 39     | 0.002870023 |                | 0.135330579         |
| 40     | 0.014021246 |                | 0.136363636         |
| 41     | 0.002829054 | 0.335266821    | 0.155991736         |

DecayMMTFullSamplesFlipImages uses exponential decay on the learning rate and trains on full samples (clipped to the specified time frame). There are 5 decay steps and a decay rate of 0.95. Samples are flipped horizontally with a probability of 0.5. 

| Epochs | Loss        | Train Accuracy | Validation Accuracy |
|--------|-------------|----------------|---------------------|
| 1      | 3.0895405   | 0.107888631    | 0.087809917         |
| 2      | 2.4449692   |                | 0.159090909         |
| 3      | 1.7798424   |                | 0.225206612         |
| 4      | 1.2374082   |                | 0.26446281          |
| 5      | 1.3520098   |                | 0.225206612         |
| 6      | 0.73245215  |                | 0.208677686         |
| 7      | 0.2687706   |                | 0.273760331         |
| 8      | 1.1641657   |                | 0.215909091         |
| 9      | 0.15552877  |                | 0.275826446         |
| 10     | 0.83024573  |                | 0.195247934         |
| 11     | 0.315046    | 0.450406032    | 0.259297521         |
| 12     | 0.1687068   |                | 0.215909091         |
| 13     | 0.15892723  |                | 0.262396694         |
| 14     | 0.2851023   |                | 0.203512397         |
| 15     | 0.05388772  |                | 0.291322314         |
| 16     | 0.34991905  |                | 0.30268595          |
| 17     | 0.07025026  |                | 0.263429752         |
| 18     | 0.29238862  |                | 0.258264463         |
| 19     | 0.4238856   |                | 0.308884298         |
| 20     | 0.030839832 |                | 0.290289256         |
| 21     | 0.010568748 | 0.481148492    | 0.260330579         |
| 22     | 0.011816612 |                | 0.294421488         |
| 23     | 0.013791129 |                | 0.297520661         |
| 24     | 0.042471677 |                | 0.260330579         |
| 25     | 0.028426033 |                | 0.270661157         |
| 26     | 0.025353407 |                | 0.280991736         |
| 27     | 0.00360061  |                | 0.27892562          |
| 28     | 0.5712563   |                | 0.27892562          |
| 29     | 0.07429622  |                | 0.247933884         |
| 30     | 0.010254495 |                | 0.274793388         |
| 31     | 0.022556359 | 0.494779582    | 0.262396694         |
| 32     | 0.013255281 |                | 0.231404959         |
| 33     | 0.047998935 |                | 0.262396694         |
| 34     | 0.010247436 |                | 0.247933884         |
| 35     | 0.00697397  |                | 0.238636364         |
| 36     | 0.037614055 |                | 0.256198347         |
| 37     | 0.003231146 |                | 0.223140496         |
| 38     | 0.009569564 |                | 0.234504132         |
| 39     | 0.003698238 |                | 0.269628099         |
| 40     | 0.017805466 |                | 0.242768595         |
| 41     | 0.010660537 | 0.52900232     | 0.269628099         |

DecayMMTFullSamples uses exponential decay on the learning rate and trains on full samples (clipped to the specified time frame). There are 5 decay steps and a decay rate of 0.95. 

| Epochs | Loss        | Train Accuracy | Validation Accuracy |
|--------|-------------|----------------|---------------------|
| 1      | 3.9886448   | 0.128480278    | 0.085743802         |
| 2      | 2.6876895   |                | 0.19731405          |
| 3      | 1.7240386   |                | 0.232438017         |
| 4      | 1.0409421   |                | 0.226239669         |
| 5      | 1.0047655   |                | 0.26446281          |
| 6      | 0.34726045  |                | 0.224173554         |
| 7      | 1.3556147   |                | 0.232438017         |
| 8      | 0.72222245  |                | 0.234504132         |
| 9      | 0.21729718  |                | 0.222107438         |
| 10     | 0.06617008  |                | 0.191115702         |
| 11     | 0.055552494 | 0.391531323    | 0.175619835         |
| 12     | 0.08448218  |                | 0.219008264         |
| 13     | 0.2538858   |                | 0.237603306         |
| 14     | 0.15053442  |                | 0.277892562         |
| 15     | 0.040565424 |                | 0.261363636         |
| 16     | 0.0591738   |                | 0.196280992         |
| 17     | 0.05683233  |                | 0.219008264         |
| 18     | 0.049134895 |                | 0.160123967         |
| 19     | 0.04725416  |                | 0.19731405          |
| 20     | 0.10584024  |                | 0.225206612         |
| 21     | 0.023177516 | 0.413283063    | 0.196280992         |
| 22     | 0.008275314 |                | 0.195247934         |
| 23     | 0.00583194  |                | 0.216942149         |
| 24     | 0.03894534  |                | 0.256198347         |
| 25     | 0.012585431 |                | 0.215909091         |
| 26     | 0.009632698 |                | 0.263429752         |
| 27     | 0.18786643  |                | 0.253099174         |
| 28     | 0.018138433 |                | 0.241735537         |
| 29     | 0.004943784 |                | 0.241735537         |
| 30     | 0.014819983 |                | 0.245867769         |
| 31     | 0.005856448 | 0.507830626    | 0.237603306         |
| 32     | 0.013255281 |                | 0.231404959         |
| 33     | 0.047998935 |                | 0.262396694         |
| 34     | 0.010247436 |                | 0.247933884         |
| 35     | 0.00697397  |                | 0.238636364         |
| 36     | 0.037614055 |                | 0.256198347         |
| 37     | 0.003231146 |                | 0.223140496         |
| 38     | 0.009569564 |                | 0.234504132         |
| 39     | 0.003698238 |                | 0.269628099         |
| 40     | 0.017805466 |                | 0.242768595         |
| 41     | 0.010660537 | 0.52900232     | 0.269628099         |

DecayMMT uses exponential decay on the learning rate and trains on shortened samples (stretched to the specified time frame).

| Epochs | Loss       | Train Accuracy | Validation Accuracy |
|--------|------------|----------------|---------------------|
| 1      | 4.464263   | 0.0962877      | 0.07231405          |
| 2      | 2.6317673  |                | 0.10950413          |
| 3      | 2.01261    |                | 0.19214876          |
| 4      | 1.6689364  |                | 0.20867769          |
| 5      | 1.2764965  |                | 0.17252066          |
| 6      | 0.7497402  |                | 0.21590909          |
| 7      | 1.5019792  |                | 0.21384298          |
| 8      | 1.002589   |                | 0.20764463          |
| 9      | 0.40192673 |                | 0.21694215          |
| 10     | 0.26164654 |                | 0.22417355          |
| 11     | 0.46887222 | 0.50522042     | 0.18904959          |
| 12     | 0.4533583  |                | 0.20557851          |
| 13     | 0.04273066 |                | 0.18904959          |
| 14     | 0.04561602 |                | 0.19731405          |
| 15     | 0.04050817 |                | 0.21384298          |
| 16     | 0.13813794 |                | 0.19008264          |
| 17     | 0.01545291 |                | 0.1838843           |
| 18     | 0.0250378  |                | 0.18491736          |
| 19     | 0.03252499 |                | 0.20144628          |
| 20     | 0.01152985 |                | 0.20041322          |
| 21     | 0.02473294 | 0.58961717     | 0.21280992          |
| 22     | 0.07082658 |                | 0.20247934          |
| 23     | 0.02743219 |                | 0.18078512          |
| 24     | 0.04117746 |                | 0.19318182          |
| 25     | 0.01361253 |                | 0.19318182          |
| 26     | 0.04341892 |                | 0.18181818          |
| 27     | 0.03743726 |                | 0.18904959          |
| 28     | 0.00627785 |                | 0.19214876          |
| 29     | 0.0083513  |                | 0.19731405          |
| 30     | 0.03056581 |                | 0.20144628          |
| 31     | 0.03275628 | 0.62035963     | 0.22107438          |
| 32     | 0.00810476 |                | 0.2035124           |
| 33     | 0.00552511 |                | 0.20041322          |
| 34     | 0.002808   |                | 0.1911157           |
| 35     | 0.01391781 |                | 0.17975207          |
| 36     | 0.00616093 |                | 0.18698347          |
| 37     | 0.13104701 |                | 0.20557851          |
| 38     | 0.00260316 |                | 0.20971074          |
| 39     | 0.00797265 |                | 0.20971074          |
| 40     | 0.01229339 |                | 0.19214876          |
| 41     | 0.01200027 | 0.56206497     | 0.19731405          |
| 42     | 0.00549112 |                | 0.19524793          |
| 43     | 0.01657116 |                | 0.19008264          |
| 44     | 0.03981929 |                | 0.20041322          |
| 45     | 0.00722971 |                | 0.19318182          |
| 46     | 0.00368284 |                | 0.21384298          |
| 47     | 0.00294622 |                | 0.20867769          |
| 48     | 0.00816746 |                | 0.20247934          |
| 49     | 0.00314474 |                | 0.19628099          |
| 50     | 0.01482229 |                | 0.19318182          |
| 51     | 0.00106964 | 0.57221578     | 0.18595041          |
| 52     | 0.00188565 |                | 0.20247934          |
| 53     | 0.00126775 |                | 0.20454545          |
| 54     | 0.0037967  |                | 0.21487603          |
| 55     | 0.01080826 |                | 0.20661157          |
| 56     | 0.00182252 |                | 0.19938017          |
| 57     | 0.0041876  |                | 0.19524793          |
| 58     | 0.01167414 |                | 0.21280992          |
| 59     | 0.00487541 |                | 0.21487603          |
| 60     | 0.00190017 |                | 0.20041322          |
| 61     | 0.00696128 | 0.57598608     | 0.20661157          |

Momentum does not use any decay on the learning rate, and uses the full samples (clipped to the frame count).

| Epochs | Loss       | Train Accuracy | Validation Accuracy |
|--------|------------|----------------|---------------------|
| 1      | 3.8365674  | 0.11772152     | 0.09926471          |
| 2      | 1.8286582  |                | 0.18259804          |
| 3      | 1.3686571  |                | 0.21691176          |
| 4      | 1.6641523  |                | 0.2120098           |
| 5      | 1.0091385  |                | 0.22181373          |
| 6      | 1.3463925  |                | 0.32107843          |
| 7      | 0.322431   |                | 0.31372549          |
| 8      | 0.21857394 |                | 0.25612745          |
| 9      | 0.7673668  |                | 0.30882353          |
| 10     | 0.05052156 |                | 0.31740196          |
| 11     | 0.39597234 | 0.69683544     | 0.34191176          |
| 12     | 0.3225846  |                | 0.31740196          |
| 13     | 0.02298437 |                | 0.32107843          |
| 14     | 0.0377494  |                | 0.33455882          |
| 15     | 0.05228121 |                | 0.33455882          |
| 16     | 0.0377458  |                | 0.27818627          |
| 17     | 0.10595778 |                | 0.26838235          |
| 18     | 0.09384969 |                | 0.33088235          |
| 19     | 0.12046275 |                | 0.22426471          |
| 20     | 0.01750935 |                | 0.29289216          |
| 21     | 0.03240437 | 0.64968354     | 0.33088235          |
| 22     | 0.02304003 |                | 0.27696078          |
| 23     | 0.01955632 |                | 0.28063725          |
