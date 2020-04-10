---
layout: post
title:  "Blog Post 2: Project Pros and Cons"
date:   2020-04-09 21:29:06 -0700
---

### 1. ASL Recognition
#### Pros
American Sign Language Recognition, as a branch of  Natural Language Processing, is a relatively unexplored problem. Although there is a body of prior work, there is definitely room for improvement, expanding the work to include more data, and trying new approaches. Whether we are retrying old approaches on new datasets or bringing our own ideas, it will be meaningful work to expand on this body of research. Further, it is a problem that excites us and is useful! 

Another pro is that this is a slightly simpler task than translation. Focusing just on individual words is easier than sentences with complex grammar structures. There are also more and bigger datasets of videos of ASL that are annotated by individual words. 

#### Cons
This is a challenging problem. Although a language analysis task, it relies heavily on computer vision. Although Isaac has taken computer vision and Mitchell has taken deep learning, we will need to make sure to expand our group’s knowledge of computer vision to address the image and video processing to address the task. Further, recognition is less applicable to the real world. 

#### Codebases, Datasets, and Papers
[**Word-level Deep Sign Language Recognition**](https://www.groundai.com/project/word-level-deep-sign-language-recognition-from-video-a-new-large-scale-dataset-and-methods-comparison/1)
- Approach: 
- Dataset: 2000 words from over 100 signers. [dataset available on github](https://github.com/dxli94/WLASL)
[**Microsoft-American Sign Language Dataset**](http://export.arxiv.org/pdf/1812.01053)
- Approach: 
- 1000 words from over 200 signers. [dataset available](https://www.microsoft.com/en-us/research/project/ms-asl/) 

### 2. ASL Translation
#### Pros

#### Cons

#### Codebases, Datasets, and Papers
[**Video-based Sign Language Recognition without Temporal Segmentation**](https://arxiv.org/abs/1801.10111)
- Approach: two-stream Convolutional Neural Network (CNN) for video feature representation generation, a Latent Space (LS) for semantic gap bridging, and a Hierarchical Attention Network (HAN) for latent space based recognition
- Datasets: RWTH-PHOENIX-Weather (German Sign Langauge), [Chinese Sign Language Recognition Dataset](http://home.ustc.edu.cn/~pjh/openresources/cslr-dataset-2015/index.html)


### 3. Disambiguating Lexical Trees
#### Pros
Improving the disambiguation of lexical trees is an interesting continuation of the topic of parsing that NLP ended on last quarter. It might have some practical usages in machine understanding. Since it was covered slightly in the slides for parsing, there are enough resources and documentation to succeed, such as datasets and code.

#### Cons
Datasets might be too limited, considering where the source for the sentences are, and in general, the size of the corpus. This may mean taking more time to experiment and expand on the dataset. Subjectively, another con may be that there is bound to be more time spent analyzing the trees produced by these different methods than there is implementing and experimenting unknown ways. Most importantly, the project may be too limited in scope, and/or it is not really groundbreaking and innovative.

#### Codebases, Datasets, and Papers
- Codebase based off of Python 3’s nltk (https://www.cs.bgu.ac.il/~elhadad/nlp16/NLTK-PCFG.html), and/or torch, tensorflow. 
- Dataset used will be the Penn treebank dataset (https://catalog.ldc.upenn.edu/LDC99T42) and ASLU (American Sign Language University) https://www.lifeprint.com/. We have also reached out to the Department of Linguistics at Gallaudet University for more clean ASL corpus.

### Topics we’d like for lecture or class discussion:
Maybe basics over ASL? (tentative, can delete if you guys want)
