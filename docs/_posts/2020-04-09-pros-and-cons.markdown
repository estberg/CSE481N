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
- Dataset: 2000 words from over 100 signers. [dataset available on github](https://github.com/dxli94/WLASL),
[**Microsoft-American Sign Language Dataset**](http://export.arxiv.org/pdf/1812.01053)
- Approach: 
- 1000 words from over 200 signers. [dataset available](https://www.microsoft.com/en-us/research/project/ms-asl/)
- [ASLU (American Sign Language University)](https://www.lifeprint.com/). We have also reached out to the Department of Linguistics at Gallaudet University for more clean ASL corpus.

### 2. ASL Translation
#### Pros
Similar to ASL Recognition, ASL Translation is still understudied yet impactful to the Deaf and Hard of Hearing (DHH) community. The ASL translation is more powerful than ASL Recognition such that it can both translate a single word and a sentence. If successfully developed, it would shrink down the gap between DHHs and people outside of the DHH community.

Because this topic is understudied, our contributions such as adding more datasets, expanding the ideas from existing works, or introducing new ideas would be a meaningful work for the research of this topic. Furthermore, what we have to learn to build this project has excited us, even if it’s challenging.
#### Cons
This project requires an isolated American Sign Language Recognition (SLR), which recognizes word by word, as a building block to translate entire sentences. Therefore, it would be more challenging and can be considered as an extension from ASL Recognition.

ASL has its own grammar structures and unique semantics, as well as English language. We need to do more research on the relationship between these two languages in order to achieve the model that translates ASL into written English language. Another challenging problem is that ASL does not only rely on the hand gestures. Expressions from face, eyebrows, shoulder, and body language in general can also convey messages, in addition to the hand gestures. There is also a time pause between each word and the translation between words and sentences that we need to take care of. It would be exciting to research and work on. However, ASL Recognition seems to fit the timeframe of this capstone class more. It’s still worth it to have this project as a stretch goal of ASL Recognition though.

#### Codebases, Datasets, and Papers
[**Video-based Sign Language Recognition without Temporal Segmentation**](https://arxiv.org/abs/1801.10111), [**DeepASL: Enabling Ubiquitous and Non-Intrusive Word and Sentence-Level Sign Language Translation**](https://arxiv.org/abs/1802.07584)
- Approach: two-stream Convolutional Neural Network (CNN) for video feature representation generation, a Latent Space (LS) for semantic gap bridging, and a Hierarchical Attention Network (HAN) for latent space based recognition
- Datasets: RWTH-PHOENIX-Weather (German Sign Langauge), [**Chinese Sign Language Recognition Dataset**](http://home.ustc.edu.cn/~pjh/openresources/cslr-dataset-2015/index.html), [**National Center for Sign Language and Gesture Resources (NCSLGR) Corpus**](http://secrets.rutgers.edu/dai/queryPages/querySelection.php)



### 3. Disambiguating Lexical Trees
#### Pros
Improving the disambiguation of lexical trees is an interesting continuation of the topic of parsing that NLP ended on last quarter. It might have some practical usages in machine understanding. Since it was covered slightly in the slides for parsing, there are enough resources and documentation to succeed, such as datasets and code.

#### Cons
Datasets might be too limited, considering where the source for the sentences are, and in general, the size of the corpus. This may mean taking more time to experiment and expand on the dataset. Subjectively, another con may be that there is bound to be more time spent analyzing the trees produced by these different methods than there is implementing and experimenting unknown ways. Most importantly, the project may be too limited in scope, and/or it is not really groundbreaking and innovative.

#### Codebases, Datasets, and Papers
- Codebase based off of [Python 3’s nltk](https://www.cs.bgu.ac.il/~elhadad/nlp16/NLTK-PCFG.html), and/or torch, tensorflow. 
- Dataset used will be the [Penn treebank dataset](https://catalog.ldc.upenn.edu/LDC99T42).

### Topics we’d like for lecture or class discussion:
- Basics on ASL, and some current work being done in the field
