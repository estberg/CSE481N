---
layout: post
title:  "Blog Post 1: Project Ideas"
date:   2020-04-07 19:58:06 -0700
---

**Team Name**: SLIM

**List of Members**:
- Sophie Tian
- Louis Maliyam
- Isaac Pang
- Mitchell Otis Estberg

[**Project Repo**](https://github.com/estberg/CSE481N)

## Overview
Here we propose three project ideas to pursue in CSE481N: NLP Capstone, Spring 2020. We are currently leaning towards working on a combination of our first two ideas: American Sign Language (ASL) Recognition and Translation. We also present a third idea Disambiguating Lexical Trees and in the appendix discuss why we won't pursue working on ASL generation.

## 1. ASL Recognition
### Motivation and Context

American Sign Language (ASL)  incorporates a variety of articulators which makes it a highly challenging recognition task. As signers communicate through their handshape, orientation, movement, upper body, and the face there are many components of the language that need to be captured by recognition models. Further, traditional language models operate on the assumption that language occurs chronologically, but there are cases in which signs are related asynchronously on multiple streams.

There is a significant population of around 500,000 people communicating with ASL, so recognition, translation, interpretation, and generation are all relevant tasks to serve this community. Computer aided translation, interpretation, and generation are all dependent on the performance of recognition. 

### Minimum Viable Action Plan

One goal would be to incorporate larger datasets than those which were previously used in many recognition projects. We are exploring the option of collecting or combining data from various sources. 

At a minimum, we would use existing datasets and try a series of different models on each of these to do word recognition from videos of signing. There are researchers that have tried a series of models on the task of recognition of short videos of signs of individual words and classifying them as these specific words. At a minimum, we would be mostly recreating their work, potentially on a new or expanded dataset. 

Some of the models that worked well for previous research of the task were architectures of CNNs, CNNs with GRU or LSTMs to capture temporality, as well as other approaches. We could attempt using some of these architectures and then expand to incorporate others as well.

### Stretch Goals

The main stretch goal as mentioned above would be incorporating a new, larger dataset and establishing a baseline of performance on this.

Another stretch goal would be adding generated videos to the training dataset. There is some work exploring this for sign language that has demonstrated some performance improvements. 

Finally, as discussed below, another project consideration we are excited about is translation where videos are continuous sequences of signs communicating entire sentences, rather than just individual words. This is another, related task. There are good amounts of work on recognition and translation, but seems to be limited work relating these two closely related tasks. It would be interesting to explore how training a model for word recognition and continuous translation compares and how these can be used together. 

## 2. ASL Translation
### Motivation and Context

There is similar motivation as the above discussion for American Sign Language (ASL) recognition, in that the task of translation serves a large community. Translation is a task that is slightly more applicable to real life in that ultimately, signs are used in sequences of words. 

The task of translation is a bit complicated by the fact that translations between spoken and signed languages (in either direction) are usually done using an intermediary representation of the signed language that can be interpreted by the computer more easily. There are models that have used various versions of these and others that have operated without them entirely. 

There have been attempts to do rule based translation, but recent developments have focused on supervised learning to address this task. 

### Minimum Viable Action Plan

The first step would be assembling or choosing a dataset to work on this problem with. Prior work used small datasets or those with relatively few signers. Further a lot of the cited papers doing this type of work were not using American Sign Language. It would be good to try to use or assemble a collection American Sign Language videos with annotated sentences to use. 

The datasets that have been used in much of this work were not American Sign Language. There is a dataset of clean, real-life video of weather reports signed in German Sign Language called RWTHPHOENIX-Weather 2014T that has been used in much work, as has SIGNUM another dataset of German Sign Language videos with sequences of annotated sentences. 

There is a recent dataset of American Sign Language videos, called How2Sign that hasn’t been cited much, but would be interesting to use for this project as it includes annotated sentences.

From this, we could explore the various models that others have used for the task, as well any others that we might consider to be fitting. Prior work has used CNNs along with models that performed well with languages such as RNNs and HMMs. There was also some prior exploration of incorporating attention. Using these approaches we could establish some baseline performances on the dataset that we use to build off of.

### Stretch Goals

One stretch goal is assembling a new dataset for this task compiled of videos with limited noise (from backgrounds and clothing) and annotations of the sentences signed. 

Another stretch goal could be to run our model on some of the foreign sign language datasets to compare with other work. We could strive for doing better than their performance. 

Finally, as discussed above, exploring the relationship between recognition and translation tasks, and how models addressing each can be used together or improve the performance on translation tasks would be another goal.

## 3. Disambiguating Lexical Trees
### Motivation and Context

Ambiguous sentences can be potentially misterpreted by people, and most definitely by machines. Disambiguating sentences could be helpful in machine understanding of the intent and meaning of sentences like “I shot an elephant in my pajamas.”

### Minimum Viable Action Plan
	
Our initial goal would be a labeled treebank corpora or a dataset with parse tags. We are most likely considering the Penn treebank. From here, we can try vertical or horizontal markovization, which would involve rewriting the corpus with newer and more refined tags. Finally, we could run either an existing CKY algorithm (or one that we create on our own) on those new trees, and look through a sample of our test results manually to see if they are results that we expected, or see if the parse tree is wrong otherwise.

We could also try tag splits, where there are existing methods such as UNARY-RB, UNARY-DT, TAG-PA etc., that refine the current treebank tags and refine them. This would also involve rewriting the treebank, and running CKY for results.

Finally, another thing that we do is use lexicalization, which adds “headwords” to each phrasal node (e.g. S, NP, VP, etc.), which would allow more sensitivity to specific words, instead of basing probability based on a treebank. It would be a process to generate children of a phrasal node based off of a parent.

### Stretch Goals

Some stretch goals might be doing manual annotations, which could create a compact grammar and be more linguistically favorable. We would be splitting NPs into subjects and objects, DTs into a subcategory of DT and demonstratives. However, this involves manually annotating trees which could take a lot longer than the scope of the capstone and isn’t ideal, since there is no strict ruleset for doing manual annotation.

We could try, as a stretch goal, turning the treebanks into sequences, that we can feed into LSTMs and do deep learning over, where the output would be a sequence of tags to a test sentence. But this would require a much larger dataset than the current Penn treebank that is available.

## 4. Appendix: ASL Generation
We were also considering a project on ASL Generation but decided against it (see below if interested). 

### Motivation and Context
Sign language generation (e.g., using Avatars & Computer Graphics to compute animations of humans) can make the information accessible to DHH individuals who prefer sign language to written language (text).

This would make a huge impact to DHH who needs an interpreter, but not able to at that time by various factors.

### Minimum Viable Action Plan
Choose dataset(s) to train on

Create/use a model that can convert spoken/written language into a symbolic representation of the sign language. (Translation)

Build a relationship between the symbolic representation of the sign language to the command for computer graphic software in order to generate the sign language. Maybe Using commercially available graphical human models built for computer games or film industry for making video sequences

Evaluate and improve the performance of the generation

### Stretch Goals
Make the transition of signing generation smoother by providing more parameters on the training dataset such as the length of the pause between words.

### Unfeasible Project
Unltimately, we thought that the challenges that would present in this project may go beyond the scope of the capstone. For example, we have to generate the symbolic representation of the sign language into animation requiring tools and knowledge of computer graphics. We also need to take into account that facial expression, shoulder movements, and body language are important for signing.