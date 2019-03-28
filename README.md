# PoseTrack-Baseline-PyTorch

## Introduction

This is a project which contains all of modules used in Posetrack and I will write a tutorial to teach everyone who knows little about deep learning and computer vision to construct an entire PoseTrack system.

------

In the PoseTrack system, there are three important components.

- **image/video object detection.**
- **human pose estimation.**
- **multiple object tracking.**

If you haven't heard these terms before, don't worry, I will write a detailed tutorial to show you that each module's function in the entire system.

If you want to learn something about image/video-level object detection, human pose estimation , video analysis or multiple object tracking, I think this would be a reliable project :)



**I hope this project will serve as a baseline/codebase and help the future research in PoseTrack task.** 

------

Now, the entire system works with **PyTorch 1.0**. 

Later, I will release another code based on **MXNet**.

## Dataset

As we all known, there are lots of datasets for us to research the object detection, human pose estimation and tracking.

In this project, I will use [coco](http://cocodataset.org/#home), [mpii](http://human-pose.mpi-inf.mpg.de/), [posetrack](https://posetrack.net/) datasets to train our system and test the performance of our system. Because these datasets are very common for researchers or engineers. If you don't know some details or annotation format about above datasets. Don't worry, in the below sections (tutorials) I will teach you some tricks to process these datasets and make you fully understand important datasets in these domains.

Then, I will teach you how to concatenate multiple kinds of dataset to train our model.

- coco dataset: 

- mpii dataset: 

- posetrack dataset: 



## Model

**The** **baseline:**

- **2-stage** method:

1. [mmdetection](https://github.com/open-mmlab/mmdetection) will be used to train our human detector model.

2. [simplebaseline](https://github.com/Microsoft/human-pose-estimation.pytorch) will be used to train our human pose estimation model.

3. [Greedy bipartite](https://en.wikipedia.org/wiki/Matching_(graph_theory)) frame by frame for on-line tracking.

- **1-stage** method:

1. [maskrcnn](https://github.com/facebookresearch/maskrcnn-benchmark) will be used to train our human detector && human pose estimation.
2. [Greedy bipartite](https://en.wikipedia.org/wiki/Matching_(graph_theory)) frame by frame for on-line tracking. 



Of course, this is the most regular operation for constructing the entire system.

 

**The advanced:**

Using temporal information from video is beneficial and after you know how the basic system works, you can read the [advanced tutorial]() to learn some techniques that combine the information from frame to frame to build a more effective system.



## Tutorial 

 Some implementation details and project structures are described in the [tutorial details]().

If you want to know the whole pipeline for PoseTrack system, I strongly recommend that you should read the [tutorial details]() more carefully. After reading this tutorial, you will know how our system works and what is the bottleneck for the entire system.

After you know the basic methods, I also recommend you to learn something about video analysis. So the [advanced tutorial]() would be your next new level!

Enjoy learning, enjoy coding! I think you will get lots of fun in constructing PoseTrack system.



## Installation

Please refer to [install details]() for installation and dataset preparation.



## Inference 

### Test a dataset

#### Test human detector 

#### Test human pose estimation

#### Test multiple object tracking 



### Test some pictures





## Train

### Train human detector

### Train human pose estimation 



## License

This project is released under the [MIT License](https://github.com/ybai62868/Posetrack_baseline_pytorch/blob/master/LICENSE).

------

To be continued.
