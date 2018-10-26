## Temporal Convolutional Network Language Model Fine-tuned for Text Classification 

My attempt to reproduce Temporal Convolutional Network (TCN) from the paper ["An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling"](https://arxiv.org/abs/1803.01271) on Wiki-2 text data. 

The goal was to train TCN as a language model on Wiki-2 data, and fine tune the language model for text classication on imdb, following the success of the recent approach "Universal Language Model Fine-tuning for Text Classification" (ULMFIT) by Jeremy Howard and Sebastian Ruder described in the [paper](https://arxiv.org/abs/1801.06146) 

Written with fastai v0.7 library (and Pytorch)
