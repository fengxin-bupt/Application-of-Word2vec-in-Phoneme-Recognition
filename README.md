# Application-of-Word2vec-in-Phoneme-Recognition
Build an attention-based model for speech recogntion (Listen Attend and Spell).Use the Word2vec model to help to train the attention model.

Some code is quoted from https://github.com/thomasschmied/Speech_Recognition_with_Tensorflow. But there are some mistakes in his file, which i have changed these mistakes.

Our model is built based on phoneme recognition. The datasets used are librispeech(http://www.openslr.org/12) and TIMIT.The pronunciation dictionary used is the 39 phoneme pronunciation dictionary of CMU.

We used a new method in the model. The word2vec model is used to initialize the embedding matrix in the attention model.This can make the distance between the embedding vectors larger, so it can improve the performance of the model.At the same time, in order to solve the overfitting problem of the attention dataset on the attention model.We use a new phoneme inverse mapping strategy to convert more 39 phoneme datasets to 61 phoneme datasets.

Dataset: Librispeech, TIMIT
Feature: 40 mel-filterbank + delta + delta delta
Encoder: 512BLSTM
Decoder: 512LSTM

Network frame work:


![image](https://github.com/fengxin-bupt/Application-of-Word2vec-in-Phoneme-Recognition/blob/master/image/network.PNG)

Result:
![image](https://github.com/fengxin-bupt/Application-of-Word2vec-in-Phoneme-Recognition/blob/master/image/result.PNG)


This is my paper (《Application of Word2vec in Phoneme Recognition》) about how to make this experiment and result.

