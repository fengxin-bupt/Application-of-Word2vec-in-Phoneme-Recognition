#-*- encoding:utf-8 -*-
#Using the word2vec system that comes with tensorflow, the default num_skips = 1 skip_window = 2
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
from config import Hparams
import numpy as np
import tensorflow as tf
import pickle

class Embedded(object):
    """docstring for Embedded"""
    def __init__(self):
        super(Embedded, self).__init__()
        self.embedding_size = Hparams.embedding_dim
        self.batch_size = Hparams.embedded_batch_size
        self.num_skips = Hparams.num_skips
        self.skip_window = Hparams.skip_window
        self.learning_rate = 0.001
        assert self.batch_size % self.num_skips == 0
        assert self.num_skips <= 2 * self.skip_window
        self.data_index = 0
        self.load_data(Hparams.embedding_voc_path)
        self.dataset()
        self.model()
    def load_data(self,path):
        f = open(path,'rb')
        embed_data = pickle.load(f)
        f.close()
        self.data = embed_data
    def generate_batch(self):
        batch = np.ndarray(shape=(self.batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
        span = 2 * self.skip_window + 1  # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin
        if (self.data_index + span) > len(self.data):
            self.data_index = 0
        buffer.extend(self.data[self.data_index:self.data_index + span])
        self.data_index += span
        for i in range(self.batch_size // self.num_skips):
            context_words = [w for w in range(span) if w != self.skip_window]
            words_to_use = random.sample(context_words, self.num_skips)
            for j, context_word in enumerate(words_to_use):
                batch[i * self.num_skips + j] = buffer[self.skip_window]
                labels[i * self.num_skips + j, 0] = buffer[context_word]
            if self.data_index == len(self.data):
                buffer.extend(self.data[0:span])
                self.data_index = span
            else:
                buffer.append(self.data[self.data_index])
                self.data_index += 1

        # Backtrack a little bit to avoid skipping words in the end of a batch
        self.data_index = (self.data_index - span) % len(self.data)
        return batch, labels

    def dataset(self):
        self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
        self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])

    def model(self):
        valid_examples = np.array([int(i) for i in range(43)])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
        
        embeddings = tf.get_variable('embedding_matrix',shape = [43,self.embedding_size],dtype = tf.float32,
            initializer = tf.random_uniform_initializer(minval = -1.0,maxval = 1.0),trainable = True)
        
        
        self.embed = tf.nn.embedding_lookup(embeddings, self.train_inputs)

        with tf.name_scope('weights'):
            nce_weights = tf.Variable(
                tf.truncated_normal([43, self.embedding_size],
                                stddev=1.0 / math.sqrt(self.embedding_size)))
        with tf.name_scope('biases'):
            nce_biases = tf.Variable(tf.zeros([43]))
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(
                  tf.nn.nce_loss(
                      weights=nce_weights,
                      biases=nce_biases,
                      labels=self.train_labels,
                      inputs=self.embed,
                      num_sampled=self.batch_size//2,
                      num_classes=43))
        self.global_step = tf.Variable(0, trainable = False)
        self.learning_rate = tf.train.exponential_decay(learning_rate = self.learning_rate,global_step = self.global_step,
                                                        decay_steps = 1000,decay_rate = 0.98)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                              valid_dataset)
        self.similarity = tf.matmul(
                        valid_embeddings, valid_embeddings, transpose_b=True)
        self.saver_embed = tf.train.Saver([embeddings])
        self.saver = tf.train.Saver(max_to_keep = 3)
        
    def train(self):
        '''
        Train the embedding_matrix weight matrix, and you need to save the model
        '''
        if not os.path.exists("embedding_model"):
            os,mkdir("embedding_model/")
        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)
        self.sess.run(tf.global_variables_initializer())
        #var = tf.global_variables()
        #variable_to_restore = [val for val in var if 'embedding_matrix' in val.name]
        
        #self.saver_embed.restore(self.sess,tf.train.latest_checkpoint("embedding_model\\"))
        
        '''
        if os.path.exists("embedding_model\\"):
            self.saver.restore(self.sess,tf.train.latest_checkpoint("embedding_model"))
        '''
        i = 1
        best_score = np.inf
        while True:
            inputs,labels = self.generate_batch()
            #print(inputs)
            _,sim = self.sess.run([self.optimizer,self.similarity],feed_dict = {self.train_inputs:inputs,self.train_labels:labels})
            
            i += 1
            if i % 5000 == 0:
                print('#batch:',i)
                sim = sim
                sim_sum = 0
                print(np.shape(sim))
                #Calculate the average cosine distance
                for n in range(43):
                    for x in sim[n]:
                        sim_sum += abs(x)
                #sim_sum += sim[41,-1]
                sim_mean = (sim_sum-43)/(43*42)
                print(sim_mean)

                if sim_mean < best_score:
                    self.saver.save(self.sess,"embedding_model/",global_step = i)
                    best_score = sim_mean

                


def main():
    seed = 97
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    e = Embedded()
    e.train()
if __name__ == '__main__':
    main()