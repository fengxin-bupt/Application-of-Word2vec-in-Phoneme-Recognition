#-*- encoding:utf-8 -*-
import tensorflow as tf
from model import Attention_sr_model
import numpy as np
import os
import pickle
from config import Hparams
def load_pickle(path):
    with open(path,'rb') as file:
        data = pickle.load(file)
    return data

phn2ind = load_pickle(Hparams.phn2ind_path)
ind2phn = load_pickle(Hparams.ind2phn_path)
seed = 97
tf.reset_default_graph()
tf.set_random_seed(seed)

#Where to store models
save_path = Hparams.save_path
if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(Hparams.summary_dir):
    os.makedirs(Hparams.summary_dir)
    os.makedirs(os.path.join(Hparams.summary_dir,'train'))
    os.makedirs(os.path.join(Hparams.summary_dir,'valid'))
#Pickle the storage address of all data in the dataset
dataset_path = Hparams.dataset_path
summary_dir = Hparams.summary_dir
dataset_root = Hparams.dataset_root
speech_recognition = Attention_sr_model(phn2ind = phn2ind,
                                        ind2phn = ind2phn,
                                        save_path = save_path,
                                        mode = Hparams.mode,
                                        feature_dim = Hparams.feature_dim,
                                        num_layer_encoder = Hparams.num_layer_encoder,
                                        num_layer_decoder = Hparams.num_layer_decoder,
                                        embedding_dim = Hparams.embedding_dim,
                                        rnn_size_encoder = Hparams.rnn_size_encoder,
                                        rnn_size_decoder = Hparams.rnn_size_decoder,
                                        learning_rate = Hparams.learning_rate,
                                        learning_rate_decay = Hparams.learning_rate_decay,
                                        learning_rate_decay_steps = Hparams.learning_rate_decay_steps,
                                        keep_probability_e = Hparams.keep_probability_e,
                                        batch_size = Hparams.batch_size,
                                        beam_width = Hparams.beam_width,
                                        epochs = Hparams.epochs,
                                        eos = Hparams.eos,
                                        sos = Hparams.sos,
                                        pad = Hparams.pad,
                                        clip = Hparams.clip,
                                        embedding_model = Hparams.embedding_model,
                                        dataset_path = Hparams.dataset_path,
                                        test_paths = None,
                                        summary_dir = Hparams.summary_dir,
                                        valid_path = Hparams.valid_path,
                                        dataset_root = Hparams.dataset_root,
                                        num_gpu_device = Hparams.num_gpu_device)
speech_recognition.build_graph()
speech_recognition.train()
