#-*- encoding:utf-8 -*-
'''
参数文件
'''
class Hparams(object):
    '''
    Model storage address and translation dictionary storage address
    '''
    phn2ind_path = 'data/phone39index'
    ind2phn_path = 'data/index39phone'
    save_path = 'model/'
    mode = 'TRAIN'
    '''
    Model parameters
    '''
    feature_dim = 120
    num_layer_encoder = 2
    num_layer_decoder = 2
    embedding_dim = 32
    rnn_size_encoder = 512
    rnn_size_decoder = 512
    keep_probability_e = 1.0

    '''
    Learning rate
    '''
    learning_rate = 1e-3
    learning_rate_decay = 0.95
    learning_rate_decay_steps = 500

    '''
    Training settings
    '''
    batch_size = 5
    beam_width = 10
    epochs = 50
    eos = "<EOS>"
    sos = "<SOS>"
    pad = "<PAD>"
    clip = 4


    '''
    Data address and summary address
    '''
    dataset_path = 'data/train_paths'
    dataset_root = ''
    valid_path = 'data/valid_paths'
    test_path = 'data/test_paths'
    summary_dir = 'summary/'

    '''
    
    '''
    num_gpu_device = 1
    embedded_batch_size = 32

    embedding_model = 'embedding_model/'
    num_skips = 2
    skip_window = 1
    embedding_voc_path = "data/embedding_voc"