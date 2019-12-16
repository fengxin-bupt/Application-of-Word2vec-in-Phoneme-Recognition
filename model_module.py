#-*- encoding:utf-8 -*-
#搭建模型所需要的一些模块
import tensorflow as tf
import os
import h5py
import numpy as np
def encoder_lstm_layer(layer_inputs,sequence_lengths,num_units,initial_state,is_training = 'True',direction = 'bidirectional'):
    '''
    Build a layer of lstm
    '''
    '''
    参数解释：
            layer_inputs:[batch_size,T,?],Input of this layer
            sequence_lengths:[batch_size],Entered true length
            num_units:Number of hidden nodes in each layer of lstm
            initial_state:Initial state
            direction:Lstm direction of each layer
    '''
    #The set environment is under the GPU, so CudnnLSTM is used
    ##cudnnlstm defaults to time-major, so you need to transform the entire input dimension
    inputs = tf.transpose(layer_inputs,[1,0,2])
    if direction == 'bidirectional':
        cudnnlstm_layer = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers = 1,num_units = num_units//2,direction = direction,kernel_initializer = tf.keras.initializers.Orthogonal())
    else:
        cudnnlstm_layer = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers = 1,num_units = num_units,direction = direction,kernel_initializer = tf.keras.initializers.Orthogonal())
    outputs,state = cudnnlstm_layer(inputs,initial_state = initial_state,sequence_lengths = sequence_lengths)

    #Transform the dimensions of the generated outputs
    outputs = tf.transpose(outputs,[1,0,2])

    return outputs,state

def reshape_pyramidal(inputs,sequence_lengths):
    '''
    Overlay the input frames so that the entire sequence can be downsampled
    参数解释：
            inputs:[batch_size,T,?]Feature input
            sequence_lengths:[batch_size]The true length of this feature sequence
    This is proposed in the "Listen Attend and Spell" paper
    '''
    shape = tf.shape(inputs)
    #Draw three dimensions
    batch_size,max_time = shape[0], shape[1]
    inputs_dim = inputs.get_shape().as_list()[-1]

    #The length of the sequence to be prevented is odd and cannot be divisible, so zero-padded operation is required
    pads = [[0,0],[0,tf.floormod(max_time,2)],[0,0]]
    padded_inputs = tf.pad(inputs, pads)

    #Reshape for frame overlay
    concat_outputs = tf.reshape(padded_inputs, (batch_size, -1, inputs_dim * 2))
    concat_sequence_lengths = tf.floordiv(sequence_lengths, 2) + tf.floormod(sequence_lengths, 2)
    return concat_outputs, concat_sequence_lengths

def rnn_cell(num_units):
    '''
    Build a single layer of lstm
    num_units:Number of hidden nodes in this layer of lstm
    '''
    cell = tf.nn.rnn_cell.LSTMCell(num_units)
    #cell = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers = 1,num_units = num_units,direction = "unidirectional")
    return cell

def attention_cell(decoder_cell,num_units,encoder_outputs,encoder_sequence_lengths):
    '''
    Encapsulate decoder_cell using high-level api in tensorflow
    参数解释：
            decoder_cell：Set up the lstm layer in the decoder
            num_units:Dimensions when calculating attention
            encoder_outputs:Encoding sequence from the encoder
            encoder_sequence_lengths:The actual length of the encoding sequence from the encoder
    '''
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units = num_units,
                                                               memory = encoder_outputs,
                                                               memory_sequence_length = encoder_sequence_lengths,
                                                               name = 'BahdanauAttention')
    #attention_layer_size：After generating the context and concat with the hidden state, it enters a fully connected layer with attention_layer_size
    return tf.contrib.seq2seq.AttentionWrapper(cell = decoder_cell,
                                                         alignment_history = True,
                                                         attention_mechanism = attention_mechanism,
                                                         attention_layer_size = 256,
                                                         output_attention = False)

    

def compute_loss(logits,labels,labels_lengths,max_time):
    '''
    Loss calculation during training
    参数解释:
            logits：[batch_size,T,num_classes]
            labels:[batch_size,T]
            max_time:maximum true length in label
    Sequence_loss is used, so you need to set the mask according to the given sequence length
    '''
    #First set a mask
    with tf.variable_scope('Loss'):
        target_weights = tf.sequence_mask(lengths = labels_lengths,
                                      maxlen = max_time,
                                      dtype = tf.float32,
                                      name = "loss_mask")
        loss = tf.contrib.seq2seq.sequence_loss(logits = logits,
                                            targets = labels,
                                            weights = target_weights,
                                            average_across_timesteps = True,
                                            average_across_batch = True)

    return loss
def dense_to_sparse(tensor,eos_id):
    '''
    Convert tensor to a specific sparse format
    Because when calculating tf.edit_distance, only sparse tensor is received
    '''
    added_values = tf.cast(tf.fill((tf.shape(tensor)[0],1), eos_id), tensor.dtype)

    #Add eos to the entire tensor
    concat_tensor = tf.concat((tensor, added_values), axis = -1)
    #Find duplicate phonemes
    diff = tf.cast(concat_tensor[:,1:] - concat_tensor[:,:-1], tf.bool)
    eos_indices = tf.where(tf.equal(concat_tensor, eos_id))
    #Find the position of the first eos in each decoded sequence
    first_eos = tf.segment_min(eos_indices[:,1],eos_indices[:, 0])
    #
    mask = tf.sequence_mask(first_eos,maxlen = tf.shape(tensor)[1])
    indices = tf.where(diff & mask & tf.not_equal(tensor, -1))
    values = tf.gather_nd(tensor, indices)
    shape = tf.shape(tensor, out_type = tf.int64)

    return tf.SparseTensor(indices, values, shape)

def compute_ler(logits,labels,eos_id):
    '''
    During training, calculate the label error rate for each batch
    参数解释：
            logits:[batch_size,T,num_classes]
            labels:[batch_size,T]
    '''
    with tf.variable_scope('Ler'):
        predicted_ids = tf.to_int32(tf.arg_max(logits,-1))
    

        hypothesis = dense_to_sparse(predicted_ids,eos_id)
        truth = dense_to_sparse(labels,eos_id)

        label_error_rate = tf.edit_distance(hypothesis, truth, normalize = True)
    return label_error_rate

def feature_extract(path):
    '''
    从文件中提取特征，一般是从h5py文件中提取，key = 'feature'
    '''
    audio_filename = os.path.abspath(path)
    feature_filename = audio_filename.split('.')[0] + '.feat'
    f = h5py.File(feature_filename, 'r')
    data = f.get('feature')
    data = np.array(data)
    f.close()
    return data
def label_extract(path):
    '''
    从文件中提取label，一般是从h5py文件中提取，key = 'label'
    '''
    audio_filename = os.path.abspath(path)
    label_filename = audio_filename.split('.')[0] + '.label'
    f = h5py.File(label_filename,'r')
    label = f.get('label')
    label = np.array(label)
    f.close()
    return label
