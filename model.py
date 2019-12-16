#-*- encoding:utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense
import random
import model_module
import time
import h5py
import pickle
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#from tensorflow.python.data.experimental import AUTOTUNE
class Attention_sr_model:
    def __init__(self,
                 phn2ind,
                 ind2phn,
                 save_path,
                 mode,
                 feature_dim,
                 num_layer_encoder,
                 num_layer_decoder,
                 embedding_dim,
                 rnn_size_encoder,
                 rnn_size_decoder,
                 learning_rate = 1e-4,
                 learning_rate_decay = 0.98,
                 learning_rate_decay_steps = 1000,
                 keep_probability_e = 1.0,
                 batch_size = 32,
                 beam_width = 10,
                 epochs = 100,
                 eos = "<EOS>",
                 sos = "<SOS>",
                 pad = "<PAD>",
                 clip = 3,
                 embedding_model = None,
                 dataset_path = None,
                 test_paths = None,
                 summary_dir = None,
                 valid_path = None,
                 dataset_root = None,
                 num_gpu_device = 2,
                 is_training = True,
                 is_testing = False):
        '''
        参数解释：
                phn2ind：Phoneme to index conversion dictionary
                ind2phn：Index to phoneme conversion dictionary
                save_path：tensorflow生成的模型存储位置
                mode：
                                'TRAIN'
                                'INFER'
                feature_dim：Feature dimension
                num_layer_encoder：Number of layers of rnn in the encoder
                num_layer_decoder：Number of layers of rnn in the decoder
                rnn_size_encoder：Number of hidden nodes of lstm / gru in the encoder
                rnn_size_decoder：Number of hidden nodes of lstm / gru in the dencoder
                embedding_dim：number of dimensions of embedding_matrix, each phoneme will be mapped to such a long vector
                learning_rate：Initial learning rate
                learning_rate_decay：The decay rate of the learning rate is generally used in conjunction with learning_rate_decay_steps
                learning_rate_decay_steps：Attenuation steps for learning rate
                keep_probability_e：embedding retention
                batch_size：Size of each mini-batch
                beam_width：Beam branch shear width, used when mode = 'INFER'
                epochs：Number of training iterations for the entire data set
                eos：end of sentence
                sos：start of sentence
                pad：做的padding
                clip：To prevent gradient explosions, you can choose to use clipping
                summary_dir：Put the stored summary into that address
        '''
        #Initialize all parameters
        #Detect the number of GPUs in the system hardware
        self.num_gpu_device = num_gpu_device
        self.phn2ind = phn2ind
        self.ind2phn = ind2phn
        self.num_classes = len(phn2ind)
        self.feature_dim = feature_dim
        self.num_layer_encoder = num_layer_encoder
        self.num_layer_decoder = num_layer_decoder
        self.rnn_size_encoder = rnn_size_encoder
        self.rnn_size_decoder = rnn_size_decoder
        self.save_path = save_path
        self.embedding_dim = embedding_dim
        self.mode = mode.upper()
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_decay_steps = learning_rate_decay_steps
        self.keep_probability_e = keep_probability_e
        self.sampling_probability = 0.1
        self.batch_size = batch_size
        self.beam_width = beam_width
        self.eos = eos
        self.sos = sos
        self.pad = pad
        self.clip = clip
        self.epochs = epochs
        self.embedding_model = embedding_model
        self.summary_dir = summary_dir
        self.dataset_path = dataset_path
        self.test_paths = test_paths
        self.valid_path = valid_path
        self.dataset_root = dataset_root
        self.is_training = is_training
        self.is_testing = is_testing
    def build_graph(self):
        self.add_embeddings()
        if not self.is_testing:

            self.batch_data_extract()
            self.phone_embedded_vector = self.add_lookup_ops(self.train_phones)
            
        else:
            self.test_data_extract()
        self.build_model()
        self.saver = tf.train.Saver(max_to_keep = 5)
    def dataset_path_pickle(self,pickle_path):
        '''
        Extract all stored addresses from the pickle file
        '''
        paths = []
        with open(pickle_path,'rb') as f:
            path_list = pickle.load(f)
            for path in path_list:
                path = self.dataset_root + path
                path = path.replace('\\','/')
                paths.append(path)
        return paths
    def read_feature(self,item):
        '''
        Read feature
        '''
        f = h5py.File(item.decode(),'r')
        feature = np.array(f.get('feature'))
        if self.is_testing:
            pad_vector = np.zeros((20,np.shape(feature)[-1]))
            feature = np.vstack((pad_vector,feature,pad_vector))

         
        f.close()

        return feature.astype(np.float32)
    def read_label(self,item):
        '''
        read label
        '''
        f = h5py.File(item.decode(),'r')
        label = np.array(f.get('label'))
        f.close()
        label = label[1:]
        return label.astype(np.int32)
    def read_train_label(self,item):
        '''
        read label
        '''
        f = h5py.File(item.decode(),'r')
        label = np.array(f.get('label'))
        f.close()
        label = label[:-1]
        return label.astype(np.int32)
    
    def extract_data_from_path(self,path):
        '''
        Read each batch of data. Because the stored data is in h5py format, you need to support the tf.py_func function.
        When reading data at the same time, pay attention to whether it is in the test stage.
        '''
        audio_filename = path
        feature_filename = tf.string_join([tf.string_split([audio_filename],".").values[0],'.40logfbank'])
        #Perform feature reading
        #Read features in h5py files, while eliminating redundant dimensions
        audio_feature = tf.squeeze(tf.py_func(self.read_feature,[feature_filename],[tf.float32]))
        #Convert the read feature into a tensor, and the length of the feature should be recorded
        audio_feature = tf.convert_to_tensor(audio_feature)
        audio_length = tf.shape(audio_feature)[0]
        if self.is_testing:
            return {'audios':audio_feature,'audio_lengths':audio_length}
        else:
            #Read the target
            label_filename = tf.string_join([tf.string_split([audio_filename],".").values[0],'.label'])
            #Read the label in the h5py file, and you need to eliminate the extra dimensions
            target_label = tf.squeeze(tf.py_func(self.read_label,[label_filename],[tf.int32]))
            train_label = tf.squeeze(tf.py_func(self.read_train_label,[label_filename],[tf.int32]))

            #Convert the read label to a tensor and record the length of the label
            target_label = tf.convert_to_tensor(target_label)
            target_length = tf.shape(target_label)[0]
            train_label = tf.convert_to_tensor(train_label)

            return {'audios':audio_feature,'audio_lengths':audio_length,'train_label':train_label,'target_label':target_label,'target_length':target_length}
    def batch_data_extract(self):
        '''
        Settings for all inputs
        audio_features:The input voice features of a batch
        audio_feature_lengths:The length of a batch voice feature input
        target_phones:Target phoneme sequence input during training phase
        target_phone_lengths:Sequence length of the target factor input during the training phase
        maximum_decode_iter_num:Maximum number of decoding steps during decoding
        '''
        with tf.variable_scope('Dataset'):
            self.train_paths = tf.placeholder(tf.string)
        
            self.valid_paths = tf.placeholder(tf.string)

            self.train_valid = tf.placeholder(tf.bool)
        
            #这是训练集的读取
            dataset = tf.data.Dataset.from_tensor_slices(self.train_paths)\
                                    .map(self.extract_data_from_path,num_parallel_calls = 3)\
                                    .prefetch(buffer_size = 2 * self.batch_size)\
                                    .padded_batch(batch_size = self.batch_size,padded_shapes = {'audios':[None,self.feature_dim],'audio_lengths':[],
                                                                                                'train_label':[None],'target_label':[None],'target_length':[]},drop_remainder = True)
            #这是验证集的读取
            valid_dataset = tf.data.Dataset.from_tensor_slices(self.valid_paths)\
                                    .map(self.extract_data_from_path,num_parallel_calls = 1)\
                                    .prefetch(buffer_size = 2 * self.batch_size)\
                                    .padded_batch(batch_size = self.batch_size,padded_shapes = {'audios':[None,self.feature_dim],'audio_lengths':[],
                                                                                                'train_label':[None],'target_label':[None],'target_length':[]},drop_remainder = True)

            #Use dataset to read data while prefetch
            self.train_iterator = dataset.make_initializable_iterator()
            self.valid_iterator = valid_dataset.make_initializable_iterator()
            #Get a batch of data
            ##Determine whether to get train or valid

            batch_data = tf.cond(self.train_valid, lambda:self.train_iterator.get_next(), lambda:self.valid_iterator.get_next())


            self.audio_features = batch_data['audios']

            self.audio_feature_lengths = batch_data['audio_lengths']

            self.target_phones = batch_data['target_label']

            self.train_phones = batch_data['train_label']

            self.target_phone_lengths = batch_data['target_length']

            self.maximum_decode_iter_num = tf.reduce_max(self.target_phone_lengths,
                                                    name = 'max_dec_len')
    def test_data_extract(self):
        '''
        During testing：
        Settings for all inputs
        audio_features:The input voice features of a batch
        audio_feature_lengths:The length of a batch voice feature input
        '''
        #Determine whether to give test paths
        if not self.test_paths:
            print("there is no test path!\n")
            print("please input the paths")
        test_dataset = tf.data.Dataset.from_tensor_slices(self.test_paths)\
                                      .map(self.extract_data_from_path,num_parallel_calls = 4)\
                                      .prefetch(buffer_size = 2 * self.batch_size)\
                                      .padded_batch(batch_size = self.batch_size,padded_shapes = {'audios':[None,self.feature_dim],'audio_lengths':[]})
        self.test_iterator = test_dataset.make_initializable_iterator()
        batch_data = self.test_iterator.get_next()

        self.audio_features = batch_data['audios']
        self.audio_feature_lengths = batch_data['audio_lengths']

    def add_embeddings(self):
        '''
        Map the input phoneme index to a fixed-length vector
        Generate a transformation vector
        '''
        
        self.embedding_matrix = tf.get_variable('embedding_matrix',shape = [self.num_classes,self.embedding_dim],dtype = tf.float32,trainable = True)
    def add_lookup_ops(self,phone_indexs):
        '''
        Convert a given phoneme to a vector
        '''
        with tf.variable_scope('Lookup'):
            phone_embedding = tf.nn.embedding_lookup(self.embedding_matrix,
                                                self.train_phones,
                                                name = 'phone_embedding')
            phone_embedded_vector = tf.nn.dropout(phone_embedding,
                                              self.keep_probability_e,
                                              name = 'phone_embedded_vector')
            return phone_embedded_vector

    def build_encoder(self,encoder_inputs,encoder_lengths):
        '''
        Build the encoder
        Parameter：
                encoder_inputs:[batch_size,T,?]
                encoder_lengths:[batch_size]
        '''
        with tf.variable_scope('Encoder'):
            inputs = encoder_inputs
            sequence_lengths = encoder_lengths
            #Set the initial state of lstm
            state = None
            for n in range(self.num_layer_encoder):
                with tf.variable_scope('encoder_layer_'+str(n+1)) as scope:
                    lstm_output, state = model_module.encoder_lstm_layer(inputs,
                                                                        sequence_lengths,
                                                                        self.rnn_size_encoder,
                                                                        state,
                                                                        self.is_training)
                    #Process the output of the network layer further
                    (forward_state,backward_state) = state
                    inputs = lstm_output
            '''    
            with tf.variable_scope('pyramidal_layer_'+str(1)):
                    inputs,sequence_lengths = model_module.reshape_pyramidal(lstm_output,sequence_lengths)
                    
                    sequence_lengths = sequence_lengths
            '''
                    
                    
            
            #Encapsulate the lstm state of the last layer
            #The encapsulated lstm state can be used in the subsequent decoder
            forward_hidden_state,backward_hidden_state = forward_state[0],backward_state[0]
            forward_cell_state,backward_cell_state = forward_state[1],backward_state[1]
            encoder_hidden_state = tf.concat((forward_hidden_state,backward_hidden_state), axis = -1)
            encoder_cell_state = tf.concat((forward_cell_state,backward_cell_state), axis = -1)
            encoder_state = tf.nn.rnn_cell.LSTMStateTuple(c = encoder_cell_state, h = encoder_hidden_state)

            return inputs, sequence_lengths, encoder_state

    def build_decoder_cell(self,encoder_outputs,encoder_state,encoder_sequence_lengths):
        '''
        Building a lstm unit in a decoder
        '''
        memory = encoder_outputs

        #If it is in the speculative stage, you need to tile each state and variable
        if self.mode == 'INFER' and self.beam_width > 0:
            memory = tf.contrib.seq2seq.tile_batch(memory,
                                        multiplier = self.beam_width)
            encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state,
                                        multiplier = self.beam_width)
            encoder_sequence_lengths = tf.contrib.seq2seq.tile_batch(encoder_sequence_lengths,
                                        multiplier = self.beam_width)
            batch_size = self.batch_size * self.beam_width
        else:
            batch_size = self.batch_size

        #Build lstm
        with tf.variable_scope('decoder_lstm_layer'):
            if self.num_layer_decoder is not None:
                lstm_cell = tf.nn.rnn_cell.MultiRNNCell(
                                                [model_module.rnn_cell(self.rnn_size_decoder) for _ in range(self.num_layer_decoder)])

            else:
                lstm_cell = model_module.rnn_cell(self.rnn_size_decoder)

            cell = model_module.attention_cell(lstm_cell,
                                           self.rnn_size_decoder,
                                           memory,
                                           encoder_sequence_lengths)
            #Set the initial state
            decoder_state = tuple([encoder_state] * self.num_layer_decoder)
            decoder_initial_state = cell.zero_state(batch_size, tf.float32).clone(cell_state = decoder_state)

        return cell, decoder_initial_state
    def build_decoder(self,encoder_outputs,encoder_state,encoder_sequence_lengths,decoder_inputs_lengths = None,decoder_inputs = None):
        '''
        Build the decoder
        '''
        '''
        参数解释：
                encoder_outputs:[batch_size,T,?]Encoding sequence from the encoder
                encoder_state:[batch_size,?]The last output state of the encoder, you can use this to initialize the initial state of the decoder
                encoder_sequence_lengths:[batch_size]The true and valid length in the last encoded sequence of the encoder
                decoder_inputs_lengths:[batch_size]In the training phase, the true length of the sequence in the input decoder
                decoder_inputs:[batch_size,T,embedding_dim]During the training phase. Input decoder's embedded_vector
        '''
        with tf.variable_scope('Decoder') as decode_scope:
            sos_id_2 = tf.cast(self.phn2ind[self.sos],tf.int32)
            eos_id_2 = tf.cast(self.phn2ind[self.eos],tf.int32)
            with tf.variable_scope('Output_layer'):
                self.output_layer = Dense(self.num_classes,name = 'output_layer')

            with tf.variable_scope('decoder_attention_wrapper'):
                decoder_cell,decoder_initial_state = self.build_decoder_cell(encoder_outputs,
                                                                        encoder_state,
                                                                        encoder_sequence_lengths)

            #During Training
            if self.mode != 'INFER':
                with tf.variable_scope('Helper'):

                    train_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
                                inputs = decoder_inputs,
                                sequence_length = decoder_inputs_lengths,
                                embedding = self.embedding_matrix,
                                sampling_probability = self.sampling_probability,
                                time_major = False)

                    train_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                                train_helper,
                                                                decoder_initial_state,
                                                                output_layer = self.output_layer)

                    #Dynamic decoding
                    outputs, final_context_state,_ = tf.contrib.seq2seq.dynamic_decode(
                                                    train_decoder,
                                                    output_time_major = False,
                                                    maximum_iterations = self.maximum_decode_iter_num,
                                                    swap_memory = False,
                                                    impute_finished = True,
                                                    scope = decode_scope)
                    sample_id = outputs.sample_id
                    logits = outputs.rnn_output
            #Prediction stage
            else:
                start_tokens = tf.fill([self.batch_size],sos_id_2)
                end_token = eos_id_2

                #If you do beam_search
                if self.beam_width > 0:
                    my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                        cell = decoder_cell,
                        embedding = self.embedding_matrix,
                        start_tokens = start_tokens,
                        end_token = end_token,
                        initial_state = decoder_initial_state,
                        beam_width = self.beam_width,
                        output_layer = self.output_layer)
                #When beam_search is zero, choose to use greedy
                else:
                    greedy_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embedding_matrix,
                                                                             start_tokens,
                                                                             end_token)
                    my_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                                 greedy_helper,
                                                                 decoder_initial_state,
                                                                 output_layer = self.output_layer)
                #Dynamic decoding
                outputs, final_context_state,_ = tf.contrib.seq2seq.dynamic_decode(
                                                                    my_decoder,
                                                                    maximum_iterations = 300,
                                                                    output_time_major = False,
                                                                    impute_finished = False,
                                                                    swap_memory = False,
                                                                    scope = decode_scope)
                if self.beam_width > 0:
                    logits = tf.no_op()
                    sample_id = outputs.predicted_ids
                else:
                    logits = tf.no_op()
                    sample_id = outputs.sample_id
        return logits, sample_id, final_context_state,outputs

    def average_gradients(self,tower_grads):
        '''
        Calculate the average gradient
        '''
        average_grads = []
        for grad_and_vars in zip(*tower_grads):

            grads = []
            for g, z in grad_and_vars:
                expanded_g = tf.expand_dims(g,0)

                grads.append(expanded_g)
            grad = tf.concat(grads,axis = 0)
            grad = tf.reduce_mean(grad,0)

            v = grad_and_vars[0][1]
            grad_and_var = (grad,v)
            average_grads.append(grad_and_var)
        return average_grads

    def build_model(self):
        '''
        Build the model
        '''
        with tf.variable_scope('optimizer'):
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate = self.learning_rate)
        with tf.variable_scope('model',dtype = tf.float32):
            if self.mode == 'TRAIN':
                self.tower_grads = []
                self.tower_loss = []
                self.tower_ler = []
                for i in range(self.num_gpu_device):
                    with tf.device("/gpu:{}".format(i)):
                        with tf.variable_scope(tf.get_variable_scope(),reuse = tf.AUTO_REUSE):
                            #If it is multi-gpu, you need to split the input data
                            audio_inputs = tf.split(value = self.audio_features,
                                                    num_or_size_splits = self.num_gpu_device,
                                                    axis = 0)[i]

                            audio_sequence_lengths = tf.split(value = self.audio_feature_lengths,
                                                              num_or_size_splits = self.num_gpu_device,
                                                              axis = 0)[i]

                            phone_embedded_inputs = tf.split(value = self.phone_embedded_vector,
                                                             num_or_size_splits = self.num_gpu_device,
                                                             axis = 0)[i]

                            phone_sequence_lengths = tf.split(value = self.target_phone_lengths,
                                                              num_or_size_splits = self.num_gpu_device,
                                                              axis = 0)[i]
                            target_phones_gpu = tf.split(value = self.target_phones,
                                                         num_or_size_splits = self.num_gpu_device,
                                                         axis = 0)[i]

                            train_phones_gpu = tf.split(value = self.train_phones,
                                                         num_or_size_splits = self.num_gpu_device,
                                                         axis = 0)[i]

                            #Put data into the encoder
                            encoder_outputs,encoder_sequence_lengths,encoder_state = self.build_encoder(encoder_inputs = audio_inputs,
                                                                                encoder_lengths = audio_sequence_lengths)

                            #Put the output of the encoder into the decoder
                            logits, sample_ids, final_context_state,decoder_output = self.build_decoder(encoder_outputs = encoder_outputs,
                                                                                         encoder_state = encoder_state,
                                                                                         encoder_sequence_lengths = audio_sequence_lengths,
                                                                                         decoder_inputs_lengths = phone_sequence_lengths,
                                                                                         decoder_inputs = phone_embedded_inputs)
                            #Perform zero padding on logits
                            logits = tf.pad(logits,[[0,0],[0,(self.maximum_decode_iter_num - tf.shape(logits)[1])],[0,0]])
                            loss = model_module.compute_loss(logits,target_phones_gpu,phone_sequence_lengths,self.maximum_decode_iter_num)
                            ler = model_module.compute_ler(logits,target_phones_gpu,self.phn2ind[self.eos])
                            grads = self.optimizer.compute_gradients(loss)
                            self.tower_grads.append(grads)
                            self.tower_loss.append(tf.reduce_mean(loss))
                            self.tower_ler.append(tf.reduce_mean(ler))
                            
                with tf.variable_scope('summary_op'):
                
                    self.train_loss = tf.reduce_mean(self.tower_loss)
                    #记录下loss
                    self.training_summary = tf.summary.scalar('loss',self.train_loss)
                
                    
                    self.train_ler = tf.reduce_mean(self.tower_ler)

                    self.ler_summary = tf.summary.scalar('ler',self.train_ler)
                    #设立总的global_step
                    self.global_step = tf.Variable(0, trainable = False)

                    self.summary_op = tf.summary.merge_all()

                #optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.8, beta2=0.99,epsilon = 1e-05)
                #optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
                #optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
                
                self.learning_rate = tf.train.exponential_decay(learning_rate = self.learning_rate,global_step = self.global_step,
                                                        decay_steps = self.learning_rate_decay_steps,decay_rate = self.learning_rate_decay)
                
                
                #Whether clipping is required
                with tf.variable_scope('Train_optimizer'):
                    grads = self.average_gradients(self.tower_grads)

                    apply_gradient_op = self.optimizer.apply_gradients(grads,global_step = self.global_step)

                    variable_averages = tf.train.ExponentialMovingAverage(0.99,self.global_step)

                    variable_averages_op = variable_averages.apply(tf.trainable_variables())

                    self.train_op = tf.group(apply_gradient_op,variable_averages_op)
                '''
                with tf.variable_scope('train_optimizer'):
                    if self.clip > 0:
                        grads,vs = zip(*optimizer.compute_gradients(self.train_loss))
                        #为了防止梯度爆炸，做梯度修正
                        grads, _ = tf.clip_by_global_norm(grads, self.clip)

                        self.train_op = optimizer.apply_gradients(zip(grads, vs),
                                                                global_step = self.global_step)
                    else:
                        #直接最小化loss
                        self.train_op = optimizer.minimize(self.train_loss,
                                                            global_step = self.global_step)
                '''
            elif self.mode == 'INFER':
                #encoder
                encoder_outputs,encoder_sequence_lengths,encoder_state = self.build_encoder(encoder_inputs = self.audio_features,
                                                                                encoder_lengths = self.audio_feature_lengths)

                #Put the output of the encoder into the decoder
                #In the speculation phase, there are no inputs to the decoder
                logits, sample_ids, final_context_state,outputs = self.build_decoder(encoder_outputs = encoder_outputs,
                                                                            encoder_state = encoder_state,
                                                                            encoder_sequence_lengths = encoder_sequence_lengths)
                loss = None

                self.infer_logits, _, self.final_context_state, self.sample_id,self.outputs = logits, loss, final_context_state, sample_ids,outputs

    def train(self,
              restore_path = None):
        '''
        Operations for training，
        
        restore_path:If this option is turned on, training is based on existing models

        '''
        self.best_loss = np.inf

        #Whether to store as summary
        if self.summary_dir is not None:
            #Set up two writers
            self.add_summary()
        #Then start the session
        self.initialize_session()


        #If restore_path exists, restore model is required
        
        if restore_path is not None:
            self.restore_session(restore_path)
        if self.embedding_model is not None:
            self.restore_embedding()
        audio_paths = self.dataset_path_pickle(self.dataset_path)
        valid_paths = self.dataset_path_pickle(self.valid_path)

        self.train_audio_paths = audio_paths
        self.valid_audio_paths = valid_paths
        random.shuffle(self.train_audio_paths)
        random.shuffle(self.valid_audio_paths)
        #Start iterative training
        #Initialize the data iterator
        self.sess.run(self.train_iterator.initializer,feed_dict = {self.train_paths:self.train_audio_paths,
                                                                    self.valid_paths:self.valid_audio_paths,
                                                                    self.train_valid:True})
        epoch = 1
        epoch_time_start = time.time()
        self.epoch_loss = []
        self.epoch_ler = []
        print('-----------------------------Epoch {} of {} ----------------------------'.format(epoch,self.epochs))
        self.batch_start_time = time.time()
        
        i = 0
        step = 0
        while True:
            try:
                if epoch > self.epochs:
                    break
                i += 1
                step += 1
                self.is_training = True
                #每个batch的运行
                loss,ler,summary_op,_ = self.sess.run([self.train_loss,self.train_ler,self.summary_op,self.train_op],feed_dict = {self.train_valid:True})
                
                self.epoch_loss.append(loss)
                self.epoch_ler.append(ler)

                if i > 0 and (i % 200 == 0):
                    batch50_end_time = time.time()
                    self.train_writer.add_summary(summary_op, step)
                    during_time = batch50_end_time - self.batch_start_time
                    self.batch_start_time = batch50_end_time
                    print('Epoch: {}, Batch: {}, train_loss: {:.4f}, train_ler: {:.4f}, during time: {:.2f}s'.format(epoch,i,loss,ler,during_time))
                if i > 0 and (i % 800 == 0):
                    #For every 200 batches, valid once
                    self.is_training = False
                    self.sampling_probability = 1.0
                    random.shuffle(self.valid_audio_paths)
                    self.sess.run(self.valid_iterator.initializer,feed_dict = {self.valid_paths :self.valid_audio_paths,
                                                                                self.train_valid:False})
                    valid_loss = []
                    valid_ler = []
                    for z in range(10):
                        loss,ler,summary_op = self.sess.run([self.train_loss,self.train_ler,self.summary_op],feed_dict = {self.train_valid:False})
                        valid_loss.append(loss)
                        valid_ler.append(ler)
                    self.eval_writer.add_summary(summary_op,step)
                    average_valid_loss = self.sess.run(tf.reduce_mean(valid_loss))
                    average_valid_ler = self.sess.run(tf.reduce_mean(valid_ler))
                    valid_end_time = time.time()
                    valid_during_time = valid_end_time - self.batch_start_time
                    self.sampling_probability = 0.75

                    print('validation loss: {:.4f}, validation ler: {:.4f}, validation time: {:.2f}s'.format(average_valid_loss,average_valid_ler,valid_during_time))
            except tf.errors.OutOfRangeError:
                epoch_ave_loss = self.sess.run(tf.reduce_mean(self.epoch_loss))
                epoch_ave_ler = self.sess.run(tf.reduce_mean(self.epoch_ler))
                epoch_time_end = time.time()
                epoch_during_time = epoch_time_end - epoch_time_start
                print('the epoch {}, average loss: {:.4f}, average_ler: {:.4f}, during time: {:.2f}s'.format(str(epoch),epoch_ave_loss,epoch_ave_ler,epoch_during_time))
        
            
                #Determine if the loss is the best to determine the storage model
                if epoch_ave_loss <= self.best_loss:
                    if not os.path.exists(self.save_path):
                        os.makedirs(self.save_path)
                    self.saver.save(self.sess,self.save_path,global_step = epoch)
                    self.best_loss = epoch_ave_loss

                    print('\n---------------------new best loss: {:.4f} --------------------\n'.format(self.best_loss))
                    print('---------------------save the best model at: {}--------------------'.format(self.save_path))
                epoch += 1
                epoch_time_start = time.time()
                self.epoch_loss = []
                self.epoch_ler = []
                print('-----------------------------Epoch {} of {} ----------------------------'.format(epoch,self.epochs))
                self.batch_start_time = time.time()
                random.shuffle(self.train_audio_paths)
                self.sess.run(self.train_iterator.initializer,feed_dict = {self.train_paths :self.train_audio_paths})
                i = 0
    def infer(self,restore_path):
        '''
        Load the model and predict the data, usually the batchsize here is set to 1
        restore_path:Is the address of the folder where the model is stored
        '''
        if restore_path == None:
            print('please input a restore path')
        #Open the session and load the model, and pay attention to the setting of some parameters
        self.initialize_session()
        if restore_path is not None:
            self.restore_session(restore_path)
        else:
            print('there is no saved model in ',restore_path)

        #Initialize the iterator
        self.sess.run(self.test_iterator.initializer)
        prediction_ids = []
        for i in range(len(self.test_paths)):
            try:
                print(self.test_paths[i])
                #logits = self.sess.run(self.outputs)[0]
                #logits = self.sess.run(tf.nn.softmax(logits))
                #print(np.shape(logits))
                pred_ids = self.sess.run(self.sample_id)
                for pred_id in pred_ids:
                    prediction_ids.append(pred_id)
                print('Have been inferred the path:',self.test_paths[i])
            except tf.errors.OutOfRangeError:
                #Prevent anomalies
                print('all test audios have been inferred !')
                return prediction_ids
        return prediction_ids


    def initialize_session(self):
        '''
        开启session
        '''
        #Global initialization
        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)
        self.sess.run(tf.global_variables_initializer())
    def restore_embedding(self):
        '''
        Extract the trained embedding matrix from the embedding model
        '''
        self.saver_embedding = tf.train.Saver([self.embedding_matrix])
        var = tf.global_variables()
        variable_to_restore = [val for val in var if 'embedding_matrix' in val.name]
        self.saver_embedding.restore(self.sess,tf.train.latest_checkpoint(self.embedding_model))
    def restore_session(self,restore_path):
        '''
        Extract the saved model from restore_path
        '''
        self.saver.restore(self.sess,tf.train.latest_checkpoint(restore_path))
        print('Done restoring from the path: ', restore_path)
    def add_summary(self):
        '''
        Set up two writers, then display the summary using tensorboard
        '''

        self.train_writer = tf.summary.FileWriter(self.summary_dir + '/train',
                                                    tf.get_default_graph())
        self.eval_writer = tf.summary.FileWriter(self.summary_dir + '/valid')
