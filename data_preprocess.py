#-*- encoding:utf-8 -*-
import numpy as np
import pickle
import librosa
from pathlib import Path
import os
import re
import sys
import h5py
import python_speech_features
from python_speech_features import mfcc
import random
import shutil
def load_txt_path(dir_path):
    '''
    Processing of librispeech data
    dir_path：dataset storage address (Generally the root directory of the dataset)

    '''
    dir_path = Path(dir_path)
    txt_list = [f for f in dir_path.glob('**/*.txt') if f.is_file()]
    audio_list = [f for f in dir_path.glob('**/*.flac') if f.is_file()]
    print('Number of audio txt paths:',len(txt_list))
    print('Number of audio file paths:',len(audio_list))

    #Perform txts extraction and phoneme conversion,
    #save all file addresses that can convert phonemes
    
    txts = []
    audios = []
    audio_paths = []

    for i, txt in enumerate(txt_list):
        
        with open(txt) as f:
            for line in f.readlines():
                #Each line of txt records the name of the audio file and everything in the audio
                #print(line)
                for audio in audio_list:
                    #判断audios是否存在
                    if audio.stem in line:
                        print(audio)
                        line = re.sub(r'[^A-Za-z]',' ',line)
                        line = line.strip()
                        txts.append(line)
                        audio_paths.append(audio)
                        break
        print('Text#:',i+1,len(audio_paths))
    #All transcripts of the returned txts, all audio data storage addresses in the audio_paths dataset
    return txts,audio_paths
def convert_txt2phn(txts,audio_paths,dict_path):
    '''
    Convert the extracted char into a phoneme sequence
    Need to ensure that all words in the entire char can be converted to phonemes
    txts:All labeled sequences (word level) in the dataset
    audio_paths:All audio addresses in the dataset
    dict_path:Dictionary address, cmu's dictionary, already stored as pickle file
    '''
    #Extraction of conversion dictionary
    f = open(dict_path,'rb')
    convert_dict = pickle.load(f)
    f.close()
    assert len(txts) == len(audio_paths)
    converted_paths = []
    converted_labels = []
    for i in range(len(txts)):
        path = audio_paths[i]
        word_list = txts[i].split(' ')
        phones = []
        can_convert = 1
        #Determine if all words can be converted
        for word in word_list:
            if word.lower() in convert_dict:
                phone = convert_dict[word.lower()]
                for phn in phone:
                    phones.append(phn)
            else:
                #There is a word that cannot be transcribed in the current sentence. Stop transcription and discard the data.
                can_convert = 0
        if can_convert:
            converted_paths.append(audio_paths[i])
            converted_labels.append(phones)
    #converted_paths:Full transcribed audio address
    #converted_labels:Full transcribed phoneme address
    return converted_paths,converted_labels,convert_dict
def make_index_dict(label_list,convert_dict):
    '''
    For mapping between phoneme and index, you need to add <PAD>, <SOS>, <EOS>
    
    '''
    specials = ['<PAD>','<SOS>','<EOS>']
    phn2ind = {}
    ind2phn = {}
    index = 0
    if specials:
        for special in specials:
            phn2ind[special] = index
            ind2phn[index] = special
            index += 1
    for phones in label_list:
        for phn in phones:
            if phn not in phn2ind:
                phn2ind[phn] = index
                ind2phn[index] = phn
                index += 1
    for word in convert_dict.keys():
        phone_list = convert_dict[word]
        for phn in phone_list:
            if phn not in phn2ind:
                phn2ind[phn] = index
                ind2phn[index] = phn
                index += 1
    phn2ind['sil'] = index
    ind2phn[index] = 'sil'
    print('phn2ind:',len(phn2ind))
    print(phn2ind)
    return phn2ind,ind2phn

def phone2index(phn2ind,converted_labels):
    '''
    Convert the extracted phoneme into index, and add sil and <SOS>, <EOS> before and after the labels file
    '''
    index_labels = []
    for labels in converted_labels:
        index_label = []
        labels.insert(0,'sil')
        labels.append('sil')
        labels.insert(0,'<SOS>')
        labels.append('<EOS>')
        for label in labels:
            index_label.append(phn2ind[label])
        index_labels.append(index_label)
    return index_labels
def audio_file(feature_filename, audios):
    #Storage feature file
    #Hdf file for storage by default

    feat_file = h5py.File(feature_filename,'w')
    feat_file['feature'] = audios
    feat_file.close()
def label_file(label_filename,labels):
    #Storage label file
    #Hdf file for storage by default

    label_file = h5py.File(label_filename,'w')
    label_file['label'] = labels
    label_file.close()
def audioToInputVector(audio_filename, delta = None, delta_2 = None):
    """
    Extract the required features. The default is the extracted log-filterbank, level 40.
    Normalize the audio amplitude when reading audio, because the load in librosa has
    reduced the amplitude to one of the original 2 ** 15
    Then determine if you need to perform delta and double delta calculations on the obtained audio
    Finally, perform the normalization operation on the features (note that the features in each dimension are obtained)
    """

    audio,fs = librosa.load(audio_filename,sr = None)
    audio_max = np.max(audio)
    audio = audio/audio_max
    features = logfbank(signal = audio,samplerate=fs,winlen=0.025,winstep=0.01,
         nfilt=40,nfft=2048,lowfreq=0,highfreq=None,preemph=0.97)
    if delta and delta_2:
        delta_features = python_speech_features.delta(features,N=1)
        delta2_features = python_speech_features.delta(delta_features,N=1)
        train_inputs = np.hstack((features,delta_features,delta2_features))
        train_inputs = (train_inputs - np.mean(train_inputs,axis = 0)) / np.std(train_inputs,axis = 0)
    else:
        train_inputs = (features - np.mean(features,axis = 0)) / np.std(features, axis = 0)

    # Return results
    return train_inputs
def feature_label_save(converted_paths,index_labels,phn2ind,ind2phn,data_paths):
    '''
    Perform audio feature file extraction
    save all extracted labels and features
    At the same time the mapping between phn2ind and ind2phn phoneme and index is stored
    '''
    delta = True
    delta_2 = True
    assert len(converted_paths) == len(index_labels)
    num_data = len(converted_paths)
    all_paths = []
    for i in range(num_data):

        audio_filename = converted_paths[i]
        #Need to be converted to an absolute path
        audio_filename = os.path.abspath(audio_filename)
        '''
        Perform data extraction and label storage
        '''
        label_filename = audio_filename.split(audio_filename.split('.')[-1])[0]+'label'
        feature_filename = audio_filename.split(audio_filename.split('.')[-1])[0]+'feat'
        print(label_filename)
        print(feature_filename)
        #Extracting desired features
        audios = audioToInputVector(audio_filename, delta, delta_2)
        #Store the obtained features
        audio_file(feature_filename,audios)
        labels = index_labels[i]
        label_file(label_filename,labels)
        all_paths.append(audio_filename)
        print(len(all_paths))
    random.shuffle(converted_paths)
    #train, valid, test data storage address
    #The data set is divided into three parts, using train, valid, and test, respectively. No crossover between sections
    with open(data_paths+ 'train_paths','wb') as f:
        pickle.dump(all_paths[:55000],f)
    with open(data_paths+ 'valid_paths','wb') as f:
        pickle.dump(all_paths[55000:57000],f)
    with open(data_paths+ 'test_paths','wb') as f:
        pickle.dump(all_paths[57000:],f)
    #将phoneme和index之间的映射关系存储为pickle文件
    with open(data_paths+ 'phone39index','wb') as b:
        pickle.dump(phn2ind,b)
    with open(data_paths+ 'index39phone','wb') as d:
        pickle.dump(ind2phn,d)

    print(data_paths,'train_paths',' has been done !')
    print(data_paths,'valid_paths',' has been done !')
    print(data_paths,'test_paths',' has been done !')
    print(data_paths,'phone39index',' has been done !')
    print(data_paths,'index39phone',' has been done !')

    
def read_pickle(filepath):
    f = open(filepath,'rb')
    data = pickle.load(f)
    f.close()
    return data
def main():
    #Create a data folder, and store the training data address, 
    #validation data address, test data address, and phoneme to index mappings in this folder
    if not os.path.exists('data'):
        os.makedirs("data")
    #39dict uses cmu's 39 phoneme dictionary, which has been converted to a pickle file
    dict_path = "39dict"
    
    dataset = "train-clean-100/LibriSpeech/train-clean-100"
    txts,audio_paths = load_txt_path(dataset)
    converted_paths,converted_labels,convert_dict = convert_txt2phn(txts,audio_paths,dict_path)
    phn2ind,ind2phn = make_index_dict(converted_labels,convert_dict)
    index_labels = phone2index(phn2ind,converted_labels)
    feature_label_save(converted_paths,index_labels,phn2ind,ind2phn,'data/')
    
if __name__ == '__main__':
    main()




