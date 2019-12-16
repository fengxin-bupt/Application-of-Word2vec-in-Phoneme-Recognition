#-*- encoding:utf-8 -*-
#This script is used to generate a corpus of training word2vector
import pickle
import h5py

#Read the storage address of all training data sets
with open('data/train_paths','rb') as f:
    data = pickle.load(f)
#Read the mapping between phoneme and index
with open('data/phone39index','rb') as b:
    phn2ind = pickle.load(b)
#All data is stored in all_voc
all_voc = []
i = 1
for path in data:
    print(i,path)
    path = path.split(path.split('.')[-1])[0] + 'label'

    label_file = h5py.File(path,'r')

    label = label_file['label'].value
    label_list = []
    for lab in label:
        label_list.append(lab)
        label_list.append(lab)
    pad = '<PAD>'
    label_list.append(phn2ind[pad])
    label_list.append(phn2ind[pad])
    label_file.close()
    all_voc.extend(label_list)
    i = i+1
#Store the extracted corpus
f = open('data/embedding_voc','wb')
pickle.dump(all_voc,f)
f.close()