#-*- encoding:utf-8 -*-
#This script is used to count errors, the number of all error types and their proportion
import os
import numpy as np
import pickle
def split_target_pred(phone_list):
    '''
    Separate all targets from pred
    All targets are in odd rows
    All preds are on even lines
    '''
    data_num = len(phone_list) // 3
    targets = []
    preds = []
    for i in range(data_num):
        target_list = phone_list[3*i+1]
        #将所有的音素分离成一个列表
        target = target_list.strip().split(' ')
        pred_list = phone_list[3*i + 2]
        pred = pred_list.strip().split(' ')
        targets.append(target)
        preds.append(pred)
    return targets,preds
def search_min(delete,insert,substitute):
    '''
    Finding out which situation is right
    There are three cases
    '''
    result = 0
    if delete < insert and delete < substitute:
        #To delete the error type, you need to add 1 to the total number of errors, 
        #and then mark the movement direction as 2
        result = delete
        direction = 2
        return result,direction
    elif insert < delete and insert < delete:
        #insert error type，
        result = insert
        direction = 1
        return result,direction
    else:
        result = substitute
        direction = 0
        return result,direction
def error_nums(target,pred,distance_mat,path_mat):
    '''
    After tracing the path, three errors are counted
    参数说明：
    target:Real phoneme sequence
    pred:Generated phoneme sequences (repeated need to be eliminated)
    distance_mat：Recorded the shortest distance from (0,0) to (i, j) point
    path_mat：Recorded the path direction during dynamic planning
    delete_num：Number of delete errors in the entire generation sequence
    delete_list：In the generated sequence, the delete error occurs at the target
    insert_num：Number of insert errors in the entire generation sequence
    insert_num：Where the insert error appears in pred in the generated sequence
    substitute_num：Number of subtitute errors in the generated sequence
    sub_ed_list：The phoneme position to be replaced in the target sequence
    sub_use_list：A substitution that occurs at that position in the pred sequence
    '''
    (i,j) = np.shape(path_mat)
    i = i - 1
    j = j - 1
    delete_num = 0
    insert_num = 0
    substitute_num = 0
    delete_list = []
    insert_list = []
    sub_ed_list = []
    sub_use_list = []
    while i > 0 or j > 0:
        #Dynamic programming for retroactive error statistics
        direction = path_mat[i,j]
        distance = distance_mat[i+1,j+1]
        if direction == 2:
            #This is a delete error
            delete_num += 1
            delete_list.append(target[i])
            i = i - 1

        elif direction == 1:
            #This is an insert error
            insert_num += 1
            insert_list.append(pred[j])
            j = j - 1
        else:
            #This is a substitute error
            if distance_mat[i+1,j+1] - distance_mat[i,j] == 1:
                substitute_num += 1
                sub_ed_list.append(target[i])
                sub_use_list.append(pred[j])
                i = i - 1
                j = j - 1
            #There is no error in this case
            else:
                i = i - 1
                j = j - 1
    sub_dict = {}
    sub_num_dict = {}
    #Statistics for phoneme errors in sub errors
    for i in range(len(sub_use_list)):
        sub_dict[sub_ed_list[i]] = sub_use_list[i]
        convert_ = str(sub_ed_list[i]) + str(2) + str(sub_use_list[i])
        if convert_ not in sub_num_dict:
            sub_num_dict[convert_] = 1
        else:
            sub_num_dict[convert_] += 1

    return delete_num,delete_list,insert_num,insert_list,substitute_num,sub_dict,sub_num_dict
def Levenshtein_distance(target,pred):
    '''
    Calculation of label error rate
    Three wrong statistics
    deletion，insertion，substitution
    '''
    len1 = len(target)+1
    len2 = len(pred)+1
    distance_mat = np.zeros([len1,len2])
    path_mat = np.zeros([len(target),len(pred)])
    for i in range(len1):
        distance_mat[i,0] = i
    for j in range(len2):
        distance_mat[0,j] = j
    for i in range(1,len1):
        for j in range(1,len2):
            if target[i-1] == pred[j-1]:
                distance_mat[i,j] = distance_mat[i-1,j-1]
                path_mat[i-1,j-1] = 0
            else:

                #Make the wrong lookup
                deletion = distance_mat[i-1,j] + 1
                insertion = distance_mat[i,j-1] + 1
                substitution = distance_mat[i-1,j-1] + 1
                distance_mat[i,j],direction = search_min(deletion,insertion,substitution)
                path_mat[i-1,j-1] = direction

    #Calculation of ler
    label_error_rate = distance_mat[-1,-1] / len(target)

    delete_num,delete_list,insert_num,insert_list,substitute_num,sub_dict,sub_num_dict = error_nums(target,pred,distance_mat,path_mat)
    delete_error_rate = delete_num / len(target)
    insert_error_rate = insert_num / len(target)
    sub_error_rate = substitute_num / len(target)
    return label_error_rate,delete_error_rate,insert_error_rate,sub_error_rate,sub_dict,sub_num_dict,delete_list

def LER(pred_path):
    '''
    Read all labels and pred from the file, and count all error types
    '''
    substitute_dict = {}
    substitute_num_dict = {}
    ler_list = []
    der_list = []
    ier_list = []
    ser_list = []
    delete_num_dict = {}
    with open(pred_path,'r') as f:
        all_data = f.readlines()
    targets,preds = split_target_pred(all_data)
    assert len(targets) == len(preds)
    for i in range(len(targets)):
        target = targets[i]
        pred = preds[i]
        ler,der,ier,ser,sub_dict,sub_num_dict,delete_list = Levenshtein_distance(target,pred)
        ler_list.append(ler)
        der_list.append(der)
        ier_list.append(ier)
        ser_list.append(ser)
        for convert in sub_dict.keys():
            if convert not in substitute_dict:
                substitute_dict[convert] = sub_dict[convert]
        for sub_num in sub_num_dict.keys():
            if sub_num not in substitute_num_dict:
                substitute_num_dict[sub_num] = 1
            else:
                substitute_num_dict[sub_num] = substitute_num_dict[sub_num] + sub_num_dict[sub_num]
        for delete in delete_list:
            if delete not in delete_num_dict.keys():
                delete_num_dict[delete] = 1
            else:
                delete_num_dict[delete] += 1
    #delete_num_dict = sorted(delete_num_dict.items(),key = lambda x:x[0])
    average_ler = np.mean(ler_list)
    average_der = np.mean(der_list)
    average_ier = np.mean(ier_list)
    average_ser = np.mean(ser_list)
    '''
    with open('substitute_dict','wb') as b:
        pickle.dump(substitute_dict,b)
    with open('substitute_num','wb') as d:
        pickle.dump(substitute_num_dict,d)
        print(substitute_num_dict)
    with open('delete_num','wb') as a:
        pickle.dump(delete_num_dict,a)
        print(delete_num_dict)
    '''
    print('LER:',average_ler,' DER:',average_der,' IER:',average_ier, ' SER:',average_ser)
path = '39phone_result_libri'
LER(path)



