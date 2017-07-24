import tensorflow as tf
from tqdm import tqdm
import numpy as np
import math
import cPickle as pickle
import os

def get_embeddings(vocab_processor, embedding_file, embedding_size):

    #[index:word]
    vocab_dict = vocab_processor.vocabulary_._mapping   #[word:index]
    vocab_dict = {i:w for w,i in vocab_dict.items()}    #[index:word]
    #print("vocab_dict",vocab_dict)
    print("len(vocab_dict):",len(vocab_dict))
    
    #embeddings(shape=[vocab_size, embedding_size])
    embeddings_dict=load_embedding_dict(embedding_file)

    embeddings = np.zeros([len(vocab_dict), embedding_size], dtype=np.float32)

    missing_count = 0
    for k, v in tqdm(vocab_dict.items()):
        if v in embeddings_dict:    
            embeddings[k] = embeddings_dict[v]

        else: 
            missing_count += 1
            # print('can not find k, v in dict: ', k, v)
            embeddings[k] = np.array(np.random.uniform(-1.0, 1.0, size=[embedding_size]), dtype=np.float32)

    print("missing_count:", missing_count)

    print("embeddings.shape:",embeddings.shape)
    print(embeddings[0:3])

    return embeddings


def load_embedding_dict(embedding_file):
    lines=open(embedding_file,'r').readlines()
    print("load embeddings_dict......")
    embeddings_dict={}
    
    len_of_lines = len(lines)
    for i in tqdm(xrange(len_of_lines - 1)):
        line_list=lines[i+1].strip().split(' ')
        word=line_list[0]
        embedding=np.array([float(v) for v in line_list[1:257]],dtype=np.float32)
        embeddings_dict[word]=embedding

    #embeddings=np.array(embeddings)
    print("finish to load embeddings_dict......")
    print("len(embeddings_dict):", len(embeddings_dict))
    return embeddings_dict

def batch_iter(data, batch_size, num_epochs, shuffle=False, show_log=False):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)

    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1

    if show_log:
        print("num_batches_per_epoch:",num_batches_per_epoch)

    for epoch in range(num_epochs):
        # Shuffle the data at each epoch

        if show_log:
            print("epoch:", epoch)

        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def eval(predict_label_and_marked_label_list):

    right_label_num = 0  
    right_label_at_pos_num = [0, 0, 0, 0, 0]  
    sample_num = 0   
    all_marked_label_num = 0    
    for predict_labels, marked_labels in predict_label_and_marked_label_list:
        sample_num += 1
        marked_label_set = set(marked_labels)
        all_marked_label_num += len(marked_label_set)
        for pos, label in zip(range(0, min(len(predict_labels), 5)), predict_labels):
            if label in marked_label_set:     
                right_label_num += 1
                right_label_at_pos_num[pos] += 1

    precision = 0.0
    for pos, right_num in zip(range(0, 5), right_label_at_pos_num):
        precision += ((right_num / float(sample_num))) / math.log(2.0 + pos)  
    recall = float(right_label_num) / all_marked_label_num

    if abs(precision + recall) < 0.01:
        return 0

    return (precision * recall) / (precision + recall)


# temp_folder = './temp'

def save_var(var_name, var, temp_folder):
    if False == os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    var_file_path = '%s/%s.pkl' %(temp_folder, var_name)
    fp = open(var_file_path, 'w+')
    pickle.dump(var, fp, -1)
    fp.close()
    print('save_var: ' + var_file_path)
    return var_file_path

def load_var(var_name, temp_folder):
    var_file_path = '%s/%s.pkl' %(temp_folder, var_name)
    if False == os.path.exists(var_file_path):
        print('var file path not exist: ' + var_file_path)
        return None

    fp = open(var_file_path)
    var = pickle.load(fp)
    fp.close()
    print('load_var: ' + var_file_path)
    return var
