# coding: utf-8

import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import pickle
import math
from six.moves import xrange

#--------------------------------------------define function---------------------------------------------#
#加载词向量，返回词向量字典
def load_embedding_dict(embedding_file):
    lines=open(embedding_file,'r').readlines()
    print("load embeddings_dict......")
    embeddings_dict={}
    for i in tqdm(xrange(len(lines))):
        if(i<len(lines)-1):
            line_list=lines[i+1].strip().split(' ')
            word=line_list[0]
            embedding=np.array([float(v) for v in line_list[1:257]],dtype=np.float32)
            embeddings_dict[word]=embedding
    #embeddings=np.array(embeddings)
    print("finish to load embeddings_dict......")
    print("len(embeddings_dict):",len(embeddings_dict))
    return embeddings_dict


def batch_iter(data, batch_size, num_epochs, shuffle=False):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    # 每个epoch的num_batch
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    print("num_batches_per_epoch:",num_batches_per_epoch)
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch

        print("epoch: ", epoch)

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
    """
    :param predict_label_and_marked_label_list: 一个元组列表。例如
    [ ([1, 2, 3, 4, 5], [4, 5, 6, 7]),
      ([3, 2, 1, 4, 7], [5, 7, 3])
     ]
    需要注意这里 predict_label 是去重复的，例如 [1,2,3,2,4,1,6]，去重后变成[1,2,3,4,6]
    
    marked_label_list 本身没有顺序性，但提交结果有，例如上例的命中情况分别为
    [0，0，0，1，1]   (4，5命中)
    [1，0，0，0，1]   (3，7命中)

    """
    right_label_num = 0  #总命中标签数量
    right_label_at_pos_num = [0, 0, 0, 0, 0]  #在各个位置上总命中数量
    sample_num = 0   #总问题数量
    all_marked_label_num = 0    #总标签数量
    for predict_labels, marked_labels in predict_label_and_marked_label_list:
        sample_num += 1
        marked_label_set = set(marked_labels)
        all_marked_label_num += len(marked_label_set)
        for pos, label in zip(range(0, min(len(predict_labels), 5)), predict_labels):
            if label in marked_label_set:     #命中
                right_label_num += 1
                right_label_at_pos_num[pos] += 1

    precision = 0.0
    for pos, right_num in zip(range(0, 5), right_label_at_pos_num):
        precision += ((right_num / float(sample_num))) / math.log(2.0 + pos)  # 下标0-4 映射到 pos1-5 + 1，所以最终+2
    recall = float(right_label_num) / all_marked_label_num

    return 1*(precision * recall) / (precision + recall )
#------------------------------------------endding define function-------------------------------------------#





# 导入question_train_set
reader = pd.read_table('./ieee_zhihu_cup/question_eval_set.txt',sep='\t',header=None)
print(reader.iloc[0:5])


# In[3]:

# 计算一段文本中最大词汇数
x_text = reader.iloc[:,2]
max_document_length = 0
for i,line in enumerate(x_text):
    try:
        temp = line.split(',')
        max_document_length = max(max_document_length,len(temp))
    except:
        # 其中有一行数据为空
        pass
#         x_text[i] = " "

print("max_document_length:",max_document_length)

# 载入字典
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore("vocab_dict")

#[index:word]词典
vocab_dict = vocab_processor.vocabulary_._mapping   #键值为[word:index]
vocab_dict = {i:w for w,i in vocab_dict.items()}   #键值为[index:word]
#print("vocab_dict",vocab_dict)
print("len(vocab_dict):",len(vocab_dict))

#构建与vocab_dict相对应的word embeddings(shape=[vocab_size, embedding_size])
embedding_file="./ieee_zhihu_cup/word_embedding.txt"
embeddings_dict=load_embedding_dict(embedding_file)

embeddings = np.zeros([len(vocab_dict), 256], dtype=np.float32)

missing_count = 0
# for i in  tqdm(xrange(len(vocab_dict)-1)):
for k, v in tqdm(vocab_dict.items()):
    #如果字典vocab_dict的词在embeddings_dict词典中出现则按照其对应的词序添加进embeddings词向量
    # if vocab_dict[i+1] in embeddings_dict:    
    if v in embeddings_dict:    
        embeddings[k] = embeddings_dict[v]

    else: #如果在词向量字典中找不到对应的词向量则随机生成
	missing_count += 1
        print('can not find in embedding dict: ', k, v)
        embeddings[k] = np.array(np.random.uniform(-1.0, 1.0,size=[256]),dtype=np.float32)

print("missing count: ", missing_count)
print("embeddings.shape:",embeddings.shape)
print(embeddings[0:3])

# In[]:

# 按','切分数据
text = []
for line in x_text:
    try:
        text.append(line.split(','))
    except:
        # 其中有一行数据为空
        text.append(' ')


# In[5]:

# 把数据集变成编号的形式
x = []
for line in tqdm(text):
    line_len = len(line)
    text2num = []
    for i in xrange(max_document_length):
        if(i < line_len):
            try:
                text2num.append(vocab_processor.vocabulary_.get(line[i])) # 把词转为数字
            except:
                text2num.append(0) # 没有对应的词
        else:
            text2num.append(0) # 填充0
    x.append(text2num)
x = np.array(x)
x[:5]


# In[6]:





# sequence_length-最长词汇数
sequence_length=x.shape[1]
# num_classes-分类数
num_classes=1999
# vocab_size-总词汇数
vocab_size=len(vocab_processor.vocabulary_)
# embedding_size-词向量长度
embedding_size=256
# filter_sizes-卷积核尺寸3，4，5
filter_sizes=list(map(int, [3,4,5]))
# num_filters-卷积核数量
num_filters=1024



#---------------------------------------------define network---------------------------------------------#
# 定义placeholder
input_x = tf.placeholder(tf.int32, [None, x.shape[1]], name="input_x")
dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

# Embedding layer
with tf.device('/cpu:0'), tf.name_scope("embedding"):
    embeddings=tf.Variable(embeddings,trainable=True,name="embeddings")
    #Weights = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="Weights")
    ## shape:[None, sequence_length, embedding_size]
    embedded_chars = tf.nn.embedding_lookup(embeddings, input_x)
    # 添加一个维度，shape:[None, sequence_length, embedding_size, 1]
    embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

# Create a convolution + maxpool layer for each filter size
pooled_outputs = []
for i, filter_size in enumerate(filter_sizes):
    with tf.name_scope("conv-maxpool-%s" % filter_size):
        # Convolution Layer
        filter_shape = [filter_size, embedding_size, 1, num_filters]
        W = tf.Variable(
            tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        b = tf.Variable(
            tf.constant(0.1, shape=[num_filters]), name="b")
        conv = tf.nn.conv2d(
            embedded_chars_expanded,
            W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")
        # Apply nonlinearity
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        # Maxpooling over the outputs
        pooled = tf.nn.max_pool(
            h,
            ksize=[1, sequence_length - filter_size + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")
        pooled_outputs.append(pooled)

# Combine all the pooled features
num_filters_total = num_filters * len(filter_sizes)
print("num_filters_total:", num_filters_total)
h_pool = tf.concat(pooled_outputs, 3)
h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

# Add dropout
with tf.name_scope("dropout"):h_drop = tf.nn.dropout(h_pool_flat,dropout_keep_prob)

# Final (unnormalized) scores and predictions
with tf.name_scope("output"):
    W = tf.get_variable(
        "W",
        shape=[num_filters_total, num_classes],
        initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
    scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
#------------------------------------------endding define network-----------------------------------------#




# 选择模型
checkpoint_file = "./models/model-42450"
    
with tf.Session() as sess:
    predict_top_5 = tf.nn.top_k(scores, k=5)
    sess.run(tf.global_variables_initializer())
    i = 0
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_file)

    # Generate batches
    batches = batch_iter(list(x), 1000, 1)
    
    for x_batch in batches:
        i = i + 1
        predict_5 = sess.run(predict_top_5,feed_dict={input_x:x_batch,dropout_keep_prob:1.0})
        if i == 1:
            predict = predict_5[1]
        else:
            predict = np.concatenate((predict,predict_5[1]))
        if (i%5==0):
            print ("Evaluation:step",i)

    np.savetxt("predict.txt",predict,fmt='%d')


# In[ ]:



