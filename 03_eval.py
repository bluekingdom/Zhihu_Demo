# coding: utf-8

import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import pickle
import math
import os
import sys
from six.moves import xrange
from utils import *
from cnn_text import *

tf.flags.DEFINE_string("checkpoint_file", "", "model restore")
tf.flags.DEFINE_integer("embedding_dim", 256, "Dimensionality of character embedding (default: 256)")
tf.flags.DEFINE_string("embedding_file", "./ieee_zhihu_cup/word_embedding.txt", "embedding source for the train data.")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

checkpoint_file = FLAGS.checkpoint_file
if False == os.path.exists(checkpoint_file + ".index"):
    print("checkpoint file do not exists: " + checkpoint_file)
    sys.exit()

# 打印所有参数
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

sort_word_counts = load_var('sort_word_counts', temp_folder='preprocess')

if sort_word_counts == None:
    print('sort word counts file not exists!')
    sys.exit()

filter_word_list = [item[0] for item in sort_word_counts[-10:]]
print('filter word list: ', filter_word_list)

# 导入question_train_set
reader = pd.read_table('./ieee_zhihu_cup/question_eval_set.txt',sep='\t',header=None)
print(reader.iloc[0:5])

max_document_length = 0

x_text = np.zeros([reader.shape[0]], dtype=np.object)

# for i,line in enumerate(x_text):
for i in tqdm(xrange(reader.shape[0])):

    title_words = []
    desc_words = []

    data = reader.iloc[i][2]
    desc = reader.iloc[i][4]

    if type(data) == str:
        title_words = data.split(',')

    if type(desc) == str:
        desc_words = desc.split(',')

    if type(data) != str and type(desc) != str:
        x_text[i] = data
        continue

    filter_words = []
    for w in title_words:
        if w not in filter_word_list:
            filter_words.append(w)
            pass
        pass

    if len(filter_words) == 0:
        for w in desc_words:
            if w not in filter_word_list:
                filter_words.append(w)
                pass
            pass

    if len(filter_words) == 0:
        continue
        
    elif len(filter_words) > 70:
        print(filter_words)
        filter_words = filter_words[0:70]

    x_text[i] = ','.join(filter_words)

    max_document_length = max(max_document_length,len(filter_words))


print("max_document_length:",max_document_length)

# 载入字典
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore("vocab_dict")

embeddings = get_embeddings(vocab_processor, FLAGS.embedding_file, FLAGS.embedding_dim)

print("embeddings.shape:",embeddings.shape)

# In[]:

# 按','切分数据
text = []
for line in x_text:
    try:
        text.append(line.split(','))
    except:
        # 其中有一行数据为空
        text.append(' ')

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

cnn = TextCNN_normal(sequence_length, num_classes, vocab_size, 
    embeddings)

# 选择模型
    
with tf.Session() as sess:
    predict_top_5 = cnn.predict_top_5
    
    sess.run(tf.global_variables_initializer())
    i = 0
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_file)

    # Generate batches
    batches = batch_iter(list(x), 1000, 1, False, True)
    
    for x_batch in batches:
        i = i + 1
        predict_5 = sess.run(predict_top_5,feed_dict={
            cnn.input_x:x_batch,
            cnn.dropout_keep_prob:1.0
            })
        if i == 1:
            predict = predict_5[1]
        else:
            predict = np.concatenate((predict,predict_5[1]))
        if (i%5==0):
            print ("Evaluation:step",i)

    predict_file_name = '%s_predict.txt' % (checkpoint_file[9:])
    np.savetxt(predict_file_name,predict,fmt='%d')


# In[ ]:



