# coding: utf-8
# In[1]:

import pandas as pd
from tqdm import tqdm
import re
import numpy as np
import tensorflow as tf
import os
from six.moves import xrange

tf.flags.DEFINE_string("predict_txt", "", "")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

predict_txt_file = FLAGS.predict_txt
if False == os.path.exists(predict_txt_file):
    print("predict file do not exists: " + predict_txt_file)
    sys.exit()

# 打印所有参数
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# In[2]:

topic_info = pd.read_table("./ieee_zhihu_cup/topic_info.txt",sep='\t',header=None)
print(topic_info.iloc[0:5])


# In[3]:

# 话题字典
topic_dict = {}
for i in xrange(topic_info.shape[0]):
    topic_dict[i] = topic_info.iloc[i][0]


# In[4]:

predict = open(predict_txt_file, "r")
examples = predict.readlines()
text = np.array([line.split(" ") for line in examples])


# In[5]:

label = []
for line in tqdm(text):
    num2label = []
    for i in xrange(5):
        num2label.append(topic_dict[int(line[i])]) # 把0-1999编号转成原来的id
    label.append(num2label)
label = np.array(label)


# In[6]:

np.savetxt("temp.txt",label,fmt='%d')


# In[7]:

def clean_str(string):
    string = re.sub(r" ", ",", string)
    return string

file1 = open('temp.txt', "r")
examples = file1.readlines()
examples = [clean_str(line) for line in examples]
file1.close()

file1 = open('temp.txt', "w")
file1.writelines(examples)
file1.close()


# In[8]:

# predict文件导入
predict_file = 'temp.txt'
predict_reader = pd.read_table(predict_file,sep=' ',header=None)
print(predict_reader.iloc[0:5])


# In[9]:

# 导入question_train_set
eval_reader = pd.read_table('./ieee_zhihu_cup/question_eval_set.txt',sep='\t',header=None)
print(eval_reader.iloc[0:3])


# In[10]:

final_predict = pd.concat([eval_reader.ix[:,0],predict_reader],axis=1)
print(final_predict.iloc[0:5])


# In[11]:

final_predict.to_csv('temp.txt', header=None, index=None, sep=',')

final_file = open('temp.txt', "r")
final_examples = final_file.readlines()
final_examples = [re.sub(r'"',"",line) for line in final_examples]
final_file.close()

final_file = open('final_predict.csv', "w")
final_file.writelines(final_examples)
final_file.close()
