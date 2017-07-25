
# coding: utf-8
# question_train_set.txt：  
#     第一列为 问题id；  
#     第二列为 title 的字符编号序列；  
#     第三列为 title 的词语编号序列；  
#     第四列为描述的字符编号序列；  
#     第五列为描述的词语标号序列。  
#     
# question_topic_train_set.txt：  
#     第一列 问题 id；  
#     第二列 话题 id。  
# 
# topic_info.txt：  
#     第一列为话题 id  
#     第二列为话题的父话题 id。话题之间是有向无环图结构，一个话题可能有 0 到多个父话题；  
#     第三列为话题名字的字符编号序列；  
#     第四列为话题名字的词语编号序列；  
#     第五列为话题描述的字符编号序列；  
#     第六列为话题描述的词语编号序列。  
# 
# 1.title通常来说包含的信息最重要。对于question_train_set.txt文件，为了简单起见，我们只取第三列，title的词语编号序列。    
# 2.对于topic_info.txt，为了简单起见，不考虑2,3,4,5,6列。只是简单的提取话题id，然后转为0-1998的数字（一共有1999个话题）  
# 3.然后合并以上一些数据，得到最后处理后的数据。  

# In[1]:

import pandas as pd
from tqdm import tqdm # pip install tqdm
from six.moves import xrange
from utils import load_embedding_dict, load_var, save_var
<<<<<<< HEAD
import sys
=======
>>>>>>> 1175eb6c677f0f6633cc29f3869477a905867402


# In[2]:

# 导入question_train_set
reader = pd.read_table('./ieee_zhihu_cup/question_train_set.txt',sep='\t',header=None)
# print(reader.iloc[0:5])

eval_reader = pd.read_table('./ieee_zhihu_cup/question_eval_set.txt',sep='\t',header=None)

# 导入question_topic_eval_set
topic_reader = pd.read_table('./ieee_zhihu_cup/question_topic_train_set.txt',sep='\t',header=None)
# print(topic_reader.iloc[0:5])

# 导入topic_info
label_reader = pd.read_table('./ieee_zhihu_cup/topic_info.txt',sep='\t',header=None)
# print(label_reader.iloc[0:5])

word_counts = {}
eval_word_counts = {}

def calc_word_counts(words):
    global word_counts
    for w in words:
        if word_counts.has_key(w):
            word_counts[w] += 1
        else:
            word_counts[w] = 1

def calc_eval_word_counts(words):
    global eval_word_counts
    for w in words:
        if eval_word_counts.has_key(w):
            eval_word_counts[w] += 1
        else:
            eval_word_counts[w] = 1

<<<<<<< HEAD
sort_word_counts = load_var('sort_word_counts', temp_folder='preprocess')
=======
filter_word_list = load_var('filter_word_list', temp_folder='preprocess')
if filter_word_list == None:
    filter_word_list = ['w11', 'w6', 'w111']

max_count_of_topic_words = 0
for i in tqdm(xrange(label_reader.shape[0])):
    words_str = label_reader.iloc[i][3]
    desc_str = label_reader.iloc[i][5]
    words = []
    desc = []
    try:
        if type(words_str) == str:
            words = words_str.split(',')
        if type(desc_str) == str:
            desc = desc_str.split(',')
>>>>>>> 1175eb6c677f0f6633cc29f3869477a905867402

if sort_word_counts == None:
    print('sort word counts file not exists!')
    sys.exit()

<<<<<<< HEAD
filter_word_list = [item[0] for item in sort_word_counts[-10:]]
print('filter word list: ', filter_word_list)
=======
        f_words = [w for w in words if w not in filter_word_list]
>>>>>>> 1175eb6c677f0f6633cc29f3869477a905867402

# max_count_of_topic_words = 0
# for i in tqdm(xrange(label_reader.shape[0])):
#     words_str = label_reader.iloc[i][3]
#     desc_str = label_reader.iloc[i][5]
#     words = []
#     desc = []
#     try:
#         if type(words_str) == str:
#             words = words_str.split(',')
#         if type(desc_str) == str:
#             desc = desc_str.split(',')

#         calc_word_counts(words)
#         calc_word_counts(desc)

#         f_words = [w for w in words if w not in filter_word_list]

#         if len(f_words) == 0:
#             words = desc

#         max_count_of_topic_words = max(max_count_of_topic_words, len(words))
#         pass
#     except Exception as e:
#         print('error line: ', words_str, desc_str)
#         print(i, e)
#     pass

# print('max count of topic words: ', max_count_of_topic_words)
# # 2017.07.22 max_count_of_topic_words = 11

# max_count_of_eval_data_words = 0
# for i in tqdm(xrange(eval_reader.shape[0])):
#     title_words = []
#     desc_words = []

#     data = eval_reader.iloc[i][2]
#     desc = eval_reader.iloc[i][4]

#     if type(data) == str:
#         title_words = data.split(',')

#     if type(desc) == str:
#         desc_words = desc.split(',')

<<<<<<< HEAD
#     calc_word_counts(title_words)
#     calc_word_counts(desc_words)
=======
    filter_words = []
    for w in title_words:
        if w not in filter_word_list:
            filter_words.append(w)
            pass
        pass
>>>>>>> 1175eb6c677f0f6633cc29f3869477a905867402

#     calc_eval_word_counts(title_words)
#     calc_eval_word_counts(desc_words)

#     if type(data) != str and type(desc) != str:
#         print('word be filtered: ', title_words, desc_words)
#         continue

#     filter_words = []
#     for w in title_words:
#         if w not in filter_word_list:
#             filter_words.append(w)
#             pass
#         pass

#     if len(filter_words) == 0:
#         for w in desc_words:
#             if w not in filter_word_list:
#                 filter_words.append(w)
#                 pass
#             pass
            
#     max_count_of_eval_data_words = max(max_count_of_eval_data_words, len(filter_words))

#     if len(filter_words) == 0:
#         print('word be filtered: ', title_words, desc_words)

# print('max count of eval data words: ', max_count_of_eval_data_words)
# # 2017.07.22 max_count_of_eval_data_words = 70


# 合并title 的词语编号序列和话题 id
data_topic = pd.concat([reader.ix[:,2], topic_reader.ix[:,1]], axis=1, ignore_index=True)
# print(data_topic.iloc[0:5])

# max_count_of_data_words = 0

# for i in tqdm(xrange(data_topic.shape[0])):
#     title_words = []
#     desc_words = []

#     data = data_topic.iloc[i][0]
#     desc = reader.iloc[i][4]

#     if type(data) == str:
#         title_words = data.split(',')

#     if type(desc) == str:
#         desc_words = desc.split(',')

#     calc_word_counts(title_words)
#     calc_word_counts(desc_words)

#     if type(data) != str and type(desc) != str:
#         print('word be filtered: ', title_words, desc_words, data_topic.iloc[i][1])
#         continue

<<<<<<< HEAD
#     filter_words = []
#     for w in title_words:
#         if w not in filter_word_list:
#             filter_words.append(w)
#             pass
#         pass

#     if len(filter_words) == 0:
#         for w in desc_words:
#             if w not in filter_word_list:
#                 filter_words.append(w)
#                 pass
#             pass
=======
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
>>>>>>> 1175eb6c677f0f6633cc29f3869477a905867402

#     max_count_of_data_words = max(max_count_of_data_words, len(filter_words))

#     if len(filter_words) == 0:
#         print('word be filtered: ', title_words, desc_words, data_topic.iloc[i][1])

# print('max count of data words: ', max_count_of_data_words)

# 把标签转为0-1998的编号
print('begin to processing label')
labels = list(label_reader.iloc[:,0])
my_labels = []
for label in labels:
    my_labels.append(label)
    
print('begin to processing topic dict')
# 建立topic字典
topic_dict = {}
for i,label in enumerate(my_labels):
    topic_dict[label] = i

<<<<<<< HEAD
# embedding_dict = load_embedding_dict('ieee_zhihu_cup/word_embedding.txt')
=======
embedding_dict = load_embedding_dict('ieee_zhihu_cup/word_embedding.txt')
>>>>>>> 1175eb6c677f0f6633cc29f3869477a905867402

print('begin to process words')
data_idx = [True] * data_topic.shape[0]

<<<<<<< HEAD
for i in tqdm(xrange(data_topic.shape[0])):
    if data_idx[i] == False:
        continue

    title_words = []
    desc_words = []
=======
# max_count_of_data_words = 0

# for i in tqdm(xrange(data_topic.shape[0])):
#     if data_idx[i] == False:
#         continue

#     title_words = []
#     desc_words = []

#     data = data_topic.iloc[i][0]
#     desc = reader.iloc[i][4]

#     if type(data) == str:
#         title_words = data.split(',')

#     if type(desc) == str:
#         desc_words = desc.split(',')
>>>>>>> 1175eb6c677f0f6633cc29f3869477a905867402

    data = data_topic.iloc[i][0]
    desc = reader.iloc[i][4]

<<<<<<< HEAD
    if type(data) == str:
        title_words = data.split(',')

    if type(desc) == str:
        desc_words = desc.split(',')
=======
#     filter_words = []
#     for w in title_words:
#         # if w in embedding_dict and w in word_counts:
#         if w not in filter_word_list:
#             filter_words.append(w)
#             pass
#         pass

#     if len(filter_words) == 0:
#         print('use desc words: ', title_words, desc_words)
#         for w in desc_words:
#             if w not in filter_word_list:
#                 filter_words.append(w)
#                 pass
#             pass
>>>>>>> 1175eb6c677f0f6633cc29f3869477a905867402

    if type(data) != str and type(desc) != str:
        data_idx[i] = False
        print("this row has error:", reader.iloc[i])
        continue

    filter_words = []
    for w in title_words:
        # if w in embedding_dict and w in word_counts:
        if w not in filter_word_list:
            filter_words.append(w)
            pass
        pass

    if len(filter_words) == 0:
        print('consider desc words: ', title_words, desc_words)
        for w in desc_words:
            if w not in filter_word_list:
                filter_words.append(w)
                pass
            pass

    if len(filter_words) == 0:
        print('all word be filtered: ', title_words, desc_words)
        data_topic.iloc[i][0] = ','.join(title_words)
        # data_idx[i] = False
    else:
        data_topic.iloc[i][0] = ','.join(filter_words)

    # 根据“,”切分话题id
    new_label = ''
    temp_topic = data_topic.iloc[i][1].split(',')
    if len(temp_topic) != 0:
        for topic in temp_topic:
            # 判断该label是否在label文件中，并得到该行
            label_num = topic_dict[int(topic)]
            new_label = new_label + str(label_num) + ','
        data_topic.iloc[i][1] = new_label[:-1]
    else:
        print('label be filtered: ', temp_topic)
        data_idx[i] = False

data_topic = data_topic.iloc[data_idx][:]

print(data_topic.iloc[:5])

data_topic.to_csv("./ieee_zhihu_cup/data_topic.txt", header=None, index=None, sep='\t')

# 切分成10块保存
# for i in xrange(10):
#     data_topic_filename = './ieee_zhihu_cup/data_topic_block_' + str(i) + '.txt'
#     if (i+1)*300000 < data_topic.shape[0]:
#         data_topic.iloc[i*300000:(i+1)*300000].to_csv(
#             data_topic_filename, header=None, index=None, sep='\t')
#     else:
#         data_topic.iloc[i*300000:data_topic.shape[0]].to_csv(
#             data_topic_filename, header=None, index=None, sep='\t')
