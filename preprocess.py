
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
import utils
import gc

def save_var(var_name, var):
    utils.save_var(var_name, var, temp_folder='preprocess')

def load_var(var_name):
    return utils.load_var(var_name, temp_folder='preprocess')

def calc_words_count(reader, word_idx, words_count_dict):
    if type(word_idx) != list or len(word_idx) == 0:
        print('word_idx must be list, eg, [1, 2], [1]\n')
        return

    print('begin to process: reader.shape: ', reader.shape)

    len_of_reader = reader.shape[0]
    for i in tqdm(xrange(len_of_reader)):
        line = reader.iloc[i]

        #get all words
        words = []
        for idx in word_idx:
            words_str = line[idx]
            if type(words_str) != str:
                continue
            words += words_str.split(',')
            pass

        # calc word count
        for w in words:
            if words_count_dict.has_key(w):
                words_count_dict[w] += 1
            else:
                words_count_dict[w] = 1
            pass

        pass

    print('process finish!\n')
    return

def build_id2wordIDset_dict(reader, id_idx, word_idx, id2wordset_dict):
    global words_to_id_dict
    print('begin to process: reader.shape: ', reader.shape)

    len_of_reader = reader.shape[0]
    for i in tqdm(xrange(len_of_reader)):
        line = reader.iloc[i]

        #get all words
        words = []
        for idx in word_idx:
            words_str = line[idx]
            if type(words_str) != str:
                continue
            words += words_str.split(',')
            pass

        words_id = [words_to_id_dict[w] for w in words]

        w_c_d = {}
        for w in words_id:
            if w_c_d.has_key(w):
                w_c_d[w] += 1
            else:
                w_c_d[w] = 1
            pass

        # words_set = set(words)

        id = long(line[id_idx])

        if id2wordset_dict.has_key(id):
            for w, c in w_c_d.items():
                if id2wordset_dict[id].has_key(w):
                    id2wordset_dict[id][w] += c
                else:
                    id2wordset_dict[id][w] = c
        else:
            id2wordset_dict[id] = w_c_d
        pass

    print('process finish!\n')
    return

# train_reader = pd.read_table('./ieee_zhihu_cup/question_train_set.txt',sep='\t',header=None)

# eval_reader = pd.read_table('./ieee_zhihu_cup/question_eval_set.txt',sep='\t',header=None)

# question_topic_reader = pd.read_table('./ieee_zhihu_cup/question_topic_train_set.txt',sep='\t',header=None)

topic_info_reader = pd.read_table('./ieee_zhihu_cup/topic_info.txt',sep='\t',header=None)

words_count_dict = load_var('words_count_dict')

# if words_count_dict == None:
#     words_count_dict = {}

#     calc_words_count(train_reader, [2, 4], words_count_dict)
#     calc_words_count(eval_reader, [2, 4], words_count_dict)
#     calc_words_count(topic_info_reader, [3, 5], words_count_dict)

#     save_var('words_count_dict', words_count_dict)

#     sort_word_counts = sorted(words_count_dict.items(), key=lambda t: t[1], reverse=False)

#     fp = open('words_count.txt', 'w+')
#     total_count = len(sort_word_counts)
#     fp.write('total count: %10d\n' % total_count)
#     for item in sort_word_counts:

#         fp.write('%-10s: %10d' % (item[0], item[1]))
#         fp.write('\n')
#     fp.close()
#     pass

words_to_id_dict = {item[0]: int(i) for i, item in enumerate(words_count_dict.items())}
id_to_words_dict = {v: k for k, v in words_to_id_dict.items()}

topicId_to_quesIDset_dict = load_var('topicId_to_quesIDset_dict')

# if topicId_to_quesIDset_dict == None:

#     topicId_to_quesIDset_dict = {}

#     len_of_ques_topic = question_topic_reader.shape[0]
#     for i in tqdm(xrange(len_of_ques_topic)):
#         line = question_topic_reader.iloc[i]
#         ques_id = long(line[0])
#         topic_ids = line[1].split(',')

#         topic_id_set = set(topic_ids)

#         for id in topic_ids:
#             topic_id = long(id)
#             if topicId_to_quesIDset_dict.has_key(topic_id):
#                 topicId_to_quesIDset_dict[topic_id].add(ques_id)
#             else:
#                 topicId_to_quesIDset_dict[topic_id] = set([ques_id])
#             pass
#     save_var('topicId_to_quesIDset_dict', topicId_to_quesIDset_dict)

# del eval_reader
# del question_topic_reader
# del topic_info_reader
# gc.collect()

# quesID_to_wordIDset_dict = load_var('quesID_to_wordIDset_dict')
# if quesID_to_wordIDset_dict == None:
#     quesID_to_wordIDset_dict = {}
#     build_id2wordIDset_dict(train_reader, 0, [2, 4], quesID_to_wordIDset_dict)
#     save_var('quesID_to_wordIDset_dict', quesID_to_wordIDset_dict)

topicID_to_wordIDset_dict = load_var('topicID_to_wordIDset_dict')
if topicID_to_wordIDset_dict == None:
    topicID_to_wordIDset_dict = {}
    build_id2wordIDset_dict(topic_info_reader, 0, [3, 5], topicID_to_wordIDset_dict)
    save_var('topicID_to_wordIDset_dict', topicID_to_wordIDset_dict)

topicID_to_infoComWord_Freq_Count_dict = {}

sort_topicID_to_wordIDset = sorted(topicID_to_wordIDset_dict.items(), key=lambda t: len(t[1].keys()), reverse=True)
fp = open('topicID_to_infoComWord_Freq_Count.txt', 'w+')
for item in tqdm(sort_topicID_to_wordIDset):
    topic_id = item[0]
    wordID_freq = item[1]
    
    sort_wordID_freq = sorted(wordID_freq.items(), key=lambda t: t[1], reverse=True)

    fp.write('topic id: %10d, word count: %d\n' % (topic_id, len(sort_wordID_freq)))

    for wid_f in sort_wordID_freq:
        wid = wid_f[0]
        freq = wid_f[1]
        word = id_to_words_dict[wid]
        count = words_count_dict[word]
        fp.write('w: %10s, f: %10d, c: %10d\n' % (word, freq, count))
    pass
    fp.write('\n')
fp.close()

# topicID_to_comWordSet_Count_dict = {}

# fp = open('topicid_to_comWordCount.txt', 'w+')
# for topic_id, ques_id_set in tqdm(topicId_to_quesIDset_dict.items()):
#     comWordSet = set()

#     for ques_id in ques_id_set:
#         for id in quesID_to_wordIDset_dict[ques_id]:
#             comWordSet.add(id_to_words_dict[id])
#         pass

#     fp.write('ques_id: %s, word count: %d\n' % (ques_id, len(comWordSet)))

#     word_count = [None] * 2
#     for w in comWordSet:
#         word_count[0] = w
#         word_count[1] = words_count_dict[w]
#         fp.write('w: %10s; c: %10d\n' % (word_count[0], word_count[1]))
#     fp.write('\n')

#     topicID_to_comWordSet_Count_dict[topic_id] = word_count
#     pass
# fp.close()

