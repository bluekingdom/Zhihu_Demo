import tensorflow as tf
import pandas as pd
import numpy as np
from tqdm import tqdm # pip install tqdm
from six.moves import xrange

reader = pd.read_table('./ieee_zhihu_cup/question_train_set.txt',sep='\t',header=None)
# print(reader.iloc[0:5])

eval_reader = pd.read_table('./ieee_zhihu_cup/question_eval_set.txt',sep='\t',header=None)

# 导入question_topic_eval_set
topic_reader = pd.read_table('./ieee_zhihu_cup/question_topic_train_set.txt',sep='\t',header=None)
# print(topic_reader.iloc[0:5])

# 导入topic_info
label_reader = pd.read_table('./ieee_zhihu_cup/topic_info.txt',sep='\t',header=None)

words = reader.iloc[:, 2]

total_samples_count = words.shape[0]

max_docu_len = 0

x_text = []
for i in tqdm(xrange(total_samples_count)):
	word = words[i]
	try:
		max_docu_len = max(max_docu_len, len(word.split(',')))
		x_text.append(word)
	except Exception as e:
		print(i, word)
		print(e)

vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_docu_len)

x_text = np.array(x_text)

vocab_processor.fit_transform(x_text)

vocab_processor.save("vocab_dict")

