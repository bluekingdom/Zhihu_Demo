## coding: utf-8
# In[1]:
#---------------------------------------------import package----------------------------------------------#
import tensorflow as tf
import numpy as np
import os
import time
import numpy as np
import pandas as pd
import math
import gc
from tqdm import tqdm
from six.moves import xrange

from cnn_text import TextCNN_normal
from rnn_net import TextLSTM
from utils import *
#-----------------------------------------endding import package------------------------------------------#

#------------------------------------------define Parameters-------------------------------------------#
# validation数据集占比
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
# 数据集
tf.flags.DEFINE_string("data_file", "./ieee_zhihu_cup/data_topic.txt", "Data source for the train data.")
# 词向量
tf.flags.DEFINE_string("embedding_file", "./ieee_zhihu_cup/word_embedding.txt", "embedding source for the train data.")
# Model Hyperparameters
# 词向量长度
tf.flags.DEFINE_integer("embedding_dim", 256, "Dimensionality of character embedding (default: 256)")
# dropout参数
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
# l2正则化参数
tf.flags.DEFINE_float("l2_reg_lambda", 0.0000, "L2 regularization lambda (default: 0.0005)")

# Training parameters
# 批次大小
tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 64)")
# 迭代周期
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 10)")
# 多少step测试一次
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 50)")
# 多少step保存一次模型
tf.flags.DEFINE_integer("checkpoint_every", 10000, "Save model after this many steps (default: 200)")
# 保存多少个模型
tf.flags.DEFINE_integer("num_checkpoints", 1, "Number of checkpoints to store (default: 5)")

tf.flags.DEFINE_string("checkpoint_file", "", "model restore")

tf.flags.DEFINE_float("lr", "0.001", "learning rate")
#--------------------------------------endding define Parameters-------------------------------------------#


# flags解析
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

# 打印所有参数
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# In[3]:
def run_training(data_file = '', checkpoint_file = ''):

    if os.path.exists(data_file):
        FLAGS.data_file = data_file
        print('load data file: %s' % data_file)

    if os.path.exists(checkpoint_file + '.index'):
        FLAGS.checkpoint_file = checkpoint_file
        print('load checkpoint file: %s' % checkpoint_file)

    # 读取训练数据和标签
    reader = pd.read_table(FLAGS.data_file,sep='\t',header=None)

    total_sample_count = reader.shape[0]
    len_of_label = 1999
    y = np.zeros([total_sample_count, len_of_label], dtype = np.bool)
    x = np.zeros(total_sample_count, dtype = np.object)

    max_document_length = 0

    data_ind = 0
    for i in tqdm(xrange(total_sample_count)):

        line = reader.iloc[i]

        text = line[0]

        max_document_length = max(max_document_length, len(text.split(',')))

        # 按','切分标签
        temp = line[1].split(',')

        # 如果分类数大于5，只取前5个分类
        # ind = [int(t) for t in temp[0: min(5, len(temp))]]
        ind = [int(t) for t in temp]
        # if len(temp) > 5: print(len(temp))

        # 设置标签的对应位置为1，其余位置为0
        y[data_ind][ind] = 1

        x[data_ind] = text
        data_ind += 1
        pass

    print('max_document_length: ', max_document_length)
    #释放内存
    #reader = None
    del reader
    gc.collect()

    # In[4]:
    if data_ind < total_sample_count:
        x = x[0 : data_ind]
        y = y[0 : data_ind, :]
        print('filter some invalid data!')

    # 打印x和y的前行
    print(x[0:5])
    print(y[0:5])

    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore("vocab_dict")
    x = np.array(list(vocab_processor.transform(x)))
    print("x_shape:",x.shape)
    print("y_shape:",y.shape)

    # get embeddings
    embeddings = get_embeddings(vocab_processor, FLAGS.embedding_file, FLAGS.embedding_dim)
    
    # Split train/test set
    # 数据集切分为两部分，训练集和验证集
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x[:dev_sample_index], x[dev_sample_index:]
    y_train, y_dev = y[:dev_sample_index], y[dev_sample_index:]
    
    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    print("x:",x_train[0:5])
    print("y:",y_train[0:5])

    # sequence_length-最长词汇数
    sequence_length=x_train.shape[1]
    # num_classes-分类数
    num_classes=y_train.shape[1]
    # vocab_size-总词汇数
    vocab_size=len(vocab_processor.vocabulary_)
    # embedding_size-词向量长度
    embedding_size=FLAGS.embedding_dim

    l2_reg_lambda = FLAGS.l2_reg_lambda

    #---------------------------------------------define network---------------------------------------------#
    # net = TextCNN_normal(sequence_length, num_classes, vocab_size, embeddings, l2_reg_lambda)
    net = TextCNN_normal(sequence_length, num_classes, vocab_size, embeddings, l2_reg_lambda)

    # 定义优化器
    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(FLAGS.lr).minimize(net.loss)
        # optimizer = tf.train.GradientDescentOptimizer(FLAGS.lr).minimize(net.loss)
 
    # 定义saver，只保存最新的5个模型
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
    #------------------------------------------endding define network-----------------------------------------#

    #---------------------------------------------  run  ---------------------------------------------#
    max_score = 0
    max_score_iter = 0

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        if os.path.exists(FLAGS.checkpoint_file + '.index'):
            saver.restore(sess, FLAGS.checkpoint_file)
            print('restore from checkpoint file: %s' % FLAGS.checkpoint_file)
        else:
            print('%s no exists!' % (FLAGS.checkpoint_file + '.index'))

        i = 0
        batches = batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs, False, True)

        train_time = 0
        
        for batch in batches:
            # 得到一个batch的数据
            x_batch, y_batch = zip(*batch)

            start = time.time()
            sess.run([optimizer], feed_dict={
                net.input_x: x_batch, 
                net.input_y: y_batch, 
                net.dropout_keep_prob: FLAGS.dropout_keep_prob,
                net.is_train: True
                })
            train_time += (time.time() - start)
    
            # 每训练50次测试1次
            if (i % FLAGS.evaluate_every == 0):

                test_start_time = time.time()

                train_loss = sess.run(net.loss, feed_dict={
                    net.input_x:x_batch, 
                    net.input_y:y_batch, 
                    net.dropout_keep_prob:1.0,
                    net.is_train: False
                    })

                score = 0
                test_count = 0
                test_loss = 0

                test_batches = batch_iter(list(zip(x_dev, y_dev)), FLAGS.batch_size, 1, True)
                for test_batch in test_batches:
                    test_count += 1
                    x_dev_b, y_dev_b = zip(*test_batch)
                    predict_5, label_5, _loss = sess.run([net.predict_top_5,net.label_top_5,net.loss], feed_dict={
                        net.input_x:x_dev_b, 
                        net.input_y:y_dev_b, 
                        net.dropout_keep_prob: 1.0,
                        net.is_train: False
                        })

                    predict_label_and_marked_label_list = []
                    for predict,label in zip(predict_5[1],label_5[1]):
                        predict_label_and_marked_label_list.append((list(predict),list(label)))
                        pass

                    s = eval(predict_label_and_marked_label_list)
                    score += s
                    test_loss += _loss

                    # print('predict: %d, score: %f' % (test_count, s))
                    if test_count >= 20:
                        break
                    pass
                pass

                test_time = time.time() - test_start_time

                score /= test_count
                test_loss /= test_count

                # print(len(train_loss))
                # print(len(test_loss))
                print ("step: %d train_loss: %f test_loss: %f score: %f (max: %f) train_time: %0.3f s test_time: %0.3f s" % (i, train_loss, test_loss, score, max_score, train_time, test_time))
                train_time = 0
                #print("score:",score)
    
                if (score > max_score) or (i % FLAGS.checkpoint_every == 0):
                    path = saver.save(sess, "models/model", global_step=i)
                    print("Saved model checkpoint to {}".format(path))

                    max_score_iter = i
                    max_score = score
                    pass
                pass

            i = i + 1
            pass
        pass

    print('max score: %f \t max_score_iter: %d' % (max_score, max_score_iter))
    return max_score_iter
    #-----------------------------------------endding  run  -------------------------------------------#

if __name__ == '__main__':
    run_training()
    

    #last_max_score_iter = 47700
    #for i in range(10):
    #    data_file = './ieee_zhihu_cup/data_topic_block_%d.txt' % i
    #    checkpoint_file = './models/model-%d' % last_max_score_iter
    #    last_max_score_iter = run_training(data_file, checkpoint_file)
