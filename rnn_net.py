import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

class TextLSTM(object):

    def __init__(self, sequence_length, num_classes, vocab_size,
      embeddings, l2_reg_lambda=0.0):
    	print('using TextLSTM.')
    	lstm_cell_layer_count = 2

        embedding_size = embeddings.shape[1]

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.is_train = tf.placeholder(tf.bool, name='is_train') 

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            embeddings=tf.Variable(embeddings,trainable=True,name="embeddings")

            # embedded_chars = [None, sequence_length, embedding_size]
            embedded_chars = tf.nn.embedding_lookup(embeddings, self.input_x)


        with tf.name_scope('lstm-layers'):
        	x = tf.unstack(embedded_chars, sequence_length, 1)
        	cell_layers = []

	        for i in range(lstm_cell_layer_count):
	        	lstm_cell = rnn.BasicLSTMCell(embedding_size, forget_bias=1.0)
	        	lstm_cell = rnn.DropoutWrapper(lstm_cell, self.dropout_keep_prob)

	        	cell_layers.append(lstm_cell)
	        stacked_lstm = rnn.MultiRNNCell(cell_layers, state_is_tuple=True)

	    	# outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
	    	outputs, states = rnn.static_rnn(stacked_lstm, x, dtype=tf.float32)

	    	# lstm_output = outputs[-1]
    		lstm_output = tf.concat(outputs, 1)
    		print("outputs shape: ", tf.shape(outputs))
    		print("lstm_output shape: ", tf.shape(lstm_output))

        with tf.name_scope("output"):
            W = tf.get_variable("output_W",
                shape=[embedding_size, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            output = tf.nn.xw_plus_b(lstm_output, W, b, name="scores")
            self.scores = tf.nn.sigmoid(output)

        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=self.input_y)
            self.loss = 10 * tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            self.predict_top_5 = tf.nn.top_k(self.scores, k=5)
            self.label_top_5 = tf.nn.top_k(self.input_y, k=5)


class TextBiLSTM(object):

    def __init__(self, sequence_length, num_classes, vocab_size,
      embeddings, l2_reg_lambda=0.0):
    	print('using TextLSTM.')

        embedding_size = embeddings.shape[1]

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.is_train = tf.placeholder(tf.bool, name='is_train') 

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            embeddings=tf.Variable(embeddings,trainable=True,name="embeddings")

            # embedded_chars = [None, sequence_length, embedding_size]
            embedded_chars = tf.nn.embedding_lookup(embeddings, self.input_x)

        with tf.name_scope('lstm'):
        	x = tf.unstack(embedded_chars, sequence_length, 1)
        	lstm_cell = rnn.BasicLSTMCell(embedding_size, forget_bias=1.0)

        	outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        	lstm_output = outputs[-1]

        with tf.name_scope("output"):
            W = tf.get_variable("output_W",
                shape=[embedding_size, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            output = tf.nn.xw_plus_b(lstm_output, W, b, name="scores")
            self.scores = tf.nn.sigmoid(output)

        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=self.input_y)
            self.loss = 10 * tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            self.predict_top_5 = tf.nn.top_k(self.scores, k=5)
            self.label_top_5 = tf.nn.top_k(self.input_y, k=5)

