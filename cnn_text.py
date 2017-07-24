import tensorflow as tf
import numpy as np

filter_sizes = [3, 4, 5]
num_filters = 1024
fc_output_num = 4096

print('filter_sizes: ', filter_sizes)
print('num_filters: ', num_filters)
print('fc_output_num: ', fc_output_num)

class TextCNN_reg(object):

    def __init__(self, sequence_length, label_sequence_length, vocab_size,
      embeddings, l2_reg_lambda=0.0):

        embedding_size = embeddings.shape[1]
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, label_sequence_length], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.is_train = tf.placeholder(tf.bool, name='is_train') 

        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            embeddings=tf.Variable(embeddings,trainable=True,name="embeddings")
            embedded_chars = tf.nn.embedding_lookup(embeddings, self.input_x)
            embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                # W = tf.get_variable('conv_W_%s' % filter_size, shape=filter_shape, initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    embedded_chars_expanded, W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                bn = tf.contrib.layers.batch_norm(conv, center=True, scale=True, 
                    is_training=self.is_train, scope=('conv-%s-bn' % filter_size) )

                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(bn, b), name="relu")
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
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        with tf.name_scope("dropout"): 
            h_drop = tf.nn.dropout(h_pool_flat,self.dropout_keep_prob)

        # with tf.name_scope("norm"): 
        #     bn = tf.contrib.layers.batch_norm(h_pool_flat, center=True, scale=True, is_training=self.is_train, scope='bn')
        #     norm = tf.nn.relu(bn, name='relu')
        
        with tf.name_scope("output"):
            W = tf.get_variable("output_W",
                shape=[num_filters_total, label_sequence_length],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[label_sequence_length]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            output = tf.nn.xw_plus_b(h_pool_flat, W, b, name="scores")
            self.scores = tf.nn.sigmoid(output)

        with tf.name_scope("loss"):
            losses = tf.nn.l2_loss(self.scores - self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

class TextCNN_normal(object):

    def __init__(self, sequence_length, num_classes, vocab_size,
      embeddings, l2_reg_lambda=0.0):

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
            embedded_chars = tf.nn.embedding_lookup(embeddings, self.input_x)
            embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                # W = tf.get_variable('conv_W_%s' % filter_size, shape=filter_shape, initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
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
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_pool_flat,self.dropout_keep_prob)
        
        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "output_W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            output = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
            self.scores = tf.nn.sigmoid(output)

        with tf.name_scope("loss"):
            # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=scores, labels=self.input_y))
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=self.input_y)
            self.loss = 10 * tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            self.predict_top_5 = tf.nn.top_k(self.scores, k=5)
            self.label_top_5 = tf.nn.top_k(self.input_y, k=5)

class TextCNN_fc(object):

    def __init__(self, sequence_length, num_classes, vocab_size,
      embeddings, l2_reg_lambda=0.0):

        embedding_size = embeddings.shape[1]
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            embeddings=tf.Variable(embeddings,trainable=True,name="embeddings")
            embedded_chars = tf.nn.embedding_lookup(embeddings, self.input_x)
            embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                # W = tf.get_variable('conv_W_%s' % filter_size, shape=filter_shape, initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
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
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_pool_flat,self.dropout_keep_prob)
        
        with tf.name_scope("fc1"):
            W = tf.get_variable("fc1_W", shape=[num_filters_total, fc_output_num], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[fc_output_num]), name="b")
            fc1 = tf.nn.dropout(tf.nn.relu(tf.nn.xw_plus_b(h_drop, W, b, name="fc1"), name='relu'), self.dropout_keep_prob)

            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
        
        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "output_W",
                shape=[fc_output_num, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            self.scores = tf.nn.xw_plus_b(fc1, W, b, name="scores")

        with tf.name_scope("loss"):
            # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=scores, labels=self.input_y))
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = 10 * tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            self.predict_top_5 = tf.nn.top_k(self.scores, k=5)
            self.label_top_5 = tf.nn.top_k(self.input_y, k=5) 


