import tensorflow as tf


import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib import layers


class DynGCN(object):
    def __init__(self, config, device, loader, mode):
        self.config = config
        self.mode = mode
        if mode == "Train":
            self.is_training = True
            self.batch_size = self.config.train_batch_size
            self.maxstep_size = self.config.train_step_size
            reuse = None
        elif mode == "Valid":
            self.is_training = False
            self.batch_size = self.config.valid_batch_size
            reuse = True
        else:
            self.is_training = False
            self.batch_size = self.config.test_batch_size
            reuse = True

        self.hidden_size = hidden_size = config.hidden_size
        self.learning_rate = learning_rate = config.learning_rate
        opt = config.sgd_opt
        batch_size = self.batch_size
        self.node_num = node_num = config.node_num
        self.max_degree = max_degree = config.max_degree
        self.n_layer = n_layer = config.n_layer
        # assert batch_size == 1
        self.path = loader.path_file
        self.embedding_path = self.path + loader.embedding_path
        hidden_stdv = np.sqrt(1. / (hidden_size))

        # embedding initial
        # with tf.device(device), tf.name_scope(mode), tf.variable_scope("gnn", reuse=reuse):


        # #------------feed-----------------##
        # input data are edge information of a batch of start, end and the changes
        self.input_x = input_x = tf.placeholder(tf.int32, (batch_size, ))
        self.input_y = input_y = tf.placeholder(tf.int32, (batch_size, ))
        self.negative_sample = negative_sample = tf.placeholder(tf.int32, (batch_size, ))
        self.adj_now = adj_now = tf.placeholder(tf.float32, (node_num, node_num))
        self.delta_adj = delta_adj = tf.placeholder(tf.float32, (node_num, node_num))
        self.feature_h0 = feature_h0 = tf.placeholder(tf.float32, (node_num, hidden_size))

        with tf.device(device), tf.name_scope(mode), tf.variable_scope("DynGCN", reuse=reuse):
            one_order_hash = tf.reshape(tf.reduce_sum(delta_adj, axis=0), (node_num,))
            one_order_hash = tf.cast(tf.cast(one_order_hash, dtype=tf.bool), dtype=tf.float32)
            # one_order_hash_embed = tf.reshape(tf.stack([one_order_hash] * hidden_size, axis=1), (node_num, hidden_size))

            two_order_hash = tf.cast(tf.cast(tf.matmul(adj_now, tf.reshape(one_order_hash, (node_num,1))), dtype=tf.bool), dtype=tf.float32)
            two_order_hash = tf.reshape(two_order_hash, (node_num,))
            two_order_hash = tf.cast(tf.cast(two_order_hash - one_order_hash > 0, dtype=tf.bool), dtype=tf.float32)
            # two_order_hash_embedd = tf.reshape(tf.stack([two_order_hash] * hidden_size, axis=1), (node_num, hidden_size))

            # three_order_hash = tf.cast(tf.cast(tf.matmul(adj_now, tf.reshape(two_order_hash, (node_num,1))), dtype=tf.bool), dtype=tf.float32)
            # three_order_hash = tf.reshape(three_order_hash, (node_num,))
            # three_order_hash = tf.cast(tf.cast(three_order_hash - one_order_hash > 0, dtype=tf.bool), dtype=tf.float32) 
            # three_order_hash = tf.cast(tf.cast(three_order_hash - two_order_hash > 0, dtype=tf.bool), dtype=tf.float32) 
            # three_order_hash_embedd = tf.reshape(tf.stack([three_order_hash] * hidden_size, axis=1), (node_num, hidden_size))

            self.embedding1order = self.gcn(delta_adj, feature_h0, one_order_hash, n_layer, resue_id=0)
            # print(one_order_hash_embed * self.embedding1order)
            self.embedding2order = self.gcn_2hop(adj_now, self.embedding1order, feature_h0, two_order_hash, n_layer, resue_id=0)
            # self.final_2order_embedding = one_order_hash_embed * self.embedding1order + (1-one_order_hash_embed) * self.embedding2order
            # self.embedding3order = self.gcn_3hop(adj_now, self.embedding2order, self.embedding1order, three_order_hash, n_layer, resue_id=0)

            self.final_embedding = self.embedding2order 

            new_embedding_x = tf.nn.embedding_lookup(self.final_embedding, input_x)
            new_embedding_y = tf.nn.embedding_lookup(self.final_embedding, input_y)
            new_embedding_n = tf.nn.embedding_lookup(self.final_embedding, negative_sample)

            result = tf.reduce_mean(new_embedding_x * new_embedding_y, axis=1)
            result_n = tf.reduce_mean(new_embedding_x * new_embedding_n, axis=1)

            true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(result), logits=result)
            negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(result_n), logits=result_n)
            loss = tf.reduce_sum(true_xent) + tf.reduce_sum(negative_xent)
            self.test1 = result
            self.test2 = result_n
            # loss = - tf.reduce_mean(tf.sigmoid(result - result_n))

        # -------------evaluation--------------
        self.label_xy = tf.placeholder(tf.int32, (batch_size,))
        self.prediction = tf.sigmoid(result)
        self.prediction_n = tf.sigmoid(result_n)

        if mode == 'Valid':
            self.auc_result, self.auc_opt = tf.metrics.auc(
                labels=self.label_xy,
                predictions=self.prediction
            )
        else:
            self.auc_result = self.auc_opt = tf.no_op()
            # self.f1_score = self.f1_opt = tf.no_op()
        # # -------------cost ---------------
        # cost_parameter = 0.

        # score_mean = tf.losses.sigmoid_cross_entropy(
        #     multi_class_labels=self.input_y,
        #     logits=s_pos
        # )
        self.cost = cost = loss

        # ---------------optimizer---------------#
        self.no_opt = tf.no_op()
        self.learning_rate = tf.Variable(config.learning_rate, trainable=False)

        if mode == 'Train':
            self.auc_opt = tf.no_op()
            self.auc_result = tf.no_op()
            if opt == 'Adam':
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)
            if opt == 'Momentum':
                self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(cost)
            if opt == 'RMSProp':
                self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(cost)
            if opt == 'Adadelta':
                self.optimizer = tf.train.AdadeltaOptimizer(self.learning_rate).minimize(cost)

            # self.optimizer = tf.no_op()
        else:
            self.optimizer = tf.no_op()
            self.cost = tf.no_op()


    def gcn(self, adj, feature, one_order_hash, n_layer=1, resue_id=0):
        # here the adj is the delta_adj
        H0 = feature
        one_order_hash_embed = tf.reshape(tf.stack([one_order_hash] * self.hidden_size, axis=1), (self.node_num, self.hidden_size))

        for idx in range(n_layer):

            gcn_w_idx = self.weights("gcn_w", self.hidden_size, idx, resue_id)
            gcn_w_self_idx = self.weights("gcn_self_w", self.hidden_size, idx, resue_id)

            H1 = tf.nn.relu(tf.matmul(tf.matmul(adj, H0), gcn_w_idx) + tf.matmul(H0, gcn_w_idx))

            H0 = H1
            # return tf.reshape(H2, [self.node_num, self.output_dim])

        final_1order_embedding =  one_order_hash_embed * H0 + (1-one_order_hash_embed) *feature

        return final_1order_embedding


    def gcn_2hop(self, adj, then_embedding_x, original_embedding_x, two_order_hash, n_layer=1, resue_id=0):

        # here the adj is the whole adj of graph at this time
        delta_H = then_embedding_x - original_embedding_x
        # this is the delta representation we need to propagate to the neighbouring node
        H0 = delta_H
        # embedding_hash
        two_order_hash_embedd = tf.reshape(tf.stack([two_order_hash] * self.hidden_size, axis=1), (self.node_num, self.hidden_size))

        for idx in range(n_layer):

            gcn_w_2hop_idx = self.weights("gcn_w_2hop", self.hidden_size, 0, resue_id)
            gcn_w_self_2hop_idx = self.weights("gcn_self_w_2hop", self.hidden_size, 0, resue_id)

            H1 = tf.nn.relu(tf.matmul(tf.matmul(adj, H0), gcn_w_2hop_idx) + tf.matmul(then_embedding_x, gcn_w_self_2hop_idx))

            H0 = H1
            # return tf.reshape(H2, [self.node_num, self.output_dim])
        final_2order_embedding = (1 - two_order_hash_embedd) * then_embedding_x + two_order_hash_embedd * H0

        return final_2order_embedding

    def gcn_3hop(self, adj, then_embedding_x, original_embedding_x, three_order_hash, n_layer=1, resue_id=0):
        delta_H = then_embedding_x - original_embedding_x

        three_order_hash_embedd = tf.reshape(tf.stack([three_order_hash] * self.hidden_size, axis=1), (self.node_num, self.hidden_size))

        H0 = delta_H

        for idx in range(n_layer):

            gcn_w_3hop_idx = self.weights("gcn_w_3hop", self.hidden_size, 0, resue_id)
            gcn_w_self_3hop_idx = self.weights("gcn_self_w_3hop", self.hidden_size, 0, resue_id)

            H1 = tf.nn.relu(tf.matmul(tf.matmul(adj, H0), gcn_w_3hop_idx) + tf.matmul(then_embedding_x, gcn_w_self_3hop_idx))

            H0 = H1

        final_3order_embedding =  three_order_hash_embedd * H0 + (1 - three_order_hash_embedd) * then_embedding_x

        return final_3order_embedding

    def mat_3_2(self, x, y):
        x1 = tf.reshape(x, (-1, self.hidden_size))
        return tf.reshape(
            tf.matmul(x1, y),
            (self.batch_size, self.max_degree, self.hidden_size)
        )

    def weights(self, name, hidden_size, layer_x, i):
        image_stdv = np.sqrt(1. / (2048))
        hidden_stdv = np.sqrt(1. / (hidden_size))
        if name == 'gcn_w':
            if i > 0:
                with tf.variable_scope("w", reuse=True):
                    w = tf.get_variable(name='gcn_w_' + str(layer_x),
                                        shape=[hidden_size, hidden_size]
                                        # initializer=tf.ones_initializer()
                                        )
            else:
                with tf.variable_scope("w"):
                    w = tf.get_variable(name='gcn_w_' + str(layer_x),
                                        shape=[hidden_size, hidden_size]
                                        # initializer=tf.ones_initializer()
                                        )

        if name == 'gcn_self_w':

            if i > 0:
                with tf.variable_scope("w", reuse=True):
                    w = tf.get_variable(name='gcn_self_w_' + str(layer_x),
                                        shape=[hidden_size, hidden_size]
                                        # initializer=tf.ones_initializer()
                                        )
            else:
                with tf.variable_scope("w"):
                    w = tf.get_variable(name='gcn_self_w_' + str(layer_x),
                                        shape=[hidden_size, hidden_size]
                                        # initializer=tf.ones_initializer()
                                        )
        if name == 'gcn_w_2hop':

            if i > 0:
                with tf.variable_scope("w", reuse=True):
                    w = tf.get_variable(name='gcn_w_2hop' + str(layer_x),
                                        shape=[hidden_size, hidden_size]
                                        # initializer=tf.ones_initializer()
                                        )
            else:
                with tf.variable_scope("w"):
                    w = tf.get_variable(name='gcn_w_2hop' + str(layer_x),
                                        shape=[hidden_size, hidden_size]
                                        # initializer=tf.ones_initializer()
                                        )
        if name == 'gcn_self_w_2hop':

            if i > 0:
                with tf.variable_scope("w", reuse=True):
                    w = tf.get_variable(name='gcn_self_w_2hop' + str(layer_x),
                                        shape=[hidden_size, hidden_size]
                                        # initializer=tf.ones_initializer()
                                        )
            else:
                with tf.variable_scope("w"):
                    w = tf.get_variable(name='gcn_self_w_2hop' + str(layer_x),
                                        shape=[hidden_size, hidden_size]
                                        # initializer=tf.ones_initializer()
                                        )
        if name == 'gcn_w_3hop':

            if i > 0:
                with tf.variable_scope("w", reuse=True):
                    w = tf.get_variable(name=name + str(layer_x),
                                        shape=[hidden_size, hidden_size]
                                        # initializer=tf.ones_initializer()
                                        )
            else:
                with tf.variable_scope("w"):
                    w = tf.get_variable(name=name + str(layer_x),
                                        shape=[hidden_size, hidden_size]
                                        # initializer=tf.ones_initializer()
                                        )
        if name == 'gcn_self_w_3hop':

            if i > 0:
                with tf.variable_scope("w", reuse=True):
                    w = tf.get_variable(name=name + str(layer_x),
                                        shape=[hidden_size, hidden_size]
                                        # initializer=tf.ones_initializer()
                                        )
            else:
                with tf.variable_scope("w"):
                    w = tf.get_variable(name=name + str(layer_x),
                                        shape=[hidden_size, hidden_size]
                                        # initializer=tf.ones_initializer()
                                        )

        return w

    def _delta_tanh(self, x):
        return 4. / tf.square(tf.exp(x) + tf.exp(-x))

    def update_lr(self, session, learning_rate):
        if self.is_training:
            session.run(tf.assign(self.learning_rate, learning_rate))



