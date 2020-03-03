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
        node_num = config.node_num
        self.max_degree = max_degree = config.max_degree
        self.n_layer = n_layer = config.n_layer
        # assert batch_size == 1
        self.path = loader.path_file
        self.embedding_path = self.path + loader.embedding_path

        hidden_stdv = np.sqrt(1. / (hidden_size))

        # embedding initial
        with tf.device(device), tf.name_scope(mode), tf.variable_scope("gnn", reuse=reuse):
            #
            self.node_embeddings_changable = tf.get_variable(
                name='node_embedding', shape=[node_num + 1, hidden_size],
                initializer=tf.random_normal_initializer(hidden_stdv),
                # initializer=tf.zeros_initializer(),
                trainable=False,
            )
            self.delta_adj = delta_adj = tf.get_variable(
                name='delta_adj', shape=[node_num, max_degree, 2],
                initializer=tf.random_normal_initializer(hidden_stdv),
                trainable=False
            )

        # #------------feed-----------------##
        # input data are edge information of a batch of start, end and the changes
        self.input_x = input_x = tf.placeholder(tf.int32, (batch_size, ))
        self.input_y = input_y = tf.placeholder(tf.int32, (batch_size, ))
        self.negative_sample = negative_sample = tf.placeholder(tf.int32, (batch_size, ))
        # self.edge_y = edge_y = tf.placeholder(tf.float32, [batch_size, 1])

        with tf.device(device), tf.name_scope(mode), tf.variable_scope("DynGCN", reuse=reuse):

            embedding_x = tf.nn.embedding_lookup(self.node_embeddings_changable, input_x)
            embedding_y = tf.nn.embedding_lookup(self.node_embeddings_changable, input_y)
            embedding_n = tf.nn.embedding_lookup(self.node_embeddings_changable, negative_sample)

            result = tf.reduce_mean(embedding_x * embedding_y, axis=1)

        # -------------evaluation--------------
        self.label_xy = tf.placeholder(tf.int32, (batch_size,))
        self.prediction = tf.sigmoid(result)

        if mode == 'Valid':
            self.auc_result, self.auc_opt = tf.metrics.auc(
                labels=self.label_xy,
                predictions=self.prediction
            )
        else:
            self.auc_result = self.auc_opt = tf.no_op()

        self.cost = cost = tf.no_op()

        # ---------------optimizer---------------#
        self.no_opt = tf.no_op()
        self.learning_rate = tf.Variable(config.learning_rate, trainable=False)

        if mode == 'Train':
            self.auc_opt = tf.no_op()
            self.auc_result = tf.no_op()
            self.optimizer = tf.no_op()
        else:
            self.optimizer = tf.no_op()
            self.cost = tf.no_op()

    def mat_3_2(self, x, y):
        x1 = tf.reshape(x, (-1, self.hidden_size))
        return tf.reshape(
            tf.matmul(x1, y),
            (self.batch_size, self.max_degree, self.hidden_size)
        )

    # def load_last_time_embedding(self, idx, sess):
    #     # self.node_embeddings_changable =
    #     path = self.path + 'as_embedding/tensor_node_embedding_' + str(idx) + '.ckpt'
    #     # print "====", path
    #     saver = tf.train.Saver([self.node_embeddings_changable, self.delta_adj])
    #     saver.restore(sess, path)

    def load_this_time_embedding(self, idx, sess):
        # self.node_embeddings_changable =
        path = self.embedding_path + 'tensor_node_embedding_' + str(idx+2) + '.ckpt'
        # print "====", path
        saver = tf.train.Saver([self.node_embeddings_changable, self.delta_adj])
        saver.restore(sess, path)

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

        return w

    def update_lr(self, session, learning_rate):
        if self.is_training:
            session.run(tf.assign(self.learning_rate, learning_rate))

