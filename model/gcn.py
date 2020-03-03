import tensorflow as tf


import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib import layers
degree_max = 100

class GCN(object):
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
        with tf.device(device), tf.name_scope(mode), tf.variable_scope("gnn", reuse=reuse):
            #
            self.W_1 = tf.get_variable(
                name='W_1', shape=[degree_max, hidden_size],
                initializer=tf.random_normal_initializer(hidden_stdv),
                # initializer=tf.zeros_initializer(),
                trainable=True,
            )
            self.W_2 = tf.get_variable(
                name='W_2', shape=[hidden_size, hidden_size],
                initializer=tf.random_normal_initializer(hidden_stdv),
                # initializer=tf.zeros_initializer(),
                trainable=True,
            )
        # #------------feed-----------------##
        # input data are edge information of a batch of start, end and the changes
        self.input_x = input_x = tf.placeholder(tf.int32, (batch_size, ))
        self.input_y = input_y = tf.placeholder(tf.int32, (batch_size, ))
        self.negative_sample = negative_sample = tf.placeholder(tf.int32, (batch_size, ))
        self.input_adj = input_adj = tf.placeholder(tf.float32, (node_num, node_num))
        # self.feature_h0 = feature_h0 = tf.ones(shape=(node_num, 100), dtype=tf.float32) * hidden_stdv
        self.feature_h0 = feature_h0 = tf.placeholder(tf.float32, (node_num, degree_max))
        # self.edge_y = edge_y = tf.placeholder(tf.float32, [batch_size, 1])

        with tf.device(device), tf.name_scope(mode), tf.variable_scope("DynGCN", reuse=reuse):
            self.final_embedding = self.gcn(input_adj, feature_h0)

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

    def gcn(self, adj, feature):
        H0 = feature
        H1 = tf.nn.relu(tf.matmul(tf.matmul(adj, H0), self.W_1))
        H2 = tf.matmul(tf.matmul(adj, H1), self.W_2)
        output = tf.tanh(H2)
        # return tf.reshape(H2, [self.node_num, self.output_dim])
        return output

    def normalize_adj(self, adj):
        if adj.shape[0] < self.node_num:
            adj = np.pad(adj, (0, self.node_num - adj.shape[0]), 'constant')

        adj = adj + np.eye(adj.shape[0])
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        return d_mat_inv_sqrt.transpose().dot(adj).dot(d_mat_inv_sqrt)

    def degree_feature(self, adj):
        node_degree = np.sum(adj, axis=0)
        # print "node_degree", node_degree.shape
        node_degree = node_degree.flatten().tolist()
        # print node_degree
        onehot = np.eye(degree_max)
        # print "node_degree2", node_degree.shape
        # print node_degree[0]
        def _quat(x):
            if x > degree_max -1 :
                x = degree_max -1
            return x
        node_degree = map(_quat,  node_degree[0])
        degree_f = map(lambda x: onehot[int(x)], node_degree)
        degree_f = np.asarray(degree_f)
        if degree_f.shape[0] < self.node_num:
            zeros_np = np.zeros((self.node_num - degree_f.shape[0], self.hidden_size))
            degree_f = np.concatenate((degree_f, zeros_np), axis=0)
        # print(degree_f.shape)
        return np.asarray(degree_f)



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
        return w

    def _delta_tanh(self, x):
        return 4. / tf.square(tf.exp(x) + tf.exp(-x))

    def update_lr(self, session, learning_rate):
        if self.is_training:
            session.run(tf.assign(self.learning_rate, learning_rate))

    def for_long_term_iter(self, session):
        # let the t-1 embedding to load to the t embedding
        # from node_embeddings_changable to node_embeddings
        # bad application
        # save the node_embeddings_changable as a temp node embedding
        saver = tf.train.Saver({'gnn/node_embedding:0': self.node_embeddings_changable})
        # print self.node_embeddings_changable
        save_name = 'temp.ckpt'
        saver_path = saver.save(session, "save/model/" + save_name)

        # load the temp node embedding to node_embeddings
        saver = tf.train.Saver({'gnn/node_embedding:0': self.node_embeddings})
        save_name = 'temp.ckpt'
        saver.restore(session, "save/model/" + save_name)
        # saver.restore(session, saver_path)



