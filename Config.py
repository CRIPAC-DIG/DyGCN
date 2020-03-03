"""
find which config to use and select

"""
import os


def Config(reader, flag="as"):
    if flag == "as":
        return config_as(reader)
    elif flag == "small_amazon":
        return smallconfig_amazon(reader)
    elif flag == "test_ptb":
        return testconfig_ptb(reader)
    else:
        raise ValueError("Invalid model: %s", flag)


class config_as(object):
    epoch_num = 100000
    train_batch_size = 128  # 128
    train_step_size = 4 # 20
    valid_batch_size = 128  # 20
    valid_step_size = 4
    hidden_size = 100  # 512

    max_grad_norm = 1
    init_scale = 0.05
    learning_rate = 0.01  # 0.001  # 0.2
    decay = 0.5
    decay_when = 0.002  # AUC
    decay_epoch = 8000
    sgd_opt = 'RMSProp'
    n_layer = 1
    dropout_prob = 0
    adagrad_eps = 1e-5
    gpu = 0
    max_degree = 100

    def __init__(self, loader):
        self.node_num = len(loader.node2id) # the num of features
        pathfile = loader.path_file

        # for file in os.listdir(path):
        file_list = os.listdir(pathfile)

        file_list = filter(lambda x: x[-5:] == '.index', file_list)

        self.max_step = len(file_list)

        print 'node_num', self.node_num
        print "maxstep_size %d" % self.max_step
        print "gpu_id {}".format(self.gpu)
        print "learning_rate {}".format(self.learning_rate)


class smallconfig_amazon(object):
    epoch_num = 1000
    train_batch_size = 4  # 128
    train_step_size = 4 # 20
    valid_batch_size = 4  # 128
    valid_step_size = 4  # 20
    test_batch_size = 1  # 20
    test_step_size = 4

    def __init__(self, loader):
        vec = loader.itemdict.values()
        # print vec
        vec_r, vec_c = zip(*vec)
        self.vocab_size = (max(vec_r) + 2, max(vec_c) + 2)
        # self.vocab_size = loader.num_items  # 10000
        max_step = 0
        for line in loader.train_set:
            if max_step < len(line):
                max_step = len(line)
        self.maxstep_size = max_step + 1
        print "word-embedding %d" % self.word_embedding_dim


class smallconfig_amazontree(object):
    epoch_num = 1000
    train_batch_size = 100  # 128
    train_step_size = 4 # 20
    valid_batch_size = 100  # 128
    valid_step_size = 4  # 20
    test_batch_size = 100  # 20
    test_step_size = 4
    word_embedding_dim = 100  # 512
    lstm_layers = 1
    lstm_size = 100  # 512
    lstm_forget_bias = 0.0
    # max_grad_norm = 0.25
    max_grad_norm = 1
    init_scale = 0.05
    learning_rate = 1  # 0.2
    decay = 0.5
    decay_when = 0.002  # AUC
    dropout_prob = 0
    adagrad_eps = 1e-5
    gpu = 1

    def __init__(self, loader):
        vec = loader.itemdict.values()
        # vec_r, vec_c = zip(*vec)
        self.tree_size = len(zip(*vec)) - 1
        cat = [max(voc) + 2 for voc in zip(*vec)]
        self.vocab_size = tuple(cat)
        # self.vocab_size = loader.num_items  # 10000
        max_step = 0
        self.loader = loader
        for line in loader.train_set:
            if max_step < len(line):
                max_step = len(line)
        self.user_size = len(loader.train_set)
        self.maxstep_size = max_step + 1
        self.layer_embed = (0.2, 0.3, 0.3, 0.2)
        self.vocab_size_all = len(loader.itemdict)
        assert len(self.layer_embed) == self.tree_size
        print "usernum", self.user_size
        print 'itemnum_vocab_size_all', self.vocab_size_all
        print 'itemnum_vocab_size', self.vocab_size
        print "word-embedding %d" % self.word_embedding_dim


class smallconfig_amazontree1(smallconfig_amazontree):
    def __init__(self, loader):
        smallconfig_amazontree.__init__(self, loader)
        self.layer_embed = (0.1, 0.1, 0.3, 0.5)
        # self.word_embedding_dim = (self.word_embedding_dim / self.layer_embed[-1]) * sum(self.layer_embed)


class smallconfig_amazontree2(smallconfig_amazontree):
    def __init__(self, loader):
        smallconfig_amazontree.__init__(self, loader)
        self.layer_embed = (0, 0, 0, 1)
        # self.word_embedding_dim = (self.word_embedding_dim / self.layer_embed[-1]) * sum(self.layer_embed)


class smallconfig_amazontree3(smallconfig_amazontree):
    def __init__(self, loader):
        smallconfig_amazontree.__init__(self, loader)
        self.layer_embed = (0.6, 0.1, 0.1, 0.2)
        # self.word_embedding_dim = (self.word_embedding_dim / self.layer_embed[-1]) * sum(self.layer_embed)



class testconfig_ptb(object):
      """Tiny config, for testing."""
      init_scale = 0.1
      learning_rate = 1.0
      max_grad_norm = 1
      num_layers = 1
      num_steps = 2
      hidden_size = 2
      max_epoch = 1
      max_max_epoch = 1
      keep_prob = 1.0
      lr_decay = 0.5
      batch_size = 20

      def __init__(self, reader):
          self.vocab_size = len(reader.vocab.words)  # 10000
