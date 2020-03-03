import tensorflow as tf
import json
import platform
import numpy as np
import os
import collections

"""

"""


def DynGCN_saver(session, config, best_auc, best_epoch, FLAG='AS'):
    saver = tf.train.Saver()
    # best_auc_str = str(best_auc)
    # best_auc_str = '_auc_' + best_auc_str.replace('.', '_') + '_epoch_' + str(best_epoch)
    # save_name = FLAG + 'model_dim' + str(config.hidden_size) + '_' + str(config.gpu) + \
    #             '_' + str(best_epoch) + '.ckpt'
    save_name = FLAG + 'model_dim' + str(config.hidden_size) + '_' + str(config.gpu) + '.ckpt'
    saver_path = saver.save(session, "save/model/" + save_name)
    print("Model saved in file:", saver_path)


def DynGCN_loader(session, config, FLAG='AS', loadwhich=0):
    saver = tf.train.Saver()
    save_name = FLAG + 'model_dim' + str(config.hidden_size) + '_' + str(config.gpu) + '.ckpt'
    saver.restore(session, "save/model/" + save_name)
    print("Model loaded in file:", "save/model/" + save_name)
