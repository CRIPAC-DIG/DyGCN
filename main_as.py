import os
import pickle
from datetime import *

import platform
import numpy as np
import tensorflow as tf
from util.load_as import Loader, INFO_LOG

from Config import Config
from util.model_saver import DynGCN_saver, DynGCN_loader
from util.LearningRateUpdater import LearningRateUpdater
from model.dynGCN_3hop import DynGCN
# from model.dynGCN import DynGCN
from util.evalutate import F1score
import time


def run(session, config, model, loader, verbose=False):
    total_cost = 0.

    num_ = 0.

    auc = 0.
    f1_score = F1score(model.batch_size)
    prediction_l = [0.] * model.batch_size
    prediction_n_l = [0.] * model.batch_size
    t_auc = 0.
    t_num = 0.

    def _add_list(x, y):
        for idx in range(len(x)):
            x[idx] += y[idx]
        return x

    time_consume = 0.
    feature_h0 = loader.last_embeddings()
    adj_now, delta_adj = loader.adj()
    for batch in loader.generate_batch_data(batchsize=model.batch_size, mode=model.mode):

        batch_id, batch_num, nodelist1, nodelist2, negative_list = batch

        if model.mode == "Train":
            feed = {
                model.input_x: nodelist1,
                model.input_y: nodelist2,
                model.adj_now: adj_now,
                model.delta_adj: delta_adj,
                model.feature_h0:feature_h0,
                model.negative_sample: negative_list
            }
            out = [model.cost, model.optimizer, model.auc_result,
                   model.auc_opt
                   # , model.prediction, model.prediction_n,
                   # model.test1, model.test2
                   ]

            output = session.run(out, feed)
            cost, _, auc, _ = output #, prediction, prediction_n, test1, test2 

            # prediction_l = _add_list(prediction_l, prediction)
            # prediction_n_l = _add_list(prediction_n_l, prediction_n)

        if model.mode == "Valid":
            # print "nodelist1", np.asarray(nodelist1 * 2).shape
            # print np.asarray(nodelist2 + negative_list).shape
            # print np.asarray([1] * (model.batch_size / 2) + [0] * (model.batch_size /2)).shape

            feed = {
                model.input_x: np.asarray(nodelist1 * 2),
                model.input_y: np.asarray(nodelist2 + negative_list),
                model.adj_now: adj_now,
                model.delta_adj: delta_adj,
                model.feature_h0:feature_h0,
                model.label_xy: np.asarray([1] * (model.batch_size / 2) + [0] * (model.batch_size /2))
            }

            out = [model.cost, model.optimizer, model.auc_result,
                   model.auc_opt, model.prediction]
            begin_time = time.time()
            output = session.run(out, feed)
            time_consume += time.time() - begin_time
            # print output
            cost, _, auc, _, prediction = output

            for idx in range(len(prediction) / 2):
                if prediction[idx] > prediction[idx + len(prediction) / 2]:
                    t_auc += 1
                t_num += 1
            # print prediction
        # print "TEST",prediction
        if model.mode == "Train":
            auc = 0.
            total_cost += cost
        else:
            f1_score.add_f1(
                np.asarray([1] * (model.batch_size / 2) + [0] * (model.batch_size / 2)), prediction
            )
            cost = 0.
            total_cost += cost

        num_ += 1.
        if verbose and batch_id % int(batch_num / 5.) == 1 and model.mode == "Valid":
            INFO_LOG("{}/{}, cost: {}, auc: {}, f1_score: {}".format(
                batch_id, batch_num, total_cost / num_,
                auc, f1_score.return_f1_score()
            ),
            True
            )
    if num_ == 0:
        INFO_LOG("===failed graph===" + str(loader.present_graph), True)
    # if model.mode == "Train":
    #     print("prediction_l", [x / batch_num for x in prediction_l])
    #     print("prediction_l_n", [x / batch_num for x in prediction_n_l])
    # else:
    #     print("valid prediction", f1_score.return_predict_mean())
    if not model.mode == "Train":
        # print "auc", t_auc / t_num
        auc = t_auc / t_num

    return total_cost / num_, {"auc": auc, "f1_score": f1_score.return_f1_score(), "time_consume": time_consume}


def main(_):

    loader = Loader(flag="as")
    config = Config(loader, flag="as")

    if platform.system() == 'Linux':
        gpuid = config.gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(gpuid)
        device = '/gpu:' + str(gpuid)
    else:
        device = '/cpu:0'

    lr_updater = LearningRateUpdater(config.learning_rate, config.decay, config.decay_epoch)

    i = 0
    graph = tf.Graph()
    with graph.as_default():
        trainm = DynGCN(config, device, loader, "Train")
        testm = DynGCN(config, device, loader, "Valid")

    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True

    with tf.Session(graph=graph, config=session_config) as session:
        # print "!!!!!!!!!!!"
        session.run(tf.global_variables_initializer())
        # print "*********"
        # trainm.load_last_time_embedding(loader.present_graph, session)

        # CTR_GNN_loader(session, config)
        best_f1_score = 0.
        best_auc_score = 0.
        best_epoch = 0
        time_consume_t = 0.
        sum_time_consume = 0.
        for epoch in range(config.epoch_num):
            if epoch % 1 == 0 and epoch != 0:
                loader.change_2_next_graph_date()
            trainm.update_lr(session, lr_updater.get_lr())
            # session.run(tf.local_variables_initializer())
            print("which graph now?", loader.present_graph)

            cost, eavluation_result = run(session, config, trainm, loader, verbose=False)
            INFO_LOG("Epoch %d  Train " % epoch + str(eavluation_result), epoch % 1 == 0)
            INFO_LOG("Epoch %d Train costs %.3f" %
                     (epoch, cost), epoch % 100 == 0)
            session.run(tf.local_variables_initializer())
            if epoch % 20 != 0:
                continue
            cost, eavluation_result = run(session, config, testm, loader, verbose=False)
            INFO_LOG("Epoch %d  Valid " % epoch + str(eavluation_result), epoch % 1 == 0)
            INFO_LOG("Epoch %d Valid cost %.3f" % (epoch, cost), epoch % 1 == 0)
            # #
            auc = eavluation_result['auc']
            f1_score = eavluation_result["f1_score"]["micro_f1_score"]
            lr_updater.update(f1_score, epoch)

            if best_f1_score < f1_score:
                best_f1_score = f1_score
                best_epoch = epoch
                DynGCN_saver(session, config, best_f1_score, best_epoch, "as_3hop")

                INFO_LOG("*** best f1_score now is %.5f in %d epoch" % (best_f1_score, best_epoch), True)
                INFO_LOG("BEST Epoch %d  Valid " % epoch + str(eavluation_result), True)

            if best_auc_score < auc:
                best_auc_score = auc
                INFO_LOG("*** best auc now is %.5f in %d epoch" % (best_auc_score, epoch), True)

            INFO_LOG("*** best f1_score now is %.4f in %d epoch" % (best_f1_score, best_epoch), True)
            INFO_LOG("*** best AUC now is %.4f in %d epoch" % (best_auc_score, best_epoch), True)

            # time_consume_t += 1.
            # sum_time_consume += eavluation_result["time_consume"]
            # print("TIME CONSUME *** ", sum_time_consume/ time_consume_t)
            



if __name__ == '__main__':
    tf.app.run()
