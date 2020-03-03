import json
import networkx as nx
import os
import time
import platform
import random
import numpy as np


def INFO_LOG(info, isshow):
    if isshow:
        print "[%s]%s" % (time.strftime("%Y-%m-%d %X", time.localtime()), info)


class Loader(object):
    def __init__(self, flag):
        if platform.system() == 'Linux':
            self.path_init_file = path_init_file = '/home/cuizeyu/pythonfile/dynGCN/data/as/'
            self.path_file = path_file = './data/as/'
            self.embedding_path = embedding_path = '/home/cuizeyu/pythonfile/dynGCN/data/lap-as_train-npy/'
            # path_init_file = '/home/hufenyu/Documents/pythonfile/dataset/'
            # path_file = '/home/hufenyu/Documents/pythonfile/lightrnn_czy/data/'

        else:
            self.path_init_file = path_init_file = '/Users/czy_yente/PycharmProjects/dynGCN/data/as/'
            self.path_file = path_file = './data/as/'
            self.embedding_path = embedding_path = '/home/cuizeyu/pythonfile/dynGCN/data/lap-as_train-npy/'


        with open(path_init_file + "node2id.json", 'r') as f:
            self.node2id = node2id = json.load(f)

        # self.embedding_path = 'as_embedding/'
        self.start_graph = 673
        self.present_graph = 673
        all_file = filter(lambda x: x[-8:] == '.gpickle', os.listdir(path_init_file))
        num_file = [int(x.split('_')[1]) for x in all_file]
        self.final_graph = max(num_file)


        self.graph_now = self.load_graph(self.present_graph)
        print self.graph_now

    def load_graph(self, present_graph, flag="train"):
        # path_last = self.path_init_file + "month_" + str(present_graph - 1) + "_graph.gpickle"
        path_now = self.path_init_file + "month_" + str(present_graph) + "_graph.gpickle"
        # G_last = nx.read_gpickle(path_last)
        G_now = nx.read_gpickle(path_now)
        # dynG = self.graph_changes(G_last, G_now)
        # print dynG
        return G_now

    def last_embeddings(self):
        last_date = self.last_date(self.present_graph)

        with open(self.embedding_path + 'month_' + str(last_date) + '_graph_embedding.npy') as f:
            load_a = np.load(f)
        # when we generate *.npy [-1] is an zeros vector for placeholder, but now it is unnecessary to exist
        return load_a[0:-1]

    def adj(self):
        adjj = nx.to_numpy_matrix(self.graph_now)
        present_graph = self.present_graph
        last_date = self.last_date(self.present_graph)
        last_graph = self.load_graph(last_date)

        delta_adjj = nx.to_numpy_matrix(self.graph_changes(last_graph, self.graph_now))
        # rowsum = np.array(adj.sum(1))
        # d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        # d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        # d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        # return d_mat_inv_sqrt.transpose().dot(adj).dot(d_mat_inv_sqrt)

        return adjj, delta_adjj


    def graph_changes(self, G_last, G_now):
        G = nx.DiGraph()
        G.add_nodes_from(range(len(self.node2id)))

        for idx in range(len(self.node2id)):
            # add edge
            adding_list = list(set(G_now.adj[idx].keys()) - set(G_last.adj[idx].keys()))
            for jdx in adding_list:
                G.add_edge(idx, jdx, weight=G_now.adj[idx][jdx]['weight'])

            # delete edge
            deleting_list = list(set(G_last.adj[idx].keys()) - set(G_now.adj[idx].keys()))
            for jdx in deleting_list:
                G.add_edge(idx, jdx, weight=-G_last.adj[idx][jdx]['weight'])

            # change weight
            for jdx, weight_dict in G_now.adj[idx].items():
                if G_last.adj[idx].has_key(jdx):
                    if G_last.adj[idx][jdx]["weight"] != weight_dict["weight"]:
                        G.add_edge(
                            idx, jdx,
                            weight_dict["weight"] - G_last.adj[idx][jdx]["weight"]
                        )

        return G

    def change_2_next_graph_date(self):
        self.present_graph += 1
        if self.present_graph == 5:
            self.present_graph = 8
        # self.dynG = self.load_graph(self.present_graph)
        if self.present_graph > self.final_graph:
            self.present_graph = self.start_graph
        self.graph_now = self.load_graph(self.present_graph)

    def change_2_the_graph_date(self, date):
        self.present_graph = date
        if self.present_graph == 5:
            self.present_graph = 8
        # self.dynG = self.load_graph(self.present_graph)
        if self.present_graph > self.final_graph:
            self.present_graph = self.start_graph
        self.graph_now = self.load_graph(self.present_graph)

    def last_date(self, present_date):
        if present_date == 8:
            last_date = 4
        else:
            last_date = present_date - 1
        return last_date

    def notTHEend(self, present_graph, endinwhere):

        return present_graph < self.final_graph - endinwhere


    def generate_batch_data(self, batchsize, mode):
        path_now = self.path_init_file +str(mode) + "\month_" + \
                   str(self.present_graph) + "_graph.gpickle"
        # print "======load data for "+ str(mode) + "======"
        # print path_now

        dataset = self.graph_now

        idlist = self.node2id.values()

        if mode == "Valid":
            batchsize = batchsize / 2
        # because the valid will double the true and false
        node1_list = []
        node2_list = []
        negative_list = []
        # edge_num = dataset.number_of_edges()
        # batch_num = edge_num / batchsize
        batchid = 0
        t = 0
        edges = []
        if mode == "Train":
            edges = [e for e in dataset.edges() if not dataset[e[0]][e[1]]['valid']]
        elif mode == "Valid":
            edges = [e for e in dataset.edges() if dataset[e[0]][e[1]]['valid']]

        edges = [e for e in dataset.edges()]

        if len(edges) < batchsize:
            edges = random.sample(edges * batchsize, batchsize)

        edge_num = len(edges)
        batch_num = edge_num / batchsize
        # print batch_num
        # print edge_num
        negative_pool = filter(lambda x: len(self.graph_now.adj[x]) != 0, self.graph_now.adj.keys())
        for idx, (node1, node2) in enumerate(edges):
            if t < batchsize:
                node1_list.append(node1)
                node2_list.append(node2)
                negative = random.choice(negative_pool)
                while self.graph_now.adj[node1].has_key(negative):
                    negative = random.choice(negative_pool)
                negative_list.append(negative)
                t += 1
            # dataset.adj[]
            elif t == batchsize:
                t = 0
                batchid += 1
                yield batchid, batch_num, node1_list, node2_list, negative_list
                node1_list = []
                node2_list = []
                negative_list = []

        if t == batchsize:
            t = 0
            batchid += 1
            yield batchid, batch_num, node1_list, node2_list, negative_list

    def generate_batch_data_gcn(self, batchsize, mode):
        path_now = self.path_init_file +str(mode) + "\month_" + \
                   str(self.present_graph) + "_graph.gpickle"

        dataset = self.graph_now

        idlist = self.node2id.values()

        if mode == "Valid":
            batchsize = batchsize / 2
        # because the valid will double the true and false
        node1_list = []
        node2_list = []
        negative_list = []
        # edge_num = dataset.number_of_edges()
        # batch_num = edge_num / batchsize
        batchid = 0
        t = 0
        edges = []
        # if mode == "Train":
        #     edges = [e for e in dataset.edges() if not dataset[e[0]][e[1]]['valid']]
        # elif mode == "Valid":
        #     edges = [e for e in dataset.edges() if dataset[e[0]][e[1]]['valid']]

        edges = [e for e in dataset.edges()]

        if len(edges) < batchsize:
            edges = random.sample(edges * batchsize, batchsize)

        edge_num = len(edges)
        batch_num = edge_num / batchsize
        # print batch_num
        # print edge_num
        negative_pool = filter(lambda x: len(self.graph_now.adj[x]) != 0, self.graph_now.adj.keys())
        for idx, (node1, node2) in enumerate(edges):
            if t < batchsize:
                node1_list.append(node1)
                node2_list.append(node2)
                negative = random.choice(negative_pool)
                while self.graph_now.adj[node1].has_key(negative):
                    negative = random.choice(negative_pool)
                negative_list.append(negative)
                t += 1
            # dataset.adj[]
            elif t == batchsize:
                t = 0
                batchid += 1
                g = self.load_graph(self.present_graph)

                yield batchid, batch_num, node1_list, node2_list, negative_list,nx.to_numpy_matrix(g)
                node1_list = []
                node2_list = []
                negative_list = []

        if t == batchsize:
            t = 0
            batchid += 1
            g = self.load_graph(self.present_graph)
            yield batchid, batch_num, node1_list, node2_list, negative_list, nx.to_numpy_matrix(g)


if __name__ == "__main__":
    loader = Loader(1)
    nodelist = []
    edgelist = []

    for idx in range(888):
        # print(len(loader.graph_now.edges()))
        nodelist.append(len(loader.graph_now.nodes()))
        edgelist.append(len(loader.graph_now.edges()))
        if loader.present_graph == 713:
            print(len(loader.graph_now.nodes()), len(loader.graph_now.edges()))
        loader.change_2_next_graph_date()


    print("node", min(nodelist), max(nodelist))
    print("edge", min(edgelist), max(edgelist))

    # for batch in loader.generate_batch_data(batchsize=128, mode="Valid"):
    #     batch_id, batch_num, nodelist1, nodelist2, negative_list = batch
    #     print batch_num,
    #     print batch_id







