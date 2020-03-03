import json
import networkx as nx
import os
import time
import platform
import random


def INFO_LOG(info, isshow):
    if isshow:
        print "[%s]%s" % (time.strftime("%Y-%m-%d %X", time.localtime()), info)


class Loader(object):
    def __init__(self, flag):
        if platform.system() == 'Linux':
            self.path_init_file = path_init_file = '/home/cuizeyu/pythonfile/dynGCN/data/data_hep/'
            self.path_file = path_file = './data/data_hep/'
            # path_init_file = '/home/hufenyu/Documents/pythonfile/dataset/'
            # path_file = '/home/hufenyu/Documents/pythonfile/lightrnn_czy/data/'

        else:
            self.path_init_file = path_init_file = '/Users/czy_yente/PycharmProjects/dynGCN/data/data_hep/'
            self.path_file = path_file = './data/data_hep/'

        with open(path_init_file + "node2id.json", 'r') as f:
            self.node2id = node2id = json.load(f)

        self.embedding_path = 'hep_train_embedding/'
        self.present_graph = 2
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
        # if self.present_graph == 5:
        #     self.present_graph = 8
        # self.dynG = self.load_graph(self.present_graph)
        if self.present_graph > self.final_graph:
            self.present_graph = 2
        self .graph_now = self.load_graph(self.present_graph)

    def change_2_the_graph_date(self, date):
        self.present_graph = date
        # if self.present_graph == 5:
        #     self.present_graph = 8
        # self.dynG = self.load_graph(self.present_graph)
        if self.present_graph > self.final_graph:
            self.present_graph = 2
        self .graph_now = self.load_graph(self.present_graph)


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



if __name__ == "__main__":
    loader = Loader(1)

    for batch in loader.generate_batch_data(batchsize=128, mode="Valid"):
        batch_id, batch_num, nodelist1, nodelist2, negative_list = batch
        print batch_num,
        print batch_id







