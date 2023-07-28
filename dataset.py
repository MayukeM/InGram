from utils import *
import numpy as np
import random
import igraph
import copy
import time
import os


class TrainData():
    def __init__(self, path):
        self.path = path  # 数据集路径
        self.rel_info = {}  # 关系信息
        self.pair_info = {}  # 实体对信息
        self.spanning = []
        self.remaining = []
        self.ent2id = None  # 实体到id的映射
        self.rel2id = None  # 关系到id的映射
        self.id2ent, self.id2rel, self.triplets = self.read_triplet(path + 'train.txt')  # 读取训练集
        self.num_triplets = len(self.triplets)  # 训练集三元组数量
        self.num_ent, self.num_rel = len(self.id2ent), len(self.id2rel)  # 实体数量，关系数量

    def read_triplet(self, path):
        id2ent, id2rel, triplets = [], [], []
        with open(path, 'r') as f:  # 读取训练集
            for line in f.readlines():
                h, r, t = line.strip().split('\t')
                id2ent.append(h)  # 实体集
                id2ent.append(t)  # 实体集
                id2rel.append(r)  # 关系集
                triplets.append((h, r, t))  # 三元组集
        id2ent = remove_duplicate(id2ent)  # 去重
        id2rel = remove_duplicate(id2rel)
        self.ent2id = {ent: idx for idx, ent in enumerate(id2ent)}
        self.rel2id = {rel: idx for idx, rel in enumerate(id2rel)}
        triplets = [(self.ent2id[h], self.rel2id[r], self.ent2id[t]) for h, r, t in triplets]
        for (h, r, t) in triplets:
            if (h, t) in self.rel_info:
                self.rel_info[(h, t)].append(r)
            else:
                self.rel_info[(h, t)] = [r]
            if r in self.pair_info:
                self.pair_info[r].append((h, t))
            else:
                self.pair_info[r] = [(h, t)]
        G = igraph.Graph.TupleList(np.array(triplets)[:, 0::2])  # 构建图，以实体为节点，以关系为边，无向图
        G_ent = igraph.Graph.TupleList(np.array(triplets)[:, 0::2], directed=True)  # 有向图
        spanning = G_ent.spanning_tree()  # 最小生成树，作用是将图中所有节点连接起来，且边权重和最小
        G_ent.delete_edges(spanning.get_edgelist())  # 删除最小生成树的边，为了得到剩余的边

        for e in spanning.es:  # 将最小生成树的边加入spanning
            e1, e2 = e.tuple  # e1,e2为边的两个节点
            e1 = spanning.vs[e1]["name"]  # e1为边的第一个节点
            e2 = spanning.vs[e2]["name"]  # e2为边的第二个节点
            self.spanning.append((e1, e2))  # 将边加入spanning

        spanning_set = set(self.spanning)

        print("-----Train Data Statistics-----")
        print(f"{len(self.ent2id)} entities, {len(self.rel2id)} relations")  # 实体数量，关系数量
        print(f"{len(triplets)} triplets")
        self.triplet2idx = {triplet: idx for idx, triplet in enumerate(triplets)}
        self.triplets_with_inv = np.array([(t, r + len(id2rel), h) for h, r, t in triplets] + triplets)  # 加入逆关系
        return id2ent, id2rel, triplets

    def split_transductive(self, p):  # p为验证集比例，即验证集占训练集的比例
        msg, sup = [], []

        rels_encountered = np.zeros(self.num_rel)
        remaining_triplet_indexes = np.ones(self.num_triplets)

        for h, t in self.spanning:
            r = random.choice(self.rel_info[(h, t)])
            msg.append((h, r, t))
            remaining_triplet_indexes[self.triplet2idx[(h, r, t)]] = 0
            rels_encountered[r] = 1

        for r in (1 - rels_encountered).nonzero()[0].tolist():  # nonzero()返回非零元素的索引，tolist()将数组或矩阵转换成列表
            h, t = random.choice(self.pair_info[int(r)])
            msg.append((h, r, t))
            remaining_triplet_indexes[self.triplet2idx[(h, r, t)]] = 0

        start = time.time()
        sup = [self.triplets[idx] for idx, tf in enumerate(remaining_triplet_indexes) if tf]

        msg = np.array(msg)
        random.shuffle(sup)
        sup = np.array(sup)
        add_num = max(int(self.num_triplets * p) - len(msg), 0)
        msg = np.concatenate([msg, sup[:add_num]])
        sup = sup[add_num:]

        msg_inv = np.fliplr(msg).copy()
        msg_inv[:, 1] += self.num_rel
        msg = np.concatenate([msg, msg_inv])

        return msg, sup


class TestNewData():
    def __init__(self, path, data_type="valid"):
        self.path = path
        self.data_type = data_type
        self.ent2id = None
        self.rel2id = None
        self.id2ent, self.id2rel, self.msg_triplets, self.sup_triplets, self.filter_dict = self.read_triplet()
        self.num_ent, self.num_rel = len(self.id2ent), len(self.id2rel)  # 实体数量，关系数量

    def read_triplet(self):
        id2ent, id2rel, msg_triplets, sup_triplets = [], [], [], []
        total_triplets = []

        with open(self.path + "msg.txt", 'r') as f:  # msg
            for line in f.readlines():
                h, r, t = line.strip().split('\t')
                id2ent.append(h)
                id2ent.append(t)
                id2rel.append(r)
                msg_triplets.append((h, r, t))
                total_triplets.append((h, r, t))

        id2ent = remove_duplicate(id2ent)
        id2rel = remove_duplicate(id2rel)
        self.ent2id = {ent: idx for idx, ent in enumerate(id2ent)}
        self.rel2id = {rel: idx for idx, rel in enumerate(id2rel)}
        num_rel = len(self.rel2id)
        msg_triplets = [(self.ent2id[h], self.rel2id[r], self.ent2id[t]) for h, r, t in msg_triplets]
        msg_inv_triplets = [(t, r + num_rel, h) for h, r, t in msg_triplets]

        with open(self.path + self.data_type + ".txt", 'r') as f:
            for line in f.readlines():
                h, r, t = line.strip().split('\t')
                sup_triplets.append((self.ent2id[h], self.rel2id[r], self.ent2id[t]))
                assert (self.ent2id[h], self.rel2id[r], self.ent2id[t]) not in msg_triplets, \
                    (self.ent2id[h], self.rel2id[r], self.ent2id[t])
                total_triplets.append((h, r, t))
        for data_type in ['valid', 'test']:
            if data_type == self.data_type:
                continue
            with open(self.path + data_type + ".txt", 'r') as f:
                for line in f.readlines():
                    h, r, t = line.strip().split('\t')
                    assert (self.ent2id[h], self.rel2id[r], self.ent2id[t]) not in msg_triplets, \
                        (self.ent2id[h], self.rel2id[r], self.ent2id[t])
                    total_triplets.append((h, r, t))

        filter_dict = {}
        for triplet in total_triplets:
            h, r, t = triplet
            if ('_', self.rel2id[r], self.ent2id[t]) not in filter_dict:
                filter_dict[('_', self.rel2id[r], self.ent2id[t])] = [self.ent2id[h]]
            else:
                filter_dict[('_', self.rel2id[r], self.ent2id[t])].append(self.ent2id[h])

            if (self.ent2id[h], '_', self.ent2id[t]) not in filter_dict:
                filter_dict[(self.ent2id[h], '_', self.ent2id[t])] = [self.rel2id[r]]
            else:
                filter_dict[(self.ent2id[h], '_', self.ent2id[t])].append(self.rel2id[r])

            if (self.ent2id[h], self.rel2id[r], '_') not in filter_dict:
                filter_dict[(self.ent2id[h], self.rel2id[r], '_')] = [self.ent2id[t]]
            else:
                filter_dict[(self.ent2id[h], self.rel2id[r], '_')].append(self.ent2id[t])

        print(f"-----{self.data_type.capitalize()} Data Statistics-----")  # capitalize()将字符串的第一个字母变成大写
        print(f"Message set has {len(msg_triplets)} triplets")  # len()返回列表元素个数
        print(f"Supervision set has {len(sup_triplets)} triplets")  # 这里是指有多少个三元组是有监督的
        print(f"{len(self.ent2id)} entities, " + \
              f"{len(self.rel2id)} relations, " + \
              f"{len(total_triplets)} triplets")

        msg_triplets = msg_triplets + msg_inv_triplets  #  msg_inv_triplets是msg_triplets的逆关系

        return id2ent, id2rel, np.array(msg_triplets), np.array(sup_triplets), filter_dict
