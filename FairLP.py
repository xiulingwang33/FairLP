from __future__ import division
import random
import numpy as np
import math
import networkx as nx

def max1(v):
    return (v * (v - 1)) / 2


def max2(u, v):
    return u * v/2

# sample g_train
def sample(g, count_nodes, train_edges, train_edges_false, method):
    def get_rates():
        mm1 = 0
        mm0 = 0
        mf1 = 0
        mf0 = 0
        ff1 = 0
        ff0 = 0
        mm1_set = []
        mm0_set = []
        mf1_set = []
        mf0_set = []
        ff1_set = []
        ff0_set = []
        for edge in train_edges:
            e1 = edge[0]
            e2 = edge[1]
            t1 = g.nodes[e1]['gender']
            t2 = g.nodes[e2]['gender']
            if t1 <= t2:
                etype = (t1, t2)
            else:
                etype = (t2, t1)
            if etype == (1, 1):
                mm1 += 1
                mm1_set.append(tuple(edge))
            elif etype == (1, 2):
                mf1 += 1
                mf1_set.append(tuple(edge))
            elif etype == (2, 2):
                ff1 += 1
                ff1_set.append(tuple(edge))
        for edge in train_edges_false:
            e1 = edge[0]
            e2 = edge[1]
            t1 = g.nodes[e1]['gender']
            t2 = g.nodes[e2]['gender']
            if t1 <= t2:
                etype = (t1, t2)
            else:
                etype = (t2, t1)
            if etype == (1, 1):
                mm0 += 1
                mm0_set.append(tuple(edge))
            elif etype == (1, 2):
                mf0 += 1
                mf0_set.append(tuple(edge))
            elif etype == (2, 2):
                ff0 += 1
                ff0_set.append(tuple(edge))
        if method == 'avg' or method == 'max' or method =='min' or method == 'nd-max' or method == 'nd-avg' or method == 'nd-min':
            count_edges = [len(mm1_set), len(mf1_set), len(ff1_set)]
            # print(count_edges)
            possible_max = [max1(count_nodes[0]),max2(count_nodes[0],count_nodes[1]),max1(count_nodes[1])]
            # inter, intra
            rates = [count_edges[1] / possible_max[1], (count_edges[0] + count_edges[2]) / (possible_max[0] + possible_max[2])]
            # print(rates)
            if method == 'avg' or method == 'nd-avg':
                rate = np.mean(rates)
            elif method == 'max' or method == 'nd-max':
                rate = max(rates)
            elif method == 'min' or method == 'nd-min':
                rate = min(rates)
            diff_num = [rate * possible_max[1] - count_edges[1], rate * (possible_max[0] + possible_max[2]) - (count_edges[0] + count_edges[2])]
            # print(diff_num)
            pos_sets = [mf1_set, mm1_set + ff1_set]
            neg_sets = [mf0_set, mm0_set + ff0_set]
        else:
            if method == 'ps':
                count_edges = [mm1 + ff1, mm0 + ff0, mf1, mf0]
                # print(count_edges)
                size = len(train_edges) + len(train_edges_false)
                intra1_rate = ((mm1 + mm0 + ff1 + ff0) * (mm1 + mf1 + ff1)) / (size * (mm1 + ff1))
                intra0_rate = ((mm1 + mm0 + ff1 + ff0) * (mm0 + mf0 + ff0)) / (size * (mm0 + ff0))
                inter1_rate = ((mf1 + mf0) * (mm1 + mf1 + ff1)) / (size * mf1)
                inter0_rate = ((mf1 + mf0) * (mm0 + mf0 + ff0)) / (size * mf0)
                rates = [intra1_rate, intra0_rate, inter1_rate, inter0_rate]
                # print(rates)
                diff_num = [count_edges[i] * rates[i] - count_edges[i] for i in range(len(rates))]
                # print(diff_num)
                pos_sets = [mm1_set + ff1_set, mf1_set]
                neg_sets = [mm0_set + ff0_set, mf0_set]
            else:
                print('error!')
                print(method)
                exit()
        return diff_num, pos_sets, neg_sets, count_edges, rates

    diff_num, pos_sets, neg_sets, count_edges, rates = get_rates()

    if method == 'ps':
        for i in range(len(pos_sets)):
            p = i * 2
            pos = pos_sets[i]
            neg = neg_sets[i]
            dynamic_samplingv(g, diff_num[p], pos, neg, method)
            dynamic_samplingv(g, diff_num[p + 1], pos, neg, method)
    else:
        for i in range(len(pos_sets)):
            pos = pos_sets[i]
            neg = neg_sets[i]
            dynamic_samplingv(g, diff_num[i], pos, neg, method)

    return count_edges, rates, diff_num

# find the edge of the minimum weight
def find_min(edgeweight):
    mine = (0,0)
    minv = math.inf
    for k in edgeweight:
        if edgeweight[k] < minv:
            minv = edgeweight[k]
            mine = k
    return mine, minv

def edge_weight(g, set, method):
    edgeweight = {}
    for edge in set:
        e1 = edge[0]
        e2 = edge[1]
        # node degree based weight scheme
        if method == 'nd-max' or method == 'nd-avg' or method == 'nd-min':
            c = list(nx.common_neighbors(g, e1, e2))
            w = len(c)
        # common neighbor based weight scheme
        else:
            d1 = g.degree[e1] + 1
            d2 = g.degree[e2] + 1
            w1 = math.log(d1)
            w2 = math.log(d2)
            w = (w1 + w2) / 2
        edgeweight[tuple(edge)] = w
    return edgeweight

def update_weight(g, edgeweight, e1, e2, method):
    if method == 'nd-max' or method == 'nd-avg' or method == 'nd-min':
        c = list(nx.common_neighbors(g, e1, e2))
        w = len(c)
    else:
        d1 = g.degree[e1] + 1
        d2 = g.degree[e2] + 1
        w1 = math.log(d1)
        w2 = math.log(d2)
        w = (w1 + w2) / 2
    edgeweight[(e1, e2)] = w

def dynamic_samplingv(g, n, set1, set0, method):
    node_set = set(g.nodes)
    # make sure n is an int
    n = int(n)
    if abs(n) >= 1:
        # add edge from negative set
        if n >= 1:
            addition_list = []
            edgeweight = edge_weight(g, set0, method)
        # remove edge from positive set
        else:
            deletion_list = []
            edgeweight = edge_weight(g, set1, method)
        # print(edgeweight)
        for i in range(abs(n)):
            if len(edgeweight):
                edge, weight = find_min(edgeweight)
                e1 = edge[0]
                e2 = edge[1]
                if n >= 1:
                    g.add_edge(e1,e2)
                    addition_list.append(edge)
                elif n <= -1 and edge in g.edges:
                    g.remove_edge(e1,e2)
                    deletion_list.append(edge)
                del edgeweight[edge]
                # add edge from NEGATIVE set
                if n >= 1:
                    e1_list = node_set - set(nx.neighbors(g, e1))
                    e2_list = node_set - set(nx.neighbors(g, e2))
                # delete edge from positive set
                elif n <= -1:
                    e1_list = nx.neighbors(g, e1)
                    e2_list = nx.neighbors(g, e2)
                for j in e1_list:
                    if (e1,j) in edgeweight:
                        update_weight(g, edgeweight, e1, j, method)
                    elif (j,e1) in edgeweight:
                        update_weight(g, edgeweight, j, e1, method)
                for k in e2_list:
                    if (e2,k) in edgeweight:
                        update_weight(g, edgeweight, e2, k, method)
                    elif (k,e2) in edgeweight:
                        update_weight(g, edgeweight, k, e2, method)


# sample g_train
def sample_2(g, count_nodes, train_edges, train_edges_false, method):
    def get_rates():
        mm1 = 0
        mm0 = 0
        mf1 = 0
        mf0 = 0
        ff1 = 0
        ff0 = 0
        mm1_set = []
        mm0_set = []
        mf1_set = []
        mf0_set = []
        ff1_set = []
        ff0_set = []
        for edge in train_edges:
            e1 = edge[0]
            e2 = edge[1]
            t1 = g.nodes[e1]['gender']
            t2 = g.nodes[e2]['gender']
            if t1 <= t2:
                etype = (t1, t2)
            else:
                etype = (t2, t1)
            if etype == (1, 1):
                mm1 += 1
                mm1_set.append(tuple(edge))
            elif etype == (1, 2):
                mf1 += 1
                mf1_set.append(tuple(edge))
            elif etype == (2, 2):
                ff1 += 1
                ff1_set.append(tuple(edge))
        for edge in train_edges_false:
            e1 = edge[0]
            e2 = edge[1]
            t1 = g.nodes[e1]['gender']
            t2 = g.nodes[e2]['gender']
            if t1 <= t2:
                etype = (t1, t2)
            else:
                etype = (t2, t1)
            if etype == (1, 1):
                mm0 += 1
                mm0_set.append(tuple(edge))
            elif etype == (1, 2):
                mf0 += 1
                mf0_set.append(tuple(edge))
            elif etype == (2, 2):
                ff0 += 1
                ff0_set.append(tuple(edge))


        if method == 'upsample_50' or method == 'upsample_70' or method == 'upsample_90'or method == 'downsample_50' or method == 'downsample_70'or method == 'downsample_90':

            count_edges = [len(mm1_set), len(mf1_set), len(ff1_set)]
            print(count_edges)
            possible_max = [max1(count_nodes[0]), max2(count_nodes[0], count_nodes[1]), max1(count_nodes[1])]
            # inter, intra
            rates = [count_edges[1] / possible_max[1],
                     (count_edges[0] + count_edges[2]) / (possible_max[0] + possible_max[2])]
            print(rates)
            exit()
            if rates[0]>=rates[1]:
                print('err!')

            if method == 'upsample_10':
                diff_num=[int((rates[1]*1.1-rates[0])*possible_max[1]),0]

            if method == 'upsample_30':
                diff_num=[int((rates[1]*1.3-rates[0])*possible_max[1]),0]

            if method == 'upsample_50':
                diff_num = [int((rates[1] * 1.5 - rates[0]) * possible_max[1]), 0]

            if method == 'upsample_70':
                diff_num = [int((rates[1] * 1.7 - rates[0]) * possible_max[1]), 0]
            if method == 'upsample_90':
                diff_num = [int((rates[1] * 1.7 - rates[0]) * possible_max[1]), 0]

            if method == 'downsample_30':
                diff_num=[0,int((rates[0]*0.7-rates[1])*(possible_max[0] + possible_max[2]))]

            if method == 'downsample_10':
                diff_num=[0,int((rates[0]*0.9-rates[1])*(possible_max[0] + possible_max[2]))]


            if method == 'downsample_50':
                diff_num=[0,int((rates[0]*0.5-rates[1])*(possible_max[0] + possible_max[2]))]

            if method == 'downsample_70':
                diff_num=[0,int((rates[0]*0.3-rates[1])*(possible_max[0] + possible_max[2]))]

            if method == 'downsample_90':
                diff_num=[0,int((rates[0]*0.1-rates[1])*(possible_max[0] + possible_max[2]))]

            pos_sets = [mf1_set, mm1_set + ff1_set]
            neg_sets = [mf0_set, mm0_set + ff0_set]

        return diff_num, pos_sets, neg_sets, count_edges, rates

    diff_num, pos_sets, neg_sets, count_edges, rates = get_rates()


    for i in range(len(pos_sets)):
        pos = pos_sets[i]
        neg = neg_sets[i]
        dynamic_samplingv(g, diff_num[i], pos, neg, method)

    return count_edges, rates, diff_num




