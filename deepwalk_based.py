import pickle
import os
from preprocessing import mask_test_edges
import deepwalk.deepwalk as DW
import pandas as pd
import sys
from FairLP import *
sys.setrecursionlimit(1000000)
def get_edge_types(edges):
    typelist = set([])
    for edge in edges:
        # print(edge)
        # print(type(edge))
        e1 = edge[0]
        e2 = edge[1]
        t1 = g.nodes[e1]['gender']
        t2 = g.nodes[e2]['gender']
        if t1 <= t2:
            etype = (t1,t2)
        else:
            etype = (t2,t1)
        typelist.add(etype)
    # print(typelist)
    return typelist



def qda_test(train_edges, train_edges_false, val_edges, val_edges_false,test_edges, test_edges_false, val_preds, test_preds,f):
    # train_format: uid0, uid1, gender0, gender1, prediction, truth, binary_prediction
    # test_format: uid0, uid1, gender0, gender1, prediction, truth, binary_prediction
    train_format=[]
    val_format = []
    test_format = []


    train_all = np.vstack((train_edges, train_edges_false))
    val_all =np.vstack((val_edges, val_edges_false))
    test_all =np.vstack((test_edges, test_edges_false))
    print(train_all)
    print(np.shape(train_all))
    print(np.shape(train_edges))

    for i in range(np.shape(train_all)[0]):
        grt=1
        if i > np.shape(train_edges)[0]:
            grt=0
        print([train_all[i][1]])
        train_f=[train_all[i][0],train_all[i][1],g.nodes[train_all[i][0]]['gender'],g.nodes[train_all[i][1]]['gender'],grt,grt,grt]
        train_format.append(train_f)

    name = ['node1', 'node2', 'gender1', 'gender2', 'grt', 'score', 'biscore']
    result = pd.DataFrame(columns=name, data=train_format)
    result.to_csv("./results/train_{0}_{1}.csv".format(f,DATA))

    for i in range(np.shape(val_all)[0]):
        grt=1
        if i > np.shape(val_edges)[0]:
            grt=0
        if val_preds[i]>=0.5:
            bi_score=1
        else:
            bi_score=0
        val_f=[val_all[i][0],val_all[i][1],g.nodes[val_all[i][0]]['gender'],g.nodes[val_all[i][1]]['gender'],grt,val_preds[i], bi_score]
        val_format.append(val_f)

    name = ['node1', 'node2', 'gender1', 'gender2', 'grt', 'score', 'biscore']
    result = pd.DataFrame(columns=name, data=val_format)
    result.to_csv("./results/val_{0}_{1}.csv".format(f,DATA))

    for i in range(np.shape(test_all)[0]):
        grt=1
        if i > np.shape(test_edges)[0]:
            grt=0
        if test_preds[i]>=0.5:
            bi_score=1
        else:
            bi_score=0
        test_f=[test_all[i][0],test_all[i][1],g.nodes[test_all[i][0]]['gender'],g.nodes[test_all[i][1]]['gender'],grt,test_preds[i], bi_score]
        test_format.append(test_f)

    name = ['node1', 'node2', 'gender1', 'gender2', 'grt', 'score', 'biscore']
    result = pd.DataFrame(columns=name, data=test_format)
    result.to_csv("./results/test_{0}_{1}.csv".format(f,DATA))

    # uid0, uid2, gender0, gender1, score, truth, prediction
    return train_format,val_format, test_format

SEED=0

DATA = 'Facebook'

f = open('./data/%s_feature_map.txt'%DATA, 'r')
invert_index = eval(f.readline())
f.close()

network_dir = './data/{0}-adj-feat.pkl'.format(DATA)

with open(network_dir, 'rb') as f:
    adj, features = pickle.load(f, encoding='latin1')

g = nx.Graph(adj)

# for different dataset, the gender attribute has different anonymized value, and different index
if DATA == 'Gplus':
    g1_index = invert_index[('gender', '1')]
    g2_index = invert_index[('gender', '2')]
elif DATA == 'Facebook':
    g1_index = invert_index[('gender', 77)]
    g2_index = invert_index[('gender', 78)]
elif DATA == 'DBLP':
    g1_index = invert_index[('gender', 1)]
    g2_index = invert_index[('gender', 2)]
else:
    print('error!')
    exit()

count_nodes = [0, 0]
for i, n in enumerate(g.nodes()):
    if features[n][g1_index] == 1:
        ginfo = 1
        count_nodes[0] += 1
    elif features[n][g2_index] == 1:
        ginfo = 2
        count_nodes[1] += 1
    else:
        print('error')
        print(features[n][g1_index], features[n][g2_index])
        exit()
    g.nodes[n]['gender'] = ginfo
node_list=g.nodes()

typelist = get_edge_types(g.edges)

np.random.seed(SEED)  # make sure train-test split is consistent between notebooks
adj_sparse = nx.to_scipy_sparse_matrix(g)

# Perform train-test split
train_test_split=mask_test_edges(adj_sparse, test_frac=.3, val_frac=.1)
adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = train_test_split
g_train = nx.from_scipy_sparse_matrix(adj_train)  # new graph object with only non-hidden edges, keep all the original nodes

# add gender information to g_train
for i, n in enumerate(g_train.nodes()):
    if features[n][g1_index] == 1:
        ginfo = 1
    elif features[n][g2_index] == 1:
        ginfo = 2
    else:
        print('error')
        print(features[n][g1_index], features[n][g2_index])
        exit()
    g_train.nodes[n]['gender'] = ginfo

# orig: no pre-processing; avg, max, min: AvgD, MaxD, MinD; ps, nd: baselines
METHODS = ['orig', 'avg', 'max', 'min', 'ps','nd-avg','nd-max','nd-min']
for METHOD in METHODS:
    print(METHOD)

    if METHOD == 'orig':
        g_copy = g_train.copy()
    else:
        # sample g_train
        g_copy = g_train.copy()
        count_edges, rates, diff_num = sample(g_copy, count_nodes, train_edges, train_edges_false, METHOD)

    res_dir = './results'

    # for edge in train_edges:
    edge_tuples0 = [(min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in g_train.edges()]
    for nd in node_list:
        edge_tuples0.append((nd, nd))
    #train_edges0 = set(edge_tuples0)  # initialize train_edges to have all edges
    train_edges1 = np.array([list(edge_tuple) for edge_tuple in set(edge_tuples0)])


    train_list_file='%s/edgelist-%s-fair-%s-train.txt' % (res_dir, DATA, METHOD)
    out = open(train_list_file, 'w')
    for item in train_edges1:
        for jtem in item:
            out.write(str(jtem) + '\t')
        out.write('\n')
    out.close()

    n2v_scores, val_edge_labels, val_preds, test_edge_labels, test_preds = DW.deepwalk(g_copy, train_test_split,
                                                                                       DATA, train_list_file,METHOD)

    train_format, val_format, test_format = qda_test(train_edges, train_edges_false, val_edges, val_edges_false,
                                                     test_edges, test_edges_false, val_preds, test_preds, METHOD)

