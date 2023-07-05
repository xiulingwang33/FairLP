import pickle
from preprocessing import mask_test_edges
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from FairLP import *
import pandas as pd

# Gplus, Facebook, DBLP
DATA = 'Facebook'
# random seed to perform train test split
SEED = 0

def qda_test(train_edges, train_edges_false, test_edges, test_edges_false, score_matrix):
    # train_format: uid0, uid1, gender0, gender1, prediction, truth, binary_prediction
    # test_format: uid0, uid1, gender0, gender1, prediction, truth, binary_prediction
    train_format = []

    pos_train = []
    pos = []
    for e in train_edges:
        pos_train.append(score_matrix[e[0], e[1]])  # predicted score for given edge
        pos.append(adj_sparse[e[0], e[1]])  # actual value (1)
        train_format.append(
            [e[0], e[1], g.nodes[e[0]]['gender'], g.nodes[e[1]]['gender'], score_matrix[e[0], e[1]],
             adj_sparse[e[0], e[1]]])

    neg_train = []
    neg = []
    for e in train_edges_false:
        neg_train.append(score_matrix[[0], e[1]])  # predicted score for given edge
        neg.append(adj_sparse[e[0], e[1]])  # actual value (0)
        train_format.append(
            [e[0], e[1], g.nodes[e[0]]['gender'], g.nodes[e[1]]['gender'], score_matrix[e[0], e[1]],
             adj_sparse[e[0], e[1]]])

    test_format_pos = []
    preds_pos = []
    labels_pos = []
    for e in test_edges:
        preds_pos.append(score_matrix[e[0], e[1]])  # predicted score
        labels_pos.append(1)
        test_format_pos.append(
            [e[0], e[1], g.nodes[e[0]]['gender'], g.nodes[e[1]]['gender'], score_matrix[e[0], e[1]],
             1])

    test_format_neg = []
    preds_neg = []
    labels_neg = []
    for e in test_edges_false:
        preds_neg.append(score_matrix[e[0], e[1]])  # predicted score
        labels_neg.append(0)
        test_format_neg.append(
            [e[0], e[1], g.nodes[e[0]]['gender'], g.nodes[e[1]]['gender'], score_matrix[e[0], e[1]],
             0])

    train = preds_pos + preds_neg
    train = np.array(train).reshape(-1, 1)
    label = labels_pos + labels_neg

    clf = QuadraticDiscriminantAnalysis()
    clf.fit(train, label)

    v = clf.predict(np.array(preds_pos).reshape(-1, 1))
    for i in range(0, len(v)):
        test_format_pos[i].append(v[i])

    v = clf.predict(np.array(preds_neg).reshape(-1, 1))
    for i in range(0, len(v)):
        test_format_neg[i].append(v[i])

    test_format = test_format_pos + test_format_neg
    # uid0, uid2, gender0, gender1, score, truth, prediction

    name = ['node1', 'node2', 'gender1', 'gender2', 'score', 'grt','biscore']
    result = pd.DataFrame(columns=name, data=test_format)
    result.to_csv("./results/{0}_{1}_{2}.csv".format(DATA, METHOD,TYPE))
    return train_format, test_format


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
            etype = (t1, t2)
        else:
            etype = (t2, t1)
        typelist.add(etype)
    # print(typelist)
    return typelist


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

typelist = get_edge_types(g.edges)

np.random.seed(SEED)  # make sure train-test split is consistent between notebooks
adj_sparse = nx.to_scipy_sparse_matrix(g)

# Perform train-test split
adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(
    adj_sparse, test_frac=.3, val_frac=.1)
g_train = nx.from_scipy_sparse_matrix(
    adj_train)  # new graph object with only non-hidden edges, keep all the original nodes

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
METHODS = ['upsample_50', 'upsample_70', 'upsample_90','downsample_50','downsample_70','downsample_90']
for METHOD in METHODS:
    print(METHOD)

    if METHOD == 'orig':
        g_copy = g_train.copy()
    else:
        # sample g_train
        g_copy = g_train.copy()
        count_edges, rates, diff_num = sample_2(g_copy, count_nodes, train_edges, train_edges_false, METHOD)

    # different similarity-based link prediction algorithms
    TYPE = 'AA'
    # Adamic-Adar
    aa_matrix = np.zeros(adj.shape)
    for u, v, p in nx.adamic_adar_index(g_copy):  # (u, v) = node indices, p = Adamic-Adar index
        aa_matrix[u][v] = p
        aa_matrix[v][u] = p  # make sure it's symmetric
        aa_matrix[v][u] = p  # make sure it's symmetric
    # Normalize array
    aa_matrix = aa_matrix / aa_matrix.max()

    train_format, test_format = qda_test(train_edges, train_edges_false, test_edges, test_edges_false, aa_matrix)
    # out = open('./results/%s-%s-%s-%d-train.txt' % (DATA, METHOD, TYPE, SEED), 'w')
    # for item in train_format:
    #     for jtem in item:
    #         out.write(str(jtem) + '\t')
    #     out.write('\n')
    # out.close()
    #
    # out = open('./results/%s-%s-%s-%d-test.txt' % (DATA, METHOD, TYPE, SEED), 'w')
    # for item in test_format:
    #     for jtem in item:
    #         out.write(str(jtem) + '\t')
    #     out.write('\n')
    # out.close()
    # # exit()

    TYPE = 'CN'
    # Common neighbors
    cn_matrix = np.zeros(adj.shape)


    def cn_predict(u, v):
        return len(list(nx.common_neighbors(g_copy, u, v)))


    def _apply_prediction(G, func):
        return ((u, v, func(u, v)) for u, v in nx.non_edges(G))


    def cn_neighbors(G):
        return _apply_prediction(G, cn_predict)


    for u, v, p in cn_neighbors(g_copy):  # (u, v) = node indices, p = Jaccard coefficient
        cn_matrix[u][v] = p
        cn_matrix[v][u] = p  # make sure it's symmetric
    # Normalize array
    cn_matrix = cn_matrix / cn_matrix.max()

    train_format, test_format = qda_test(train_edges, train_edges_false, test_edges, test_edges_false, cn_matrix)
    # out = open('./results/%s-%s-%s-%d-train.txt' % (DATA, METHOD, TYPE, SEED), 'w')
    # for item in train_format:
    #     for jtem in item:
    #         out.write(str(jtem) + '\t')
    #     out.write('\n')
    # out.close()
    #
    # out = open('./results/%s-%s-%s-%d-test.txt' % (DATA, METHOD, TYPE, SEED), 'w')
    # for item in test_format:
    #     for jtem in item:
    #         out.write(str(jtem) + '\t')
    #     out.write('\n')
    # out.close()

