import tensorflow as tf
import numpy as np
import argparse
from line.model import LINEModel
from line.utils import DBLPDataLoader
import pickle
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


def LINE(g_train, train_test_split,graph_file,DATASET,METHOD):
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', default=128)
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--K', default=5)
    parser.add_argument('--proximity', default='second-order', help='first-order or second-order')
    parser.add_argument('--learning_rate', default=0.025)
    parser.add_argument('--mode', default='train')
    parser.add_argument('--num_batches', default=8000)
    parser.add_argument('--total_graph', default=True)
    parser.add_argument('--graph_file', default='./')
    parser.add_argument('--edge_score_mode', default='edge-emb')
    parser.add_argument('--uid', default='0')
    parser.add_argument('--flag', default='weighted')
    args = parser.parse_args()
    #args.proximity='first-order'
    args.graph_file=graph_file
    args.uid = str(DATASET)
    args.flag=str(METHOD)
    #print(args.graph_file)
    if args.mode == 'train':
        normalized_embedding=train(args)
        data_loader = DBLPDataLoader(graph_file=args.graph_file)
        emb_list = []
        #print(np.shape(g_train)[0])
        for node_index in range(0, np.shape(g_train)[0]):
            node_str = str(node_index)
            node_emb = normalized_embedding[node_index]
            emb_list.append(node_emb)
        emb_matrix = np.vstack(emb_list)


        with open('./results/embeds/'+ DATASET + '-' + METHOD,
                  'w') as f:
            f.write('%d %d\n' % (np.shape(g_train)[0], args.embedding_dim))
            for i in range(np.shape(g_train)[0]):
                e = ' '.join(map(lambda x: str(x), emb_list[i]))
                f.write('%s %s\n' % (str(i), e))


        n2v_scores, val_edge_labels, val_preds, test_edge_labels, test_preds=linkpre_scores(args, emb_matrix, train_test_split, DATASET,METHOD)
        return n2v_scores, val_edge_labels, val_preds, test_edge_labels, test_preds



    elif args.mode == 'test':
        test(args)


def train(args):
    data_loader = DBLPDataLoader(graph_file=args.graph_file)
    suffix = args.proximity
    args.num_of_nodes = data_loader.num_of_nodes
    model = LINEModel(args)
    with tf.Session() as sess:
        print(args)
        print('batches\tloss\tsampling time\ttraining_time\tdatetime')
        tf.global_variables_initializer().run()
        initial_embedding = sess.run(model.embedding)
        learning_rate = args.learning_rate
        sampling_time, training_time = 0, 0
        for b in range(args.num_batches):
            t1 = time.time()
            u_i, u_j, label = data_loader.fetch_batch(batch_size=args.batch_size, K=args.K)
            feed_dict = {model.u_i: u_i, model.u_j: u_j, model.label: label, model.learning_rate: learning_rate}
            t2 = time.time()
            sampling_time += t2 - t1
            if b % 100 != 0:
                sess.run(model.train_op, feed_dict=feed_dict)
                training_time += time.time() - t2
                if learning_rate > args.learning_rate * 0.0001:
                    learning_rate = args.learning_rate * (1 - b / args.num_batches)
                else:
                    learning_rate = args.learning_rate * 0.0001
            else:
                loss = sess.run(model.loss, feed_dict=feed_dict)
                print('%d\t%f\t%0.2f\t%0.2f\t%s' % (b, loss, sampling_time, training_time,
                                                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
                sampling_time, training_time = 0, 0
            if b % 1000 == 0 or b == (args.num_batches - 1):
                embedding = sess.run(model.embedding)

                normalized_embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
                #pickle.dump(data_loader.embedding_mapping(normalized_embedding),open('data/embedding_%s_%s.pkl' % (args.uid,args.flag), 'wb'))
    return (embedding)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def linkpre_scores(args, emb_matrix, train_test_split,DATASET,METHOD):
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
    test_edges, test_edges_false = train_test_split
    start_time = time.time()
    # Generate bootstrapped edge embeddings (as is done in node2vec paper)
    # Edge embedding for (v1, v2) = hadamard product of node embeddings for v1, v2
    if args.edge_score_mode == "edge-emb":

        def get_edge_embeddings(edge_list,DATASET,METHOD,flag):
            embs = []
            for edge in edge_list:
                node1 = edge[0]
                node2 = edge[1]
                emb1 = emb_matrix[node1]
                emb2 = emb_matrix[node2]
                edge_emb = np.multiply(emb1, emb2)
                #edge_emb = np.array(emb1) + np.array(emb2)
                embs.append(list(edge_emb))
            embs = np.array(embs)

            return embs

        # Train-set edge embeddings
        pos_train_edge_embs = get_edge_embeddings(train_edges,DATASET, METHOD, flag='pos-train')
        neg_train_edge_embs = get_edge_embeddings(train_edges_false,DATASET,METHOD, flag='neg-train')
        train_edge_embs = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])

        # Create train-set edge labels: 1 = real edge, 0 = false edge
        train_edge_labels = np.concatenate([np.ones(len(train_edges)), np.zeros(len(train_edges_false))])

        # Val-set edge embeddings, labels
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            pos_val_edge_embs = get_edge_embeddings(val_edges,DATASET, METHOD, flag='pos-val')
            neg_val_edge_embs = get_edge_embeddings(val_edges_false,DATASET, METHOD, flag='neg-val')
            val_edge_embs = np.concatenate([pos_val_edge_embs, neg_val_edge_embs])
            val_edge_labels = np.concatenate([np.ones(len(val_edges)), np.zeros(len(val_edges_false))])

        # Test-set edge embeddings, labels
        pos_test_edge_embs = get_edge_embeddings(test_edges,DATASET, METHOD, flag='pos-test')
        neg_test_edge_embs = get_edge_embeddings(test_edges_false,DATASET, METHOD, flag='neg-test')
        test_edge_embs = np.concatenate([pos_test_edge_embs, neg_test_edge_embs])


        # Create val-set edge labels: 1 = real edge, 0 = false edge
        test_edge_labels = np.concatenate([np.ones(len(test_edges)), np.zeros(len(test_edges_false))])

        # Train logistic regression classifier on train-set edge embeddings
        edge_classifier = LogisticRegression(random_state=0)
        edge_classifier.fit(train_edge_embs, train_edge_labels)

        # Predicted edge scores: probability of being of class "1" (real edge)
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            val_preds = edge_classifier.predict_proba(val_edge_embs)[:, 1]
        test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]
        print(test_preds)
        print(np.shape(test_preds))

        runtime = time.time() - start_time

        # Calculate scores
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            n2v_val_roc = roc_auc_score(val_edge_labels, val_preds)
            # n2v_val_roc_curve = roc_curve(val_edge_labels, val_preds)
            n2v_val_ap = average_precision_score(val_edge_labels, val_preds)
        else:
            n2v_val_roc = None
            n2v_val_roc_curve = None
            n2v_val_ap = None

        n2v_test_roc = roc_auc_score(test_edge_labels, test_preds)
        # n2v_test_roc_curve = roc_curve(test_edge_labels, test_preds)
        n2v_test_ap = average_precision_score(test_edge_labels, test_preds)


    # Generate edge scores using simple dot product of node embeddings (like in GAE paper)
    elif args.edge_score_mode == "dot-product":
        score_matrix = np.dot(emb_matrix, emb_matrix.T)
        runtime = time.time() - start_time

        # Val set scores
        if len(val_edges) > 0:
            n2v_val_roc, n2v_val_ap = get_roc_score(val_edges, val_edges_false, score_matrix, apply_sigmoid=True)
        else:
            n2v_val_roc = None
            n2v_val_roc_curve = None
            n2v_val_ap = None

        # Test set scores
        n2v_test_roc, n2v_test_ap = get_roc_score(test_edges, test_edges_false, score_matrix, apply_sigmoid=True)

    else:
        print
        "Invalid edge_score_mode! Either use edge-emb or dot-product."

    # Record scores
    n2v_scores = {}

    n2v_scores['test_roc'] = n2v_test_roc
    # n2v_scores['test_roc_curve'] = n2v_test_roc_curve
    n2v_scores['test_ap'] = n2v_test_ap

    n2v_scores['val_roc'] = n2v_val_roc
    # n2v_scores['val_roc_curve'] = n2v_val_roc_curve
    n2v_scores['val_ap'] = n2v_val_ap

    n2v_scores['runtime'] = runtime

    return n2v_scores, val_edge_labels, val_preds, test_edge_labels, test_preds


# Input: positive test/val edges, negative test/val edges, edge score matrix
# Output: ROC AUC score, ROC Curve (FPR, TPR, Thresholds), AP score
def get_roc_score(edges_pos, edges_neg, score_matrix, apply_sigmoid=False):
    # Edge case
    if len(edges_pos) == 0 or len(edges_neg) == 0:
        return (None, None, None)

    # Store positive edge predictions, actual values
    preds_pos = []
    pos = []
    for edge in edges_pos:
        if apply_sigmoid == True:
            preds_pos.append(sigmoid(score_matrix[edge[0], edge[1]]))
        else:
            preds_pos.append(score_matrix[edge[0], edge[1]])
        pos.append(1)  # actual value (1 for positive)

    # Store negative edge predictions, actual values
    preds_neg = []
    neg = []
    for edge in edges_neg:
        if apply_sigmoid == True:
            preds_neg.append(sigmoid(score_matrix[edge[0], edge[1]]))
        else:
            preds_neg.append(score_matrix[edge[0], edge[1]])
        neg.append(0)  # actual value (0 for negative)

    # Calculate scores
    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    # roc_curve_tuple = roc_curve(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    # return roc_score, roc_curve_tuple, ap_score
    return roc_score, ap_score


# Return a list of tuples (node1, node2) for networkx link prediction evaluation
def get_ebunch(train_test_split):
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
    test_edges, test_edges_false = train_test_split

    test_edges_list = test_edges.tolist()  # convert to nested list
    test_edges_list = [tuple(node_pair) for node_pair in test_edges_list]  # convert node-pairs to tuples
    test_edges_false_list = test_edges_false.tolist()
    test_edges_false_list = [tuple(node_pair) for node_pair in test_edges_false_list]
    return (test_edges_list + test_edges_false_list)


def test(args):
    pass

#if __name__ == '__main__':
    #main()