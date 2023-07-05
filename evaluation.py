import numpy as np
# import cvxopt as cvx
# from cvxopt import spmatrix, matrix, solvers
from scipy.sparse import coo_matrix
import sys
import networkx as nx
import pickle as pk
import pandas as pd
import csv
from sklearn.metrics import confusion_matrix,roc_auc_score, average_precision_score, accuracy_score, recall_score, precision_score, \
    f1_score
# from sklearn.metrics import  confusion_matrix,roc_auc_score, average_precision_score, accuracy_score, recall_score, precision_score, \
#     f1_score


# ratio=3
# datasets=['DBLP','Facebook','Gplus']
datasets=['DBLP']
# METHODS = ['upsample_10', 'upsample_30','downsample_10','downsample_30','upsample_50', 'upsample_70', 'upsample_90','downsample_50','downsample_70','downsample_90']
METHODS = ['downsample_10','downsample_30','downsample_50','downsample_70','downsample_90']

TYPES = ['AA','CN']
file_path='/Users/xiulingwang/results/'
exps=[]
tsts = []
for dt in datasets:
    for tp in TYPES:
        for md in METHODS:

            fname=dt+'_'+md+'_'+tp+'.csv'

            print(fname)

            exps.append(fname)
            all = []
            grd = []
            prd = []
            prob = []

            pred = []

            cnt = 0
            with open(file_path + fname) as csvfile:
                csv_reader = csv.reader(csvfile)
                # print(csv_reader)
                result_header = next(csv_reader)
                mm = []
                fm = []
                ff = []
                protect = []
                unprotect = []

                for row in csv_reader:
                    tst = []

                    node1 = int(float(row[1]))
                    node2 = int(float(row[2]))
                    gd1= int(float(row[3]))
                    gd2= int(float(row[4]))
                    grd = int(float(row[6]))
                    prd = int(float(row[7]))
                    score = float(row[5])
                    cnt += 1

                    if gd1 + gd2 == 2:
                        mm.append([prd, grd, score])
                        unprotect.append([prd, grd, score])
                        # print('%%%')


                    elif gd1 + gd2 == 3:
                        fm.append([prd, grd, score])
                        protect.append([prd, grd, score])
                        # print('---')

                    elif gd1 + gd2 == 4:
                        ff.append([prd, grd, score])
                        unprotect.append([prd, grd, score])
                        # print('###')

                    else:
                        print('***')
                        exit()

            protect = np.array(protect)
            unprotect = np.array(unprotect)

            protect_pred_labels = protect[:, 0]
            unprotect_pred_labels = unprotect[:, 0]

            protect_grd_labels = protect[:, 1]
            unprotect_grd_labels = unprotect[:, 1]

            protect_score = protect[:, 2]
            unprotect_score = unprotect[:, 2]
            # Calculate scores

            tn1, fp1, fn1, tp1 = confusion_matrix(protect_grd_labels, protect_pred_labels).ravel()
            print(tn1, fp1, fn1, tp1)
            tn2, fp2, fn2, tp2 = confusion_matrix(unprotect_grd_labels, unprotect_pred_labels).ravel()
            print(tn2, fp2, fn2, tp2)

            # tpd
            tpd = tp1/len(protect_grd_labels) - tp2/len(unprotect_grd_labels)
            # td
            td = (fp1 + tp1)/len(protect_grd_labels)-(fp2 + tp2)/len(unprotect_grd_labels)

            tsts.append([fname, td, tpd])

            # acc1 = accuracy_score(protect_grd_labels , protect_pred_labels)
            # recall1 = recall_score(protect_grd_labels , protect_pred_labels)
            # precision1 = precision_score(protect_grd_labels , protect_pred_labels)
            # f11 = f1_score(protect_grd_labels , protect_pred_labels)
            # roc_score1 = roc_auc_score(protect_grd_labels , protect_score)
            # ap_score1 = average_precision_score(protect_grd_labels , protect_score)
            #
            # acc2 = accuracy_score(unprotect_grd_labels, unprotect_pred_labels)
            # recall2 = recall_score(unprotect_grd_labels, unprotect_pred_labels)
            # precision2 = precision_score(unprotect_grd_labels, unprotect_pred_labels)
            # f12 = f1_score(unprotect_grd_labels, unprotect_pred_labels)
            # roc_score2 = roc_auc_score(unprotect_grd_labels, unprotect_score)
            # ap_score2 = average_precision_score(unprotect_grd_labels, unprotect_score)

name = ['name', 'td', 'tpd']
for tst in tsts:
    print(tst)

result = pd.DataFrame(columns=name, data=tsts)
result.to_csv("./results/all_bias.csv")

# # ratio=3
# # datasets=['DBLP','Facebook','Gplus']
# datasets=['DBLP']
# METHODS = ['upsample_10', 'upsample_30','downsample_10','downsample_30','upsample_50', 'upsample_70', 'upsample_90','downsample_50','downsample_70','downsample_90']
# TYPES = ['AA','CN']
# file_path='/Users/xiulingwang/results/'
# exps=[]
# tsts = []
# for dt in datasets:
#     for md in METHODS:
#         for tp in TYPES:
#             fname=dt+'_'+md+'_'+tp+'.csv'
#
#             print(fname)
#
#             exps.append(fname)
#             all = []
#             grd = []
#             prd = []
#             prob = []
#
#             pred = []
#
#             cnt = 0
#             with open(file_path + fname) as csvfile:
#                 csv_reader = csv.reader(csvfile)
#                 # print(csv_reader)
#                 result_header = next(csv_reader)
#                 mm = []
#                 fm = []
#                 ff = []
#                 protect = []
#                 unprotect1 = []
#                 unprotect2 = []
#
#                 for row in csv_reader:
#                     tst = []
#
#                     node1 = int(float(row[1]))
#                     node2 = int(float(row[2]))
#                     gd1= int(float(row[3]))
#                     gd2= int(float(row[4]))
#                     grd = int(float(row[6]))
#                     prd = int(float(row[7]))
#                     score = float(row[5])
#                     cnt += 1
#
#                     if gd1 + gd2 == 2:
#                         mm.append([prd, grd, score])
#                         unprotect1.append([prd, grd, score])
#                         # print('%%%')
#
#
#                     elif gd1 + gd2 == 3:
#                         fm.append([prd, grd, score])
#                         protect.append([prd, grd, score])
#                         # print('---')
#
#                     elif gd1 + gd2 == 4:
#                         ff.append([prd, grd, score])
#                         unprotect2.append([prd, grd, score])
#                         # print('###')
#
#                     else:
#                         print('***')
#                         exit()
#
#             protect = np.array(protect)
#             unprotect1 = np.array(unprotect1)
#             unprotect2 = np.array(unprotect2)
#
#             protect_pred_labels = protect[:, 0]
#             unprotect_pred_labels1 = unprotect1[:, 0]
#             unprotect_pred_labels2 = unprotect2[:, 0]
#
#             protect_grd_labels = protect[:, 1]
#             unprotect_grd_labels1 = unprotect1[:, 1]
#             unprotect_grd_labels2 = unprotect2[:, 1]
#
#             protect_score = protect[:, 2]
#             unprotect_score1 = unprotect1[:, 2]
#             unprotect_score12 = unprotect2[:, 2]
#
#             # Calculate scores
#
#             tn1, fp1, fn1, tp1 = confusion_matrix(protect_grd_labels, protect_pred_labels).ravel()
#             print(tn1, fp1, fn1, tp1)
#             tn2, fp2, fn2, tp2 = confusion_matrix(unprotect_grd_labels1, unprotect_pred_labels1).ravel()
#             print(tn2, fp2, fn2, tp2)
#             tn3, fp3, fn3, tp3 = confusion_matrix(unprotect_grd_labels2, unprotect_pred_labels2).ravel()
#             print(tn3, fp3, fn3, tp3)
#
#             # tpd
#             # tpd = 2*tp1/len(protect_grd_labels) - tp2/len(unprotect_grd_labels)
#             # # td
#             # td = 2*(fp1 + tp1)/len(protect_grd_labels)-(fp2 + tp2)/len(unprotect_grd_labels)
#
#             tpd = tp1 / len(protect_grd_labels) - tp2 / len(unprotect_grd_labels1)+tp1 / len(protect_grd_labels) - tp2 / len(unprotect_grd_labels2)
#             # td
#             td = (fp1 + tp1) / len(protect_grd_labels) - (fp2 + tp2) / len(unprotect_grd_labels1)+(fp1 + tp1) / len(protect_grd_labels) - (fp2 + tp2) / len(unprotect_grd_labels2)
#
#             tsts.append([fname, td, tpd])
#
#             # acc1 = accuracy_score(protect_grd_labels , protect_pred_labels)
#             # recall1 = recall_score(protect_grd_labels , protect_pred_labels)
#             # precision1 = precision_score(protect_grd_labels , protect_pred_labels)
#             # f11 = f1_score(protect_grd_labels , protect_pred_labels)
#             # roc_score1 = roc_auc_score(protect_grd_labels , protect_score)
#             # ap_score1 = average_precision_score(protect_grd_labels , protect_score)
#             #
#             # acc2 = accuracy_score(unprotect_grd_labels, unprotect_pred_labels)
#             # recall2 = recall_score(unprotect_grd_labels, unprotect_pred_labels)
#             # precision2 = precision_score(unprotect_grd_labels, unprotect_pred_labels)
#             # f12 = f1_score(unprotect_grd_labels, unprotect_pred_labels)
#             # roc_score2 = roc_auc_score(unprotect_grd_labels, unprotect_score)
#             # ap_score2 = average_precision_score(unprotect_grd_labels, unprotect_score)
#
# name = ['name', 'td', 'tpd']
# for tst in tsts:
#     print(tst)