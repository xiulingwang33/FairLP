from FLIP.utils.parameter_parser import parameter_parser
from FLIP.utils.utils import tab_printer, read_data,  walk_exist
from FLIP.utils.eval import get_accuracy_scores, get_modularity

from FLIP.skipGram.walk_generator import walk_generator
from FLIP.algs.flip import DW_GAN_LP
from FLIP.utils.logger import Logger

import numpy as np
import os
from sklearn.model_selection import train_test_split
import csv

import pandas as pd




def main():
	args = parameter_parser()
	tab_printer(args)
	data = read_data(args)

	logger = Logger(args)
	train, test = train_test_split(data['examples'], test_size=args.test_size, random_state=2)

	G = data['G']


	walker = walk_generator(G,args)
	#print(walker)

	walker.walker()
	walker.write_walks()

	#train, valid = train_test_split(grt['train'], test_size = 0)
	DW = DW_GAN_LP(args, G , communities= data['communities'], train_data = train, logger = logger )
	y_pred_train = DW.train()
	# Get test results
	y_pred = DW.test(test)
	acc , roc_auc, pa_ap = get_accuracy_scores(test, y_pred, median = np.median(y_pred_train))
	tsts = []
	tst = []
	median3 = np.median(y_pred_train)
	for i in range(len(y_pred)):
		if test[i][0] in (data['communities'])[0]:
			gd1 = 1
		if test[i][0] in (data['communities'])[1]:
			gd1=2

		if test[i][1] in (data['communities'])[0]:
			gd2 = 1
		if test[i][1] in (data['communities'])[1]:
			gd2=2

		if (y_pred[i]>median3 or y_pred[i]==median3):
			bipre = 1
		if y_pred[i] < median3:
			bipre = 0
		tst = [test[i][0], test[i][1], gd1, gd2, test[i][2], y_pred[i], bipre]
		tsts.append(tst)
	name = ['node1', 'node2', 'gender1', 'gender2', 'grt', 'score', 'biscore']
	result = pd.DataFrame(columns=name, data=tsts)

	file=args.file_name.split('.')[0]
	result.to_csv("./results/{0}.csv".format(file))
	modularity_new, modularity_ground  = get_modularity(G,y_pred,data['communities'] , test)
	modred = np.round((modularity_ground- modularity_new)/np.abs(modularity_ground), 4)
	print('flip acc:', np.round(acc, 4))
	print('flip auc:', np.round(roc_auc,4))
	#print('flip modred:', modred3)


if __name__ =="__main__":
	main()








