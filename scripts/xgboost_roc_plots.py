
import sys
import os
import argparse

import random
import json
import xgboost as xgb
from sklearn import datasets
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle
import numpy as np

from sklearn import metrics 
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser(description='Load xgboost model to predict test data.')
parser.add_argument('dataset', type=str)
parser.add_argument('--data_path', type=str, default=None)
parser.add_argument('--train_method', type=str, default="greedy")
parser.add_argument('--num_attacks', type=int, default=5000)
parser.add_argument('--num_classes', type=str, default="2")
parser.add_argument('--attack', action="store_true", default=False)
parser.add_argument('--threads', type=int, default=8)
args = parser.parse_args()

train_method = args.train_method
dataset = args.dataset


if args.dataset in ["cod-rna", "binary_mnist"]:
	fstart = "0"
else:
	fstart = "1"


print("dataset:", dataset)
if dataset == "cod-rna":
	data_path = "data/cod-rna_s.t"
	nfeat = "8"
	args.num_classes = 2
	args.num_attacks = 5000
elif dataset == "binary_mnist":
	data_path = "data/binary_mnist0.t"
	nfeat = "784"
	args.num_classes = 2
	args.num_attacks = 100
elif dataset == "ijcnn":
	data_path = "data/ijcnn1s0.t"
	nfeat = "22"
	args.num_classes = 2
	args.num_attacks = 100
elif dataset == "breast_cancer":
	data_path = "data/breast_cancer_scale0.test"
	nfeat = "10"
	args.num_classes = 2
	args.num_attacks = 137

elif dataset == "fashion":
	data_path = "data/fashion.test0"
	nfeat = "10"
	args.num_classes = 10
	args.num_attacks = 100

else:
	print("no such dataset")
	exit()

if args.data_path is None:
	sample_tail = "_n"+str(args.num_attacks)
else:
    data_path = args.data_path
    sample_tail = "_c"+"_n"+str(args.num_attacks)

def eval(y, y_p):
    try:
        tn, fp, fn, tp = confusion_matrix(y, y_p).ravel()
        print(tp, tn, fp, fn)
        acc = (tp+tn)/float(tp+tn+fp+fn)
        fpr = fp/float(fp+tn)
        tpr = tp/float(tp+fn)
        print(tp, fp, fn)
        return acc, fpr, tpr
    except ValueError:
        return accuracy_score(y, y_p), None, None


def get_roc_curve(model_path, data_path, nfeat, fstart):
    model = xgb.Booster()
    model.load_model(model_path)
    fstart, nfeat = int(fstart), int(nfeat)

    print("data loaded from ", data_path)
    if data_path.endswith(".pickle"):
        x_test = pickle.load(open(data_path, 'rb'))
        y_test = np.ones(x_test.shape[0])
    elif data_path.endswith(".csv"):
        x_test = np.loadtxt(data_path, delimiter=',', usecols=list(range(1, nfeat+1)))
        y_test = np.loadtxt(data_path, delimiter=',', usecols=0).astype(int)
    else:
        # libsvm file format
        x_test, y_test = datasets.load_svmlight_file(data_path)
        x_test = x_test.toarray()
        if fstart > 0:
            x_test = np.hstack((np.zeros((x_test.shape[0],fstart)),x_test))
        y_test = y_test[:,np.newaxis].astype(int)
        
    dtest = xgb.DMatrix(x_test, label=y_test)
    preds = model.predict(dtest)
    y_pred = [1 if p > 0.5 else 0 for p in preds]
    acc, fpr, tpr = eval(y_test, y_pred)
    print("accuracy: ", acc, "fpr: ", fpr, "tpr:", tpr)
    print("accuracy_score: {:.4f}".format(accuracy_score(y_test, y_pred)))
    
    fps, tps, thresholds = metrics.roc_curve(y_test, preds)
    auc = metrics.auc(fps, tps)
    print("AUC: {:.5f}".format(auc))
    return fps, tps, thresholds, auc

model_paths = {"breast_cancer": {
                    "nature": "models/gbdt/nature_breast_cancer.bin",
                    "robust": "models/gbdt/robust_breast_cancer.bin",
                    "greedy": "models/gbdt/greedy_breast_cancer.bin"
                    },
                "ijcnn":{
                    "nature": "models/gbdt/nature_ijcnn.bin",
                    "robust": "models/gbdt/robust_ijcnn.bin",
                    "greedy": "models/gbdt/greedy_ijcnn_center_eps0.02_nr60_md8.bin"
                    },
                "cod-rna":{
                    "nature": "models/gbdt/nature_cod-rna.bin",
                    "robust": "models/gbdt/robust_cod-rna.bin",
                    "greedy": "models/gbdt/greedy_cod-rna_center_eps0.03.bin"
                    },
                "binary_mnist":{
                    "nature": "models/gbdt/nature_binary_mnist.bin",
                    "robust": "models/gbdt/robust_binary_mnist.bin",
                    "greedy": "models/gbdt/greedy_binary_mnist_center.bin"
                    }
        }
lists = ["nature", "robust", "greedy"]
markers = ['o', '*', 'v']
markevery = 0.1
for i, train_method in enumerate(lists):
    
    print("=================[{}]=====================".format(train_method))
    model_path = model_paths[dataset][train_method]
    print("model path:", model_path)

    fps, tps, thresholds, auc = get_roc_curve(model_path, data_path, nfeat, fstart)
    #print(fps, tps, thresholds)
    plt.plot(fps, tps, label=train_method+ "({:.5f})".format(auc), marker=markers[i], markevery=markevery)

plt.plot([0, 1], [0, 1], 'k--')
plt.legend()
if not os.path.exists("roc_plots/"):
    os.mkdir("roc_plots/")
    print("make dir", "roc_plots/")
plt.title(dataset)
plt.savefig("roc_plots/"+dataset+"_roc.png")



#cmd = "python scripts/roc_plots.py --model_path "+model_path+" --test_data "+data_path+\
    #" --nfeat "+nfeat+" --fstart "+fstart
