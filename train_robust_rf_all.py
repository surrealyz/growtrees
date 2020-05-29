#! /usr/bin/env python
import os
import sys
import argparse
import pandas as pd
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle
import logging
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Train all robust sklearn RF model.')
    return parser.parse_args()

datasets = {"binary_mnist": ['binary_mnist0', 'binary_mnist0.t'],
            "breast_cancer": ['breast_cancer_scale0.train', 'breast_cancer_scale0.test'],
            "cod-rna": ['cod-rna_s', 'cod-rna_s.t'],
            "covtype.scale01": ['covtype.scale01.train0', 'covtype.scale01.test0'],
			#"diabetes": ['diabetes_scale0.train', 'diabetes_scale0.test'],
            "fashion": ['fashion.train0', 'fashion.test0'],
            #"higgs": ['HIGGS_s.train0', 'HIGGS_s.test0'],
            "ijcnn": ['ijcnn1s0', 'ijcnn1s0.t'],
            "ori_mnist": ['ori_mnist.train0', 'ori_mnist.test0'],
            "Sensorless": ['Sensorless.scale.tr0', 'Sensorless.scale.val0'],
            "webspam": ['webspam_wc_normalized_unigram.svm0.train', 'webspam_wc_normalized_unigram.svm0.test']}

tree_size = {"binary_mnist": [60, 14],
            "breast_cancer": [20, 6],
            "cod-rna": [60, 14],
            "covtype.scale01": [100, 24],
            "fashion": [100, 24],
            "ijcnn": [80, 14],
            "ori_mnist": [100, 14],
            "Sensorless": [60, 14],
            "webspam": [200, 14]}


eps_val = {"binary_mnist": 0.3,
            "breast_cancer": 0.3,
            "cod-rna": 0.2,
            "covtype.scale01": 0.2,
			"diabetes": 0.2,
            "fashion": 0.1,
            "higgs": 0.05,
            "ijcnn": 0.1,
            "ori_mnist": 0.3,
            "Sensorless": 0.05,
            "webspam": 0.05}

n_feat = {"binary_mnist": 784,
            "breast_cancer": 10,
            "cod-rna": 8,
            "covtype.scale01": 54,
			"diabetes": 8,
            "fashion": 784,
            "higgs": 28,
            "ijcnn": 22,
            "ori_mnist": 784,
            "Sensorless": 48,
            "webspam": 254}

zero_based = {"binary_mnist": True,
            "breast_cancer": False,
            "cod-rna": True,
            "covtype.scale01": False,
			"diabetes": False,
            "fashion": False,
            "higgs": True,
            "ijcnn": False,
            "ori_mnist": False,
            "Sensorless": False,
            "webspam": False}

binary_class = {"binary_mnist": True,
            "breast_cancer": True,
            "cod-rna": True,
            "covtype.scale01": False,
			"diabetes": True,
            "fashion": False,
            "higgs": True,
            "ijcnn": True,
            "ori_mnist": False,
            "Sensorless": False,
            "webspam": True}

def main(args):
    data_path = 'data/'
    all_models_path = 'models/rf/greedy/'
    log_file_path = 'logs/train_robust_rf_all.log'

    logging.basicConfig(filename=log_file_path,
                        filemode='a',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO,
                        )
    logger = logging.getLogger('robust.rf')
    logger.info("Starting training robust sklearn RF models...")

    # command for training one
    '''
    python train_rf_one.py --train data/binary_mnist0
    --test data/binary_mnist0.t
    -m models/rf/greedy/sklearn_greedy_binary_mnist.pickle
    -b -z -r -s robust -e 0.3 -c gini -n 784 --nt 1000 -d 6
    '''
    for dataset in ['binary_mnist', 'breast_cancer', 'cod-rna', 'ijcnn', 'webspam', 'covtype.scale01', 'fashion', 'ori_mnist', 'Sensorless']:
        if dataset in ['binary_mnist', 'breast_cancer', 'cod-rna', 'ijcnn']:
            continue
        fname = datasets[dataset]
        # track time for each one
        start = time.time()
        train, test = fname
        train_path = data_path + train
        test_path = data_path + test
        model_path = all_models_path + 'sklearn_greedy_' + dataset + '.pickle'
        options = ''
        if binary_class[dataset] is True:
            options += '-b '
        if zero_based[dataset] is True:
            options += '-z '
        # add number of features for the dataset
        options += '-n %s ' % n_feat[dataset]
        # add epsilon value for robust model
        options += '-r -s robust '
        options += '-e %s ' % eps_val[dataset]
        # use gini for now
        options += '-c gini '
        # batch tree number and max max_depth
        #options += '--nt 60 -d 8'
        options += '--nt %s -d %s' % (tree_size[dataset][0], tree_size[dataset][1])

        cmd = 'python3 train_rf_one.py --train %s --test %s -m %s %s' \
            % (train_path, test_path, model_path, options)

        logging.info(cmd)
        os.system(cmd)
        end = time.time()
        logging.info('time in seconds: %f' % (end - start))

    return


if __name__=='__main__':
    args = parse_args()
    main(args)
