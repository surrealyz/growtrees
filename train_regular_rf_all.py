#! /usr/bin/env python
import os
import sys
import argparse
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle
import logging
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Train all regular sklearn RF model.')
    return parser.parse_args()

datasets = {"binary_mnist": ['binary_mnist0', 'binary_mnist0.t'],
            "breast_cancer": ['breast_cancer_scale0.train', 'breast_cancer_scale0.test'],
            "cod-rna": ['cod-rna_s', 'cod-rna_s.t'],
            "ijcnn": ['ijcnn1s0', 'ijcnn1s0.t']}

tree_size = {"binary_mnist": [60, 14],
            "breast_cancer": [20, 6],
            "cod-rna": [60, 14],
            "ijcnn": [80, 14]}

n_feat = {"binary_mnist": 784,
            "breast_cancer": 10,
            "cod-rna": 8,
            "ijcnn": 22}

zero_based = {"binary_mnist": True,
            "breast_cancer": False,
            "cod-rna": True,
            "ijcnn": False}

binary_class = {"binary_mnist": True,
            "breast_cancer": True,
            "cod-rna": True,
            "ijcnn": True}

def main(args):
    data_path = 'data/'
    all_models_path = 'models/rf/nature/'
    log_file_path = 'logs/train_regular_rf_all.log'

    logging.basicConfig(filename=log_file_path,
                        filemode='a',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO,
                        )
    logger = logging.getLogger('regular.rf')
    logger.info("Starting training regular sklearn RF models...")

    # command for training one
    '''
    python train_rf_one.py --train data/binary_mnist0
    --test data/binary_mnist0.t
    -m models/rf/nature/sklearn_nature_binary_mnist.pickle
    -b -z -c gini -n 784 --nt 60 -d 8
    '''
    for dataset, fname in datasets.items():
        # track time for each one
        start = time.time()
        train, test = fname
        train_path = data_path + train
        test_path = data_path + test
        model_path = all_models_path + 'sklearn_nature_' + dataset + '.pickle'
        options = ''
        if binary_class[dataset] is True:
            options += '-b '
        if zero_based[dataset] is True:
            options += '-z '
        # add number of features for the dataset
        options += '-n %s ' % n_feat[dataset]
        # use the regular 'best' splitter
        options += '-s best '
        # use gini for now
        options += '-c gini '
        # batch tree number and max max_depth
        options += '--nt %s -d %s' % (tree_size[dataset][0], tree_size[dataset][1])

        cmd = 'python train_rf_one.py --train %s --test %s -m %s %s' \
            % (train_path, test_path, model_path, options)

        logging.info(cmd)
        os.system(cmd)
        end = time.time()
        logging.info('time in seconds: %f' % (end - start))

    return


if __name__=='__main__':
    args = parse_args()
    main(args)
