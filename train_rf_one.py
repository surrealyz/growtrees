#! /usr/bin/env python
import sys
import os
import argparse
from sklearn import datasets
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import RandomForestClassifier
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description='Train one sklearn RF model.')
    parser.add_argument('--train', '--train_data', type=str, help='train data file name.', required=True)
    parser.add_argument('--test', '--test_data', type=str, help='test data file name.', required=True)
    parser.add_argument('-m', '--model_path', type=str, help='save sklearn model pickle path.', required=True)
    parser.add_argument('-b', '--binary_class', default=False, help='whether it is binary class.', action='store_true')
    parser.add_argument('-n', '--nfeat', type=int, help='number of features.', required=True)
    parser.add_argument('-z', '--zero_start', default=False, help='whether the feature starts from 0.', action='store_true')
    parser.add_argument('-r', '--robust', default=False, help='whether train a robust model.', action='store_true')
    parser.add_argument('-s', '--splitter', type=str, default='best', choices=['best', 'robust'], help='choose the splitter.', required=False)
    parser.add_argument('-e', '--eps', type=float, default=0.0, help='robust epsilon range.', required=False)
    parser.add_argument('-c', '--criterion', type=str, default='gini', help='the splitting criterion.', required=False)
    parser.add_argument('--nt', type=int, help='number of decision trees.', required=True)
    parser.add_argument('-d', '--max_depth', type=int, help='maximum tree depth.', required=True)
    return parser.parse_args()

def eval(y, y_p):
    try:
        tn, fp, fn, tp = confusion_matrix(y, y_p).ravel()
        acc = (tp+tn)/float(tp+tn+fp+fn)
        fpr = fp/float(fp+tn)
        return acc, fpr
    except ValueError:
        return accuracy_score(y, y_p), None

def main(args):
    #print(args)
    # load test input and test output
    x_train, y_train = datasets.load_svmlight_file(args.train,
                                       n_features=args.nfeat,
                                       multilabel=(not args.binary_class),
                                       zero_based=args.zero_start)

    x_test, y_test = datasets.load_svmlight_file(args.test,
                                       n_features=args.nfeat,
                                       multilabel=(not args.binary_class),
                                       zero_based=args.zero_start)

    # train model
    clf = RandomForestClassifier(robust=args.robust, epsilon=args.eps,
            splitter=args.splitter,
            verbose=0, criterion=args.criterion,
            n_estimators=args.nt, max_depth=args.max_depth, random_state=0,
            n_jobs=12,
            max_features=0.5)
    clf.fit(x_train.toarray(), y_train)
    y_hat = clf.predict(x_test.toarray())
    #print(y_hat)
    #print(accuracy_score(y_test, y_hat))
    acc, fpr = eval(y_test, y_hat)
    print(args.model_path, "RF Accuracy: ", acc, "FPR: ", fpr)

    # save model to pickle
    pickle.dump(clf, open(args.model_path, "wb"))

    # save to json
    json_path = '%s.json' % args.model_path.split('.pickle')[0]
    cmd = 'python3 save_sklearn_rf_to_json.py \
            --model_path %s \
            --output %s' % (args.model_path, json_path)
    print(cmd)
    os.system(cmd)
    return


if __name__=='__main__':
    args = parse_args()
    main(args)
