#! /usr/bin/env python
import sys
import json
import pickle
import pprint
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Convert sklearn Random Forest model to the json format.')
    parser.add_argument('--model_path', type=str, help='pickled sklearn model path. e.g., models/rf/sklearn_greedyrobust.pickle', required=True)
    parser.add_argument('--output', type=str, help='output json file name.', required=True)
    return parser.parse_args()

# recusion
def get_children(cur_tree, node_id, depth):
    fid = cur_tree.feature[node_id]
    # non leaf
    if fid != -2:
        thisitem = {"nodeid": node_id, "depth": depth, "split": int(fid),
                    "split_condition": cur_tree.threshold[node_id],
                    "yes": int(cur_tree.children_left[node_id]),
                    "no": int(cur_tree.children_right[node_id]),
                    "missing": int(cur_tree.children_left[node_id]),
                    "children": [
                        get_children(cur_tree, int(cur_tree.children_left[node_id]), depth+1),
                        get_children(cur_tree, int(cur_tree.children_right[node_id]), depth+1)
                    ]}
    else:
        # leaf
        output_vec = cur_tree.value[node_id][0]
        malprob = output_vec[1]/(output_vec[0]+output_vec[1])
        thisitem = {"nodeid": node_id, "leaf": malprob}
    return thisitem

def main(args):
    rf_cls = pickle.load(open(args.model_path, 'rb'))

    pretty_forest = []
    # getting the dict of one tree
    for estimator in rf_cls.estimators_:
        cur_tree = estimator.tree_
        root = 0
        depth = 0
        fid = cur_tree.feature[root]
        pretty_tree = {"nodeid": root, "depth": depth, "split": int(fid),
            "split_condition": cur_tree.threshold[root],
            "yes": int(cur_tree.children_left[root]),
            "no": int(cur_tree.children_right[root]),
            "missing": int(cur_tree.children_left[root]),
            "children": [get_children(cur_tree, int(cur_tree.children_left[root]), depth+1),
                        get_children(cur_tree, int(cur_tree.children_right[root]), depth+1)]}
        pretty_forest.append(pretty_tree)
    #print(pretty_tree)
    #print(json.dumps(pretty_forest, indent=4))
    json.dump(pretty_forest, open(args.output, 'w'), indent=4)

if __name__=='__main__':
    args = parse_args()
    main(args)
