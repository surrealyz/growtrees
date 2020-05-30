#! /usr/bin/env python
import sys
import argparse
import xgboost as xgb

def parse_args():
    parser = argparse.ArgumentParser(description='Save xgboost model to the json format.')
    parser.add_argument('--model_path', type=str, help='binary xgboost model path.', required=True)
    parser.add_argument('--output', type=str, help='output json file name.', required=True)
    return parser.parse_args()

def main(args):
    # load the trained model
    model = xgb.Booster()
    model.load_model(args.model_path)
    model.dump_model(args.output, dump_format='json')
    return

if __name__=='__main__':
    args = parse_args()
    main(args)
