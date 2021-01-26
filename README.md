# Cost-Aware Robust Tree Ensembles for Security Applications
Code for the paper "[Cost-Aware Robust Tree Ensembles for Security Applications](https://arxiv.org/abs/1912.01149)" (USENIX Security'21), Yizheng Chen, Shiqi Wang, Weifan Jiang, Asaf Cidon, Suman Jana.

We utilize security domain knowledge to increase the evasion cost against security classifiers, specifically, tree ensemble models that are widely used by security tasks. We propose a new cost modeling method to capture the domain knowledge of features as constraint, and then we integrate the cost-driven constraint into the node construction process to train robust tree ensembles. During the training process, we use the constraint to find data points that are likely to be perturbed given the costs of the features, and we optimize the quality of the trees using a new robust training algorithm. Our cost-aware training method can be applied to different types of tree ensembles, including random forest model (scikit-learn) and gradient boosted decision trees (XGBoost).

## Robust training algorithm

### Implementation in scikit-learn

* Clone our dev version of [scikit-learn](https://github.com/surrealyz/scikit-learn/)
* Check out the [robust](https://github.com/surrealyz/scikit-learn/tree/robust) branch
* We recommend using a virtualenv to install this
* After activating your virtualenv, install the required packages ```pip install numpy scipy joblib threadpoolctl Cython```
* Then install sklearn with our robust training algorithm ```python setup.py install```
* Run `data/download_data.sh` under the current repo [(source)](https://github.com/chenhongge/RobustTrees/blob/master/data/download_data.sh)
* Example usage
  ```
  python train_rf_one.py --train data/binary_mnist0
                        --test data/binary_mnist0.t
                        -m models/rf/greedy/sklearn_greedy_binary_mnist.pickle
                        -b -z -n 784 -r -s robust -e 0.3
                        -c gini --nt 1000 -d 6
  ```

### Implementation in XGBoost

* Clone our dev version of [XGBoost RobustTrees](https://github.com/surrealyz/RobustTrees)
* Check out the [greedy](https://github.com/surrealyz/RobustTrees/tree/greedy) branch
* Run `build.sh`
* `gunzip` all the `*.csv.gz` files under `RobustTrees/data` to obtain the csv datasets. Reading libsvm sometimes has issues in that version of XGBoost, so we converted the dataset to csv files.
* Example usage
  ```
  ./xgboost data/breast_cancer.greedy.conf
  ```

## Datasets

We evaluated our core training algorithm without cost constraints over four benchmark datasets, see the table below.

| Dataset | Train set size  | Test set size  | Majority class in train, test (%)  | # of features  |
|---|---|---|---|---|
| breast-cancer  | 546 | 137  | 62.64, 74.45  | 10  |
| cod-rna  | 59,535  | 271,617  | 66.67, 66.67  | 8  |
| ijcnn1  | 49,990  | 91,701  | 90.29, 90.50  | 22  |
| MNIST 2 vs. 6  | 11,876  | 1,990  | 50.17, 51.86  | 784  |

We have also evaluated our cost-aware training algorihtm over a Twitter spam detection dataset used in the paper ["A Domain-Agnostic Approach to Spam-URL Detection via Redirects"](https://www.andrew.cmu.edu/user/lakoglu/pubs/17-pakdd-urlspam.pdf). We re-extracted 25 features (see Table 7 in our paper) as the Twitter spam detection dataset.

| Twitter spam dataset  | Training  |  Testing |
|---|---|---|
| Malicious  | 130,794  | 55,732  |
| Benign  | 165,076  | 71,070  |
| Total  | 295,870  | 126,802  |

Both datasets are available in `data/`, and the files need to be uncompressed.
Please also run `cd data/; ./download_data.sh` to get libsvm files under `data/` directory, since some of our Python scripts read the libsvm data.

## Benchmark datasets evaluation

### GBDT models

#### Trained models in the paper

* Regular training, **natural** model in the paper: `models/gbdt/nature_*.bin`
* [Chen's robust training algorithm](https://github.com/chenhongge/RobustTrees), **Chen's** model in the paper: `models/gbdt/robust_*.bin`
* Our training algorithm, **ours** model in the paper: `models/gbdt/greedy_*.bin`

#### Evaluate the models

* **Performance:** To evaluate model accuracy, false positive rate, AUC, and plot the ROC curves, please run the following commands:
  * `python scripts/xgboost_roc_plots.py breast_cancer`
  * `python scripts/xgboost_roc_plots.py ijcnn`
  * `python scripts/xgboost_roc_plots.py cod-rna`
  * `python scripts/xgboost_roc_plots.py binary_mnist`
  * The model performance numbers correspond to Table 3, and the generated plots in `roc_plots/` correspond to Figure 7 in the paper.
* **Robustness:** To evaluate the robustness of models, we use the MILP attack: `xgbKantchelianAttack.py`. It uses Gurobi solver, so you need to obtain a licence from Gurobi to use it. They provide free academic license.
  * `mkdir logs`
  * `mkdir adv_examples`
  * `mkdir -p result/gbdt`
  * breast_cancer:
    * `c='nature'; dt='breast_cancer'; python xgbKantchelianAttack.py --data 'data/breast_cancer_scale0.test' --model_type 'xgboost' --model "models/gbdt/${c}_${dt}.bin" --num_classes 2 --nfeat 10 --feature_start 1 --both --maxone -n 100 --out "result/gbdt/${c}_${dt}.txt" --adv "adv_examples/gbdt/${c}_${dt}_adv.pickle" >! logs/milp_gbdt_${c}_${dt}.log 2>&1&`
    * `c='robust'; dt='breast_cancer'; python xgbKantchelianAttack.py --data 'data/breast_cancer_scale0.test' --model_type 'xgboost' --model "models/gbdt/${c}_${dt}.bin" --num_classes 2 --nfeat 10 --feature_start 1 --both --maxone -n 100 --out "result/gbdt/${c}_${dt}.txt" --adv "adv_examples/gbdt/${c}_${dt}_adv.pickle" >! logs/milp_gbdt_${c}_${dt}.log 2>&1&`
    * `c='greedy'; dt='breast_cancer'; python xgbKantchelianAttack.py --data 'data/breast_cancer_scale0.test' --model_type 'xgboost' --model "models/gbdt/${c}_${dt}.bin" --num_classes 2 --nfeat 10 --feature_start 1 --both --maxone -n 100 --out "result/gbdt/${c}_${dt}.txt" --adv "adv_examples/gbdt/${c}_${dt}_adv.pickle" >! logs/milp_gbdt_${c}_${dt}.log 2>&1&`
  * cod-rna
    * `md='nature_cod-rna'; python xgbKantchelianAttack.py --data 'data/cod-rna_s.t' --model_type 'xgboost' --model "models/gbdt/${md}.bin" --num_classes 2 --nfeat 8 --feature_start 0 --both --maxone -n 5000 --out "result/gbdt/${md}.txt" --adv "adv_examples/gbdt/${md}_adv.pickle" >! logs/milp_gbdt_${md}.log 2>&1&`
    * `md='robust_cod-rna'; python xgbKantchelianAttack.py --data 'data/cod-rna_s.t' --model_type 'xgboost' --model "models/gbdt/${md}.bin" --num_classes 2 --nfeat 8 --feature_start 0 --both --maxone -n 5000 --out "result/gbdt/${c}_${dt}.txt" --adv "adv_examples/gbdt/${c}_${dt}_adv.pickle" >! logs/milp_gbdt_${c}_${dt}.log 2>&1&`
    * `md='greedy_cod-rna_center_eps0.03.bin'; python xgbKantchelianAttack.py --data 'data/cod-rna_s.t' --model_type 'xgboost' --model "models/gbdt/${md}.bin" --num_classes 2 --nfeat 8 --feature_start 0 --both --maxone -n 5000 --out "result/gbdt/${c}_${dt}.txt" --adv "adv_examples/gbdt/${c}_${dt}_adv.pickle" >! logs/milp_gbdt_${c}_${dt}.log 2>&1&`
  * ijcnn:
    * `md='nature_ijcnn'; python xgbKantchelianAttack.py --data 'data/ijcnn1s0.t' --model_type 'xgboost' --model "models/gbdt/${md}.bin" --num_classes 2 --nfeat 22 --feature_start 1 --both --maxone -n 100 --out "result/gbdt/${md}.txt" --adv "adv_examples/gbdt/${md}_adv.pickle" >! logs/milp_gbdt_${md}.log 2>&1&`
    * `md='robust_ijcnn'; python xgbKantchelianAttack.py --data 'data/ijcnn1s0.t' --model_type 'xgboost' --model "models/gbdt/${md}.bin" --num_classes 2 --nfeat 22 --feature_start 1 --both --maxone -n 100 --out "result/gbdt/${md}.txt" --adv "adv_examples/gbdt/${md}_adv.pickle" >! logs/milp_gbdt_${md}.log 2>&1&`
    * `md='greedy_ijcnn_center_eps0.02_nr60_md8'; python xgbKantchelianAttack.py --data 'data/ijcnn1s0.t' --model_type 'xgboost' --model "models/gbdt/${md}.bin" --num_classes 2 --nfeat 22 --feature_start 1 --both --maxone -n 100 --out "result/gbdt/${md}.txt" --adv "adv_examples/gbdt/${md}_adv.pickle" >! logs/milp_gbdt_${md}.log 2>&1&`
  * binary_mnist:
    * `md='nature_binary_mnist'; python xgbKantchelianAttack.py -n 100 --data 'data/binary_mnist0.t' --model_type 'xgboost' --model "models/gbdt/${md}.bin" --num_classes 2 --nfeat 784 --both --maxone --feature_start 0 --out "result/gbdt/${md}.txt" --adv "adv_examples/gbdt/${md}.pickle" >! logs/milp_gbdt_${md}.log &`
    * `md='robust_binary_mnist'; python xgbKantchelianAttack.py -n 100 --data 'data/binary_mnist0.t' --model_type 'xgboost' --model "models/gbdt/${md}.bin" --num_classes 2 --nfeat 784 --both --maxone --feature_start 0 --out "result/gbdt/${md}.txt" --adv "adv_examples/gbdt/${md}.pickle" >! logs/milp_gbdt_${md}.log &`
    * `md='greedy_binary_mnist_center'; python xgbKantchelianAttack.py -n 100 --data 'data/binary_mnist0.t' --model_type 'xgboost' --model "models/gbdt/${md}.bin" --num_classes 2 --nfeat 784 --both --maxone --feature_start 0 --out "result/gbdt/${md}.txt" --adv "adv_examples/gbdt/${md}.pickle" >! logs/milp_gbdt_${md}.log &`

The commands for cod-rna dataset take weeks to run, whereas the commands for the other datasets should finish within a day. Therefore, we have provided the result logs here.

#### How to train the models

### Random Forest models

* **Performance:** To evaluate model accuracy, false positive rate, AUC, and plot the roc curve figures, please run the following commands:
  * `python scripts/`

  * The model performance numbers correspond to Table , and the generated plots in `roc_plots/` correspond to Figure  in the paper.
* **Robustness:** 

#### Trained models in the paper

#### Evaluate the models

#### How to train the models

## Twitter Spam Detection Application

### Box Constraint Specification

### Trained models in the paper

### Evaluate the models

