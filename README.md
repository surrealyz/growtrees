# Cost-Aware Robust Tree Ensembles for Security Applications
Code for the paper "[Cost-Aware Robust Tree Ensembles for Security Applications](https://arxiv.org/abs/1912.01149)", Yizheng Chen, Shiqi Wang, Weifan Jiang, Asaf Cidon, Suman Jana.

We utilize security domain knowledge to increase the evasion cost against security classifiers, specifically, tree ensemble models that are widely used by security tasks. We propose a new cost modeling method to capture the domain knowledge of features as constraint, and then we integrate the cost-driven constraint into the node construction process to train robust tree ensembles. During the training process, we use the constraint to find data points that are likely to be perturbed given the costs of the features, and we optimize the quality of the trees using a new robust training algorithm. Our cost-aware training method can be applied to different types of tree ensembles, including random forest model (scikit-learn) and gradient boosted decision trees (Xgboost).

## Training algorithm

### Download benchmark datasets

* Run `data/download_data.sh`, [source](https://github.com/chenhongge/RobustTrees/blob/master/data/download_data.sh)

### Implementation in scikit-learn

* Clone our dev version of [scikit-learn](https://github.com/surrealyz/scikit-learn/)
* Switch to the [robust](https://github.com/surrealyz/scikit-learn/tree/robust) branch
* We recommend using a virtualenv to install this
* After activating your virtualenv, install the required packages ```pip install numpy scipy joblib threadpoolctl Cython```
* Then install sklearn with our robust training algorithm ```python setup.py install```
* Example usage
  ```
  python train_rf_one.py --train data/binary_mnist0
                        --test data/binary_mnist0.t
                        -m models/rf/greedy/sklearn_greedy_binary_mnist.pickle
                        -b -z -n 784 -r -s robust -e 0.3
                        -c gini --nt 1000 -d 6
  ```

### Implementation in Xgboost
