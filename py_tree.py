import abc
import numbers
from math import ceil
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from cy_tree import Tree, DepthFirstTreeBuilder
from cy_tree_splitter import Splitter, BestSplitter
from cy_tree_criterion import Criterion
import cy_tree
from py_utilities import deprecated
from sklearn.model_selection import train_test_split
from sklearn.utils import check_array, check_X_y
from sklearn.utils import check_random_state
from sklearn.utils.validation import _check_sample_weight


DTYPE = _tree.DTYPE
DOUBLE = _tree.DOUBLE

SPLITTERS = {"best": BestSplitter, }


class BaseTree(BaseEstimator):
    def __init__(
        self, 
        *,
        criterion,
        splitter = "best",
        max_depth = None,
        min_samples_split = 10,
        min_samples_leaf = 5,
        min_weight_fraction_leaf = 0.,
        min_var_leaf = None,
        min_var_leaf_on_val = False,
        max_features = None,
        random_state = None,
        min_impurity_decrease = 0.,
        min_balancedness_tol = 0.45,
        honest=True
    ):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.min_var_leaf = min_var_leaf
        self.min_var_leaf_on_val = min_var_leaf_on_val
        self.max_features = max_features
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
        self.min_balancedness_tol = min_balancedness_tol
        self.honest = honest
        
    def _get_valid_min_var_leaf_criteria(self):
        return ()
    
    def get_depth(self):
        check_is_fitted(self)
        return self.tree_.max_depth
    
    def get_n_leaves(self):
        check_is_fitted(self)
        return self.tree_.n_leaves
    
    def fit(
        self, 
        X, y, n_y, 
        n_outputs, 
        n_relevant_outputs, 
        sample_weight = None, 
        check_input = True
    ):
        random_state = self.random_state_

        n_samples, self.n_features_in_ = X.shape
        self.n_outputs_ = n_outputs
        self.n_relevant_outputs_ = n_relevant_outputs
        self.n_y_ = n_y
        self.n_samples_ = n_samples
        self.honest_ = self.honest

        inds = np.arange(n_samples, dtype = np.intp)
        if self.honest:
            random_state.shuffle(inds)
            samples_train, samples_val = inds[ : n_samples // 2], inds[n_samples // 2: ]
        else:
            samples_train, samples_val = inds, inds

        if check_input:
            if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
                y = np.ascontiguousarray(y, dtype = DOUBLE)
            y = np.atleast_1d(y)
            if y.ndim == 1:
                y = np.reshape(y, (-1, 1))
            if len(y) != n_samples:
                raise ValueError(
                    "Number of labels=%d does not match "
                    "number of samples=%d"
                    % (len(y), n_samples)
                )

            if (sample_weight is not None):
                sample_weight = _check_sample_weight(sample_weight, X, DOUBLE)

        max_depth = (np.iinfo(np.int32).max if self.max_depth is None
                     else self.max_depth)

        if isinstance(self.min_samples_leaf, numbers.Integral):
            if not 1 <= self.min_samples_leaf:
                raise ValueError(
                    "min_samples_leaf must be at least 1 "
                    "or in (0, 0.5], got %s"
                    % self.min_samples_leaf
                )
            min_samples_leaf = self.min_samples_leaf
        else:
            if not 0. < self.min_samples_leaf <= 0.5:
                raise ValueError(
                    "min_samples_leaf must be at least 1 "
                    "or in (0, 0.5], got %s"
                    % self.min_samples_leaf
                )
            min_samples_leaf = int(ceil(self.min_samples_leaf * n_samples))

        if isinstance(self.min_samples_split, numbers.Integral):
            if not 2 <= self.min_samples_split:
                raise ValueError(
                    "min_samples_split must be an integer "
                    "greater than 1 or a float in (0.0, 1.0]; "
                    "got the integer %s"
                    % self.min_samples_split
                )
            min_samples_split = self.min_samples_split
        else:
            if not 0. < self.min_samples_split <= 1.:
                raise ValueError(
                    "min_samples_split must be an integer "
                    "greater than 1 or a float in (0.0, 1.0]; "
                    "got the float %s"
                    % self.min_samples_split
                )
            min_samples_split = int(ceil(self.min_samples_split * n_samples))
            min_samples_split = max(2, min_samples_split)

        min_samples_split = max(min_samples_split, 2 * min_samples_leaf)

        if isinstance(self.max_features, str):
            if self.max_features == "auto":
                max_features = self.n_features_in_
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_in_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_in_)))
            else:
                raise ValueError(
                    "Invalid value for max_features. "
                    "Allowed string values are 'auto', "
                    "'sqrt' or 'log2'."
                )
        elif self.max_features is None:
            max_features = self.n_features_in_
        elif isinstance(self.max_features, numbers.Integral):
            max_features = self.max_features
        else:
            if self.max_features > 0.0:
                max_features = max(1, int(self.max_features * self.n_features_in_))
            else:
                max_features = 0

        self.max_features_ = max_features

        if not 0 <= self.min_weight_fraction_leaf <= 0.5:
            raise ValueError("min_weight_fraction_leaf must in [0, 0.5]")
        if max_depth < 0:
            raise ValueError("max_depth must be greater than or equal to zero. ")
        if not (0 <= max_features <= self.n_features_in_):
            raise ValueError("max_features must be in [0, n_features]")
        if not 0 <= self.min_balancedness_tol <= 0.5:
            raise ValueError("min_balancedness_tol must be in [0, 0.5]")

        if self.min_var_leaf is None:
            min_var_leaf = -1.0
        elif isinstance(self.min_var_leaf, numbers.Real) and (self.min_var_leaf >= 0.0):
            min_var_leaf = self.min_var_leaf
        else:
            raise ValueError(
                "min_var_leaf must be either None or a real in [0, infinity). "
                "Got {}".format(self.min_var_leaf)
            )
        if not isinstance(self.min_var_leaf_on_val, bool):
            raise ValueError(
                "min_var_leaf_on_val must be either True or False. "
                "Got {}".format(self.min_var_leaf_on_val)
            )

        if sample_weight is None:
            min_weight_leaf = (self.min_weight_fraction_leaf *
                               n_samples)
        else:
            min_weight_leaf = (self.min_weight_fraction_leaf *
                               np.sum(sample_weight))

        max_train = len(samples_train) if sample_weight is None else np.count_nonzero(sample_weight[samples_train])
        if self.honest:
            max_val = len(samples_val) if sample_weight is None else np.count_nonzero(sample_weight[samples_val])
        if callable(self.criterion):
            criterion = self.criterion(
                self.n_outputs_, 
                self.n_relevant_outputs_, 
                self.n_features_in_, 
                self.n_y_,
                n_samples, 
                max_train,
                random_state.randint(np.iinfo(np.int32).max)
            )
            if not isinstance(criterion, Criterion):
                raise ValueError("Input criterion is not a valid criterion")
            if self.honest:
                criterion_val = self.criterion(
                    self.n_outputs_, 
                    self.n_relevant_outputs_, 
                    self.n_features_in_,
                    self.n_y_, 
                    n_samples, 
                    max_val,
                    random_state.randint(np.iinfo(np.int32).max)
                )
            else:
                criterion_val = criterion
        else:
            valid_criteria = self._get_valid_criteria()
            if not (self.criterion in valid_criteria):
                raise ValueError("Input criterion is not a valid criterion")
            criterion = valid_criteria[self.criterion](
                self.n_outputs_, 
                self.n_relevant_outputs_, 
                self.n_features_in_, 
                self.n_y_, 
                n_samples, 
                max_train,
                random_state.randint(np.iinfo(np.int32).max)
            )
            if self.honest:
                criterion_val = valid_criteria[self.criterion](
                    self.n_outputs_, 
                    self.n_relevant_outputs_, 
                    self.n_features_in_, 
                    self.n_y_, 
                    n_samples, 
                    max_val,
                    random_state.randint(np.iinfo(np.int32).max)
                )
            else:
                criterion_val = criterion

        if (min_var_leaf >= 0.0 and (not isinstance(criterion, self._get_valid_min_var_leaf_criteria())) and
                (not isinstance(criterion_val, self._get_valid_min_var_leaf_criteria()))):
            raise ValueError("This criterion does not support min_var_leaf constraint!")

        splitter = self.splitter
        if not isinstance(self.splitter, Splitter):
            splitter = SPLITTERS[self.splitter](
                criterion, 
                criterion_val,
                self.max_features_,
                min_samples_leaf,
                min_weight_leaf,
                self.min_balancedness_tol,
                self.honest,
                min_var_leaf,
                self.min_var_leaf_on_val,
                random_state.randint(np.iinfo(np.int32).max)
            )

        self.tree_ = Tree(
            self.n_features_in_, 
            self.n_outputs_,
            self.n_relevant_outputs_, 
            store_jac = self._get_store_jac()
        )

        builder = DepthFirstTreeBuilder(
            splitter, 
            min_samples_split,
            min_samples_leaf,
            min_weight_leaf,
            max_depth,
            self.min_impurity_decrease
        )
        builder.build(
            self.tree_, 
            X, y, 
            samples_train, 
            samples_val,
            sample_weight = sample_weight,
            store_jac = self._get_store_jac()
        )

        return self
    
    def _validate_X_predict(self, X, check_input):
        if check_input:
            X = check_array(
                X, 
                dtype = DTYPE, 
                accept_sparse = False, 
                ensure_min_features = 0
            )

        n_features = X.shape[1]
        if self.n_features_in_ != n_features:
            raise ValueError(
                "Number of features of the model must "
                "match the input. Model n_features is %s and "
                "input n_features is %s "
                % (self.n_features_in_, n_features)
            )

        return X
    
    def get_train_test_split_inds(self,):
        check_is_fitted(self)
        random_state = check_random_state(self.random_seed_)
        inds = np.arange(self.n_samples_, dtype = np.intp)
        if self.honest_:
            random_state.shuffle(inds)
            return inds[:self.n_samples_ // 2], inds[self.n_samples_ // 2:]
        else:
            return inds, inds
    
    def apply(self, X, check_input = True):
        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)
        return self.tree_.apply(X)
    
    def decision_path(self, X, check_input = True):
        X = self._validate_X_predict(X, check_input)
        return self.tree_.decision_path(X)

    @property
    @deprecated(
        message = (
            "This attribute is deprecated and will be removed in a future version; "
            "please use the 'n_features_in_' attribute instead."
        )
    )
    def n_features_(self):
        return self.n_features_in_
