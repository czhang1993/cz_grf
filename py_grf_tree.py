import numpy as np
from cy_grf_tree_criterion import LinearMomentGRFCriterionMSE, LinearMomentGRFCriterion
from py_tree import BaseTree
from sklearn.model_selection import train_test_split
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
import copy


CRITERIA_GRF = {
  "het": LinearMomentGRFCriterion,
  "mse": LinearMomentGRFCriterionMSE
}


class GRFTree(BaseTree):
    def __init__(
      self, 
      *,
      criterion = "mse",
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
      super().__init__(
        criterion = criterion,
        splitter = splitter,
        max_depth = max_depth,
        min_samples_split = min_samples_split,
        min_samples_leaf = min_samples_leaf,
        min_weight_fraction_leaf = min_weight_fraction_leaf,
        min_var_leaf = min_var_leaf,
        min_var_leaf_on_val = min_var_leaf_on_val,
        max_features = max_features,
        random_state = random_state,
        min_impurity_decrease = min_impurity_decrease,
        min_balancedness_tol = min_balancedness_tol,
        honest = honest
      )

    def _get_valid_criteria(self):
        return CRITERIA_GRF

    def _get_valid_min_var_leaf_criteria(self):
        return (LinearMomentGRFCriterion,)

    def _get_store_jac(self):
        return True

    def init(self,):
        self.random_seed_ = self.random_state
        self.random_state_ = check_random_state(self.random_seed_)
        return self

    def fit(self, 
            X, y, n_y, 
            n_outputs, 
            n_relevant_outputs, 
            sample_weight = None, 
            check_input = True
           ):
        return super().fit(
          X, y, n_y, 
          n_outputs, 
          n_relevant_outputs,
          sample_weight = sample_weight, 
          check_input = check_input
        )

    def predict(self, X, check_input=True):
        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)
        pred = self.tree_.predict(X)
        return pred

    def predict_full(self, X, check_input=True):
        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)
        pred = self.tree_.predict_full(X)
        return pred

    def predict_alpha_and_jac(self, X, check_input=True):
        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)
        return self.tree_.predict_precond_and_jac(X)

    def predict_moment(self, X, parameter, check_input=True):
        alpha, jac = self.predict_alpha_and_jac(X)
        return alpha - np.einsum('ijk, ik->ij', jac.reshape((-1, self.n_outputs_, self.n_outputs_)), parameter)

    def feature_importances(self, max_depth=4, depth_decay_exponent=2.0):
        check_is_fitted(self)

        return self.tree_.compute_feature_heterogeneity_importances(
          normalize = True, 
          max_depth = max_depth,
          depth_decay = depth_decay_exponent
        )

    @property
    def feature_importances_(self):
        return self.feature_importances()
