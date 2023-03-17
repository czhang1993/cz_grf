import numpy as np
import pandas as pd
import scipy.sparse
import sparse as sp
import itertools
import inspect
import types
from operator import getitem
from collections import defaultdict, Counter
from sklearn import clone
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.linear_model import LassoCV, MultiTaskLassoCV, Lasso, MultiTaskLasso
from functools import reduce, wraps
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import assert_all_finite
from sklearn.preprocessing import PolynomialFeatures
import warnings
from warnings import warn
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from collections.abc import Iterable
from sklearn.utils.multiclass import type_of_target
import numbers
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.summary import summary_return
from statsmodels.compat.python import lmap
import copy
from inspect import


def deprecated(message, category=FutureWarning):
    def decorator(to_wrap):
        if isinstance(to_wrap, type):
            old_init = to_wrap.__init__

            @wraps(to_wrap.__init__)
            def new_init(*args, **kwargs):
                warn(message, category, stacklevel = 2)
                old_init(*args, **kwargs)

            to_wrap.__init__ = new_init

            return to_wrap
        else:
            @wraps(to_wrap)
            def m(*args, **kwargs):
                warn(message, category, stacklevel = 2)
                return to_wrap(*args, **kwargs)
            return m
    return decorator
