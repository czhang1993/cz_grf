import numpy as np
cimport numpy as np

from cy_criterion cimport Criterion

from cy_tree cimport DTYPE_t
from cy_tree cimport DOUBLE_t
from cy_tree cimport SIZE_t
from cy_tree cimport INT32_t
from cy_tree cimport UINT32_t

cdef struct SplitRecord:
    SIZE_t feature
    SIZE_t pos
    SIZE_t pos_val
    double threshold
    double improvement
    double impurity_left
    double impurity_right
    double impurity_left_val
    double impurity_right_val

cdef class Splitter:
    cdef public Criterion criterion
    cdef public Criterion criterion_val
    cdef public SIZE_t max_features
    cdef public SIZE_t min_samples_leaf
    cdef public double min_weight_leaf
    cdef public double min_eig_leaf
    cdef public bint min_eig_leaf_on_val
    cdef public double min_balancedness_tol
    cdef public bint honest

    cdef UINT32_t rand_r_state

    cdef SIZE_t* samples
    cdef SIZE_t n_samples
    cdef double weighted_n_samples
    cdef SIZE_t* samples_val
    cdef SIZE_t n_samples_val
    cdef double weighted_n_samples_val
    cdef SIZE_t* features
    cdef SIZE_t* constant_features
    cdef SIZE_t n_features
    cdef DTYPE_t* feature_values
    cdef DTYPE_t* feature_values_val

    cdef SIZE_t start
    cdef SIZE_t end
    cdef SIZE_t start_val
    cdef SIZE_t end_val

    cdef const DTYPE_t[ : , : ] X
    cdef const DOUBLE_t[ : , : : 1] y
    cdef DOUBLE_t* sample_weight

    cdef int init_sample_inds(
      self, 
      SIZE_t* samples,
      const SIZE_t[ : : 1] np_samples,
      DOUBLE_t* sample_weight,
      SIZE_t* n_samples, double* weighted_n_samples
    ) 
    nogil except -1

    cdef int init(
      self, 
      const DTYPE_t[ : , : ] X, 
      const DOUBLE_t[ : , : : 1] y,
      DOUBLE_t* sample_weight,
      const SIZE_t[ : : 1] np_samples_train,
      const SIZE_t[ : : 1] np_samples_val
    ) 
    nogil except -1

    cdef int node_reset(
      self, 
      SIZE_t start, 
      SIZE_t end, 
      double* weighted_n_node_samples,
      SIZE_t start_val, 
      SIZE_t end_val, 
      double* weighted_n_node_samples_val
    ) 
    nogil except -1

    cdef int node_split(
      self,
      double impurity,
      SplitRecord* split,
      SIZE_t* n_constant_features
    )
    nogil except -1

    cdef void node_value_val(self, double* dest) nogil
    cdef void node_jacobian_val(self, double* dest) nogil
    cdef void node_precond_val(self, double* dest) nogil
    cdef double node_impurity(self) nogil
    cdef double node_impurity_val(self) nogil
    cdef double proxy_node_impurity(self) nogil
    cdef double proxy_node_impurity_val(self) nogil
    cdef bint is_children_impurity_proxy(self) nogil
