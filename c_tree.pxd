import numpy as np
cimport numpy as np

ctypedef np.npy_float64 DTYPE_t
ctypedef np.npy_float64 DOUBLE_t
ctypedef np.npy_intp SIZE_t
ctypedef np.npy_int32 INT32_t
ctypedef np.npy_uint32 UINT32_t

from c_splitter cimport Splitter
from c_splitter cimport SplitRecord

cdef struct Node:
    SIZE_t left_child
    SIZE_t right_child
    SIZE_t depth
    SIZE_t feature
    DOUBLE_t threshold
    DOUBLE_t impurity
    SIZE_t n_node_samples
    DOUBLE_t weighted_n_node_samples
    DOUBLE_t impurity_train
    SIZE_t n_node_samples_train
    DOUBLE_t weighted_n_node_samples_train

cdef class Tree:
    cdef public SIZE_t n_features
    cdef public SIZE_t n_outputs
    cdef public SIZE_t n_relevant_outputs
    cdef SIZE_t* n_classes
    cdef public SIZE_t max_n_classes

    cdef public SIZE_t max_depth
    cdef public SIZE_t node_count
    cdef public SIZE_t capacity
    cdef Node* nodes
    cdef double* value
    cdef SIZE_t value_stride
    cdef bint store_jac
    cdef double* jac
    cdef SIZE_t jac_stride
    cdef double* precond
    cdef SIZE_t precond_stride

    cdef SIZE_t _add_node(
      self, 
      SIZE_t parent, 
      bint is_left, 
      bint is_leaf,
      SIZE_t feature, 
      double threshold, 
      double impurity_train, 
      SIZE_t n_node_samples_train,
      double weighted_n_samples_train,
      double impurity_val, 
      SIZE_t n_node_samples_val,
      double weighted_n_samples_val
    ) 
    nogil except -1
    
    cdef int _resize(
      self, 
      SIZE_t capacity
    ) 
    nogil except -1
    
    cdef int _resize_c(self, SIZE_t capacity = *) nogil except -1

    cdef np.ndarray _get_value_ndarray(self)
    cdef np.ndarray _get_jac_ndarray(self)
    cdef np.ndarray _get_precond_ndarray(self)
    cdef np.ndarray _get_node_ndarray(self)

    cpdef np.ndarray predict(self, object X)
    cpdef np.ndarray predict_jac(self, object X)
    cpdef np.ndarray predict_precond(self, object X)
    cpdef predict_precond_and_jac(self, object X)
    cpdef np.ndarray predict_full(self, object X)

    cpdef np.ndarray apply(self, object X)
    cdef np.ndarray _apply(self, object X)
    cpdef object decision_path(self, object X)
    cdef object _decision_path(self, object X)

    cpdef compute_feature_importances(
      self, 
      normalize = *, 
      max_depth = *, 
      depth_decay = *
    )
    
    cpdef compute_feature_heterogeneity_importances(
      self, 
      normalize = *, 
      max_depth = *, 
      depth_decay = *
    )

cdef class TreeBuilder:
    cdef Splitter splitter

    cdef SIZE_t min_samples_split
    cdef SIZE_t min_samples_leaf
    cdef double min_weight_leaf
    cdef SIZE_t max_depth
    cdef double min_impurity_decrease

    cpdef build(
      self, 
      Tree tree, 
      object X, np.ndarray y,
      np.ndarray samples_train,
      np.ndarray samples_val,
      np.ndarray sample_weight = *,
      bint store_jac = *
    )
    
    cdef _check_input(
      self, 
      object X, np.ndarray y, 
      np.ndarray sample_weight
    )
