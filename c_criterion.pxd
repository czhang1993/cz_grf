import numpy as np
cimport numpy as np

from c_tree cimport DOUBLE_t
from c_tree cimport SIZE_t
from c_tree cimport UINT32_t

cdef class Criterion:
    cdef bint proxy_children_impurity
    cdef const DOUBLE_t[ : , : : 1] y
    cdef DOUBLE_t* sample_weight

    cdef SIZE_t n_outputs
    cdef SIZE_t n_relevant_outputs
    cdef SIZE_t n_features
    cdef SIZE_t n_y

    cdef UINT32_t random_state

    cdef SIZE_t* samples
    cdef SIZE_t start
    cdef SIZE_t pos
    cdef SIZE_t end

    cdef SIZE_t n_samples
    cdef SIZE_t max_node_samples
    cdef SIZE_t n_node_samples
    cdef double weighted_n_samples
    cdef double weighted_n_node_samples
    cdef double weighted_n_left
    cdef double weighted_n_right

    cdef double* sum_total
    cdef double* sum_left
    cdef double* sum_right

    cdef int init(
      self, 
      const DOUBLE_t[ : , : : 1] y,            
      DOUBLE_t* sample_weight, 
      double weighted_n_samples,
      SIZE_t* samples
    ) 
    nogil except -1
    
    cdef int node_reset(self, SIZE_t start, SIZE_t end) nogil except -1
    cdef int reset(self) nogil except -1
    cdef int reverse_reset(self) nogil except -1
    cdef int update(self, SIZE_t new_pos) nogil except -1
    cdef double node_impurity(self) nogil
    cdef double proxy_node_impurity(self) nogil
    cdef void children_impurity(self, double* impurity_left, double* impurity_right) nogil
    cdef void node_value(self, double* dest) nogil
    cdef void node_jacobian(self, double* dest) nogil
    cdef void node_precond(self, double* dest) nogil
    cdef double impurity_improvement(self, double impurity) nogil
    cdef double proxy_impurity_improvement(self) nogil
    cdef double min_eig_left(self) nogil
    cdef double min_eig_right(self) nogil


cdef class RegressionCriterion(Criterion):
    cdef double sq_sum_total
