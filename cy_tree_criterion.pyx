from libc.stdlib cimport calloc
from libc.stdlib cimport free
from libc.string cimport memcpy
from libc.string cimport memset
from libc.math cimport fabs

import numpy as np
cimport numpy as np
np.import_array()

from ._utils cimport log
from ._utils cimport safe_realloc
from ._utils cimport sizet_ptr_to_ndarray

cdef class Criterion:
    def __dealloc__(self):
        free(self.sum_total)
        free(self.sum_left)
        free(self.sum_right)

    def __getstate__(self):
        return {}

    cdef int init(
      self, 
      const DOUBLE_t[ : , : : 1] y, 
      DOUBLE_t* sample_weight,
      double weighted_n_samples,
      SIZE_t* samples
    ) 
    nogil except -1:
        pass
    
    cdef int node_reset(
      self, 
      SIZE_t start, 
      SIZE_t end
    ) 
    nogil except -1:
        pass

    cdef int reset(self) nogil except -1:
        pass

    cdef int reverse_reset(self) nogil except -1:
        pass

    cdef int update(self, SIZE_t new_pos) nogil except -1:
        pass

    cdef double node_impurity(self) nogil:
        pass

    cdef double proxy_node_impurity(self) nogil:
        return self.node_impurity()

    cdef void children_impurity(
      self, 
      double* impurity_left,
      double* impurity_right
    ) 
    nogil:
        pass

    cdef void node_value(
      self, 
      double* dest
    ) 
    nogil:
        pass
    
    cdef void node_jacobian(
      self, 
      double* dest
    ) 
    nogil:
        with gil:
            raise AttributeError("Criterion does not support jacobian calculation")
    
    cdef void node_precond(
      self, 
      double* dest
    ) 
    nogil:
        with gil:
            raise AttributeError("Criterion does not support preconditioned value calculation")
    
    cdef double min_eig_left(self) nogil:
        with gil:
            raise AttributeError("Criterion does not support jacobian and eigenvalue calculation!")
    
    cdef double min_eig_right(self) nogil:
        with gil:
            raise AttributeError("Criterion does not support jacobian and eigenvalue calculation!")

    cdef double proxy_impurity_improvement(self) nogil:
        cdef double impurity_left
        cdef double impurity_right
        self.children_impurity(&impurity_left, &impurity_right)

        return (- self.weighted_n_right * impurity_right
                - self.weighted_n_left * impurity_left)

    cdef double impurity_improvement(self, double impurity) nogil:
        cdef double impurity_left
        cdef double impurity_right

        self.children_impurity(&impurity_left, &impurity_right)

        return ((self.weighted_n_node_samples / self.weighted_n_samples) *
                (impurity - (self.weighted_n_right / 
                             self.weighted_n_node_samples * impurity_right)
                          - (self.weighted_n_left / 
                             self.weighted_n_node_samples * impurity_left)))


cdef class RegressionCriterion(Criterion):
    def __cinit__(
      self, 
      SIZE_t n_outputs, 
      SIZE_t n_relevant_outputs, 
      SIZE_t n_features, 
      SIZE_t n_y,
      SIZE_t n_samples, 
      SIZE_t max_node_samples, 
      UINT32_t random_state
    ):

        self.n_outputs = n_outputs
        self.n_relevant_outputs = n_relevant_outputs
        self.n_features = n_features
        self.n_y = n_y
        self.random_state = random_state
        self.proxy_children_impurity = False

        self.samples = NULL
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_samples = n_samples
        self.max_node_samples = max_node_samples
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        self.sq_sum_total = 0.0

        self.sum_total = NULL
        self.sum_left = NULL
        self.sum_right = NULL

        self.sum_total = <double*> calloc(n_outputs, sizeof(double))
        self.sum_left = <double*> calloc(n_outputs, sizeof(double))
        self.sum_right = <double*> calloc(n_outputs, sizeof(double))

        if (self.sum_total == NULL or 
                self.sum_left == NULL or
                self.sum_right == NULL):
            raise MemoryError()

    def __reduce__(self):
        return (
          type(self), 
          (
            self.n_outputs, 
            self.n_relevant_outputs, 
            self.n_features, 
            self.n_y,
            self.n_samples, 
            self.max_node_samples, 
            self.random_state
          ), 
          self.__getstate__()
        )

    cdef int init(
      self, 
      const DOUBLE_t[ : , : : 1] y, 
      DOUBLE_t* sample_weight,
      double weighted_n_samples,
      SIZE_t* samples
    ) 
    nogil except -1:

      self.y = y
      self.sample_weight = sample_weight
      self.samples = samples
      self.weighted_n_samples = weighted_n_samples

      return 0

    cdef int node_reset(
      self, 
      SIZE_t start, 
      SIZE_t end
    ) 
    nogil except -1:
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_node_samples = 0.

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t y_ik
        cdef DOUBLE_t w_y_ik
        cdef DOUBLE_t w = 1.0

        self.sq_sum_total = 0.0
        memset(
          self.sum_total, 
          0, 
          self.n_outputs * sizeof(double)
        )

        for p in range(start, end):
            i = self.samples[p]

            if self.sample_weight != NULL:
                w = self.sample_weight[i]

            for k in range(self.n_outputs):
                y_ik = self.y[i, k]
                w_y_ik = w * y_ik
                self.sum_total[k] += w_y_ik
                self.sq_sum_total += w_y_ik * y_ik

            self.weighted_n_node_samples += w

        # Reset to pos=start
        self.reset()
        return 0

    cdef int reset(self) nogil except -1:
        cdef SIZE_t n_bytes = self.n_outputs * sizeof(double)
        memset(self.sum_left, 0, n_bytes)
        memcpy(self.sum_right, self.sum_total, n_bytes)

        self.weighted_n_left = 0
        self.weighted_n_right = self.weighted_n_node_samples
        self.pos = self.start

        return 0

    cdef int reverse_reset(self) nogil except -1:
        cdef SIZE_t n_bytes = self.n_outputs * sizeof(double)
        memset(self.sum_right, 0, n_bytes)
        memcpy(self.sum_left, self.sum_total, n_bytes)

        self.weighted_n_right = 0
        self.weighted_n_left = self.weighted_n_node_samples
        self.pos = self.end

        return 0

    cdef int update(self, SIZE_t new_pos) nogil except -1:
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double* sum_total = self.sum_total

        cdef SIZE_t* samples = self.samples
        cdef DOUBLE_t* sample_weight = self.sample_weight

        cdef SIZE_t pos = self.pos
        cdef SIZE_t end = self.end
        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t w = 1.0

        if (new_pos - pos) <= (end - new_pos):
            for p in range(pos, new_pos):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    sum_left[k] += w * self.y[i, k]

                self.weighted_n_left += w
        else:
            self.reverse_reset()

            for p in range(end - 1, new_pos - 1, -1):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    sum_left[k] -= w * self.y[i, k]

                self.weighted_n_left -= w

        self.weighted_n_right = (self.weighted_n_node_samples -
                                 self.weighted_n_left)
        for k in range(self.n_outputs):
            sum_right[k] = sum_total[k] - sum_left[k]

        self.pos = new_pos

        return 0

    cdef double node_impurity(self) nogil:
        pass

    cdef void children_impurity(
      self, 
      double* impurity_left,
      double* impurity_right
    ) 
    nogil:
        pass

    cdef void node_value(self, double* dest) nogil:
        cdef SIZE_t k

        for k in range(self.n_outputs):
            dest[k] = self.sum_total[k] / self.weighted_n_node_samples


cdef class MSE(RegressionCriterion):
    cdef double node_impurity(self) nogil:
        cdef double* sum_total = self.sum_total
        cdef double impurity
        cdef SIZE_t k

        impurity = self.sq_sum_total / self.weighted_n_node_samples
        for k in range(self.n_outputs):
            impurity -= (sum_total[k] / self.weighted_n_node_samples) ** 2.0

        return impurity / self.n_outputs

    cdef double proxy_impurity_improvement(self) nogil:
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right

        cdef SIZE_t k
        cdef double proxy_impurity_left = 0.0
        cdef double proxy_impurity_right = 0.0

        for k in range(self.n_outputs):
            proxy_impurity_left += sum_left[k] * sum_left[k]
            proxy_impurity_right += sum_right[k] * sum_right[k]

        return (proxy_impurity_left / self.weighted_n_left +
                proxy_impurity_right / self.weighted_n_right)

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        cdef DOUBLE_t* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t pos = self.pos
        cdef SIZE_t start = self.start

        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef DOUBLE_t y_ik

        cdef double sq_sum_left = 0.0
        cdef double sq_sum_right

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t w = 1.0

        for p in range(start, pos):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i]

            for k in range(self.n_outputs):
                y_ik = self.y[i, k]
                sq_sum_left += w * y_ik * y_ik

        sq_sum_right = self.sq_sum_total - sq_sum_left

        impurity_left[0] = sq_sum_left / self.weighted_n_left
        impurity_right[0] = sq_sum_right / self.weighted_n_right

        for k in range(self.n_outputs):
            impurity_left[0] -= (sum_left[k] / self.weighted_n_left) ** 2.0
            impurity_right[0] -= (sum_right[k] / self.weighted_n_right) ** 2.0

        impurity_left[0] /= self.n_outputs
        impurity_right[0] /= self.n_outputs
