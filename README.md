# cz_grf: Calvin Zhang's Generalised Random Forest

1. General Tree:

- Tree (Cython): c_tree.pxd

- Splitter of Tree (Cython): cy_tree_splitter.pxd (header), cy_tree_splitter.pyx (source)

- Criterion of Tree (Cython): cy_tree_criterion.pxd (header), cy_tree_criterion.pyx (source)

- Tree (Python): py_tree.py

<br>

2. GRF Tree:

- GRF Tree (Python): py_grf_tree.py

- Criterion of GRF Tree (Cython): cy_grf_tree_criterion.pxd

<br>

3. Utilities:

- Utilities (Python): py_utilities.py
