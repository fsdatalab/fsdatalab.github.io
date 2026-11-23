---
title: "When I change an earlier Jupyter cell, how do I know which later cells to rerun?"
section: interfaces
order: 7
description: "Compute a forward slice: the cells that read anything the changed cell wrote. Reactive kernels such as ipyflow, and notebooks such as marimo, do this for you. In nbslicer, we compute forward slices from dependencies seen at run time; on 374 real notebook sessions they were on average 54% smaller than a static slicer's, so fewer cells need to rerun."
guides: [data-interfaces]
sources:
  - title: "Bolt-on, Compact, and Rapid Program Slicing for Notebooks (VLDB 2023)"
    url: https://smacke.net/papers/nbslicer.pdf
asked:
  - https://stackoverflow.com/questions/67622132/a-way-to-instruct-python-in-jupyter-to-run-selected-previous-cells
  - https://stackoverflow.com/questions/37190931/indicator-for-the-modified-code-in-cell-at-jupyter-notebooks
---
