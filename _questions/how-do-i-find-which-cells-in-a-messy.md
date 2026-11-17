---
title: "How do I find which cells in a messy Jupyter notebook actually produced this plot or model, so I can clean it up?"
section: interfaces
order: 6
description: "Use a program slicer, which traces the code a result depends on. Static slicers such as Gather read the code without running it, so they keep extra cells. Our nbslicer tracks dependencies while cells run; on 374 real notebook sessions, its slices were on average 66% smaller than a static slicer's and still correct, and they ran 1.34x faster."
guides: [data-interfaces]
sources:
  - title: "Bolt-on, Compact, and Rapid Program Slicing for Notebooks (VLDB 2023)"
    url: https://smacke.net/papers/nbslicer.pdf
asked:
  - https://devblogs.microsoft.com/python/gather-a-new-way-to-clean-notebooks
  - https://discourse.jupyter.org/t/extract-specific-cells-from-students-notebooks/7951
---
