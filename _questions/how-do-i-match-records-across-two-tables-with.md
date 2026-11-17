---
title: "How do I match records across two tables with an LLM without paying for every pair?"
section: database-systems
order: 8
description: "Use Featurized-Decomposition Join, our method for semantic joins: an LLM extracts features from each record and combines them into logical rules that drop non-matching pairs with cheap comparisons, with statistical guarantees on result quality. Embedding similarity, the usual filter, often fails to predict matches. FDJ cut cost up to 10x compared with prior methods at the same guarantees."
how: "A semantic join asks an LLM whether each pair of records matches, so cost grows with the number of pairs. FDJ first has an LLM extract features from each record, such as a product's brand or size. It combines the features into rules, such as \"brands must be equal\", that rule out most non-matching pairs with cheap comparisons, and sends only the remaining pairs to the LLM. It chooses the rules so the result still meets your quality target with high probability."
sources:
  - title: "Featurized-Decomposition Join: Low-Cost Semantic Joins with Guarantees (VLDB 2026)"
    url: https://arxiv.org/abs/2512.05399
asked:
  - https://datascience.stackexchange.com/questions/137980/best-approach-for-matching-the-same-product-across-20-large-retail-catalogs
  - https://news.ycombinator.com/item?id=38968751
---
