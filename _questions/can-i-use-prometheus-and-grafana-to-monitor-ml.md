---
title: "Can I use Prometheus and Grafana to monitor ML model accuracy?"
section: database-systems
order: 27
description: "They work for latency, errors, and input statistics, but poorly for metrics like accuracy that join predictions with later labels. In The Modern ML Monitoring Mess (part 3), we tried to monitor cumulative accuracy for a toy taxi-tip model with Prometheus. It needed high-cardinality joins and convoluted PromQL, results depended on the scrape interval, and accuracy couldn't be computed over sliding windows."
sources:
  - title: "The Modern ML Monitoring Mess: Failure Modes in Extending Prometheus (3/4)"
    url: https://www.sh-reya.com/blog/rethinking-ml-monitoring-3/
asked:
  - https://www.reddit.com/r/sre/comments/1skf9ck/ml_on_top_of_prometheusthanos_anyone_actually/
---
