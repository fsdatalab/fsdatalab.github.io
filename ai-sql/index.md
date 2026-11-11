---
layout: default
title: What Is AI-SQL?
description: "AI-SQL extends SQL with functions that call an LLM on each row. What it is, why it is expensive, and how our open-source engine Quail runs it faster."
permalink: /ai-sql/
---
<article class="content-narrow prose" markdown="1">

<div class="post-header"><h1>What is AI-SQL?</h1></div>

AI-SQL, also called AI functions, extends SQL with functions that call an
LLM. You write each function as a natural-language prompt. For example,
this query keeps the reviews that discuss the ending:

```sql
SELECT *
FROM reviews AS r
WHERE AI.IF(PROMPT('Does this review discuss the ending?\n\n{0}', r.review));
```

Snowflake, BigQuery, Databricks, and MotherDuck all support AI-SQL.

Running it is expensive. An AI function evaluates its prompt row by row, so
one query can make hundreds of thousands or millions of model calls. A
filter needs one LLM call per row. A naive join needs one LLM call for every
pair of rows in its two input tables.

Our lab is building [Quail](https://github.com/fsdatalab/quail), an
open-source inference engine for AI functions in SQL. Quail plans the query
and the LLM inference together, on open-weight models. Across 29
[QUAIL-B](https://github.com/fsdatalab/quail-bench) queries, it is 1.84x
faster on average than well-tuned vLLM baselines.

## Posts

{% assign ai_sql_posts = site.tags["ai-sql"] | where_exp: "post", "post.unlisted != true" | where_exp: "post", "post.date <= site.time" %}
<div class="topic-posts">
{%- for post in ai_sql_posts %}
<a href="{{ post.url | relative_url }}" class="post-item">
  <div class="post-title">{{ post.title }}</div>
  <div class="post-meta">{{ post.date | date: "%b %-d, %Y" }} | {% include author-names.html authors=post.authors %}</div>
  <div class="post-description">{{ post.description }}</div>
</a>
{%- endfor %}
</div>

## Code

- [Quail](https://github.com/fsdatalab/quail): the engine.
- [QUAIL-B](https://github.com/fsdatalab/quail-bench): the benchmark of
  AI-SQL queries we evaluate on.

</article>
