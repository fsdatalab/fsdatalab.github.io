---
layout: post
title: "Estimating Costs for AI-Powered Filters"
date: 2026-09-30
author: "Arnav Dhariya, Shreya Shankar"
permalink: /blog/ai-filter-cost-estimates/
math: true
description: "A roofline-based cost model for AI-powered SQL filters that estimates FLOPs, HBM traffic, KV-cache storage, and latency, and extends to conjunctions of filters with an optimal ordering rule."
---

<aside class="tldr"><strong>TL;DR:</strong> How fast could an AI-SQL query run on a given LLM and GPU? We walk through how to estimate speed-of-light (SoL) latency for individual filters and conjunctions of filters, providing a baseline for evaluating system performance. SoL estimates power <a href="https://github.com/fsdatalab/quail">Quail</a>'s cost models. You can try out our <a href="#6-conjunction-of-filters-playground">interactive playground</a> to explore how filter ordering affects estimated latency on Qwen3-4B and an H100.</aside>

<nav class="post-toc" aria-label="Table of contents">
<strong>Contents</strong>
<ol>
  <li><a href="#1-introduction">Introduction.</a></li>
  <li><a href="#2-background">Background.</a>
    <ol>
      <li><a href="#21-ai-powered-filters">AI-powered filters.</a></li>
      <li><a href="#22-gpu-execution">GPU Execution.</a></li>
      <li><a href="#23-transformer-forward-pass">Transformer Forward Pass.</a></li>
      <li><a href="#24-roofline-model--speed-of-light">Roofline Model &amp; Speed of Light.</a></li>
    </ol>
  </li>
  <li><a href="#3-cost-model-for-one-filter">Cost Model for One Filter.</a>
    <ol>
      <li><a href="#31-workload-and-notation">Workload and Notation.</a></li>
      <li><a href="#32-projection-cost">Projection Cost.</a></li>
      <li><a href="#33-attention-cost">Attention Cost.</a></li>
      <li><a href="#34-mlp-cost">MLP Cost.</a></li>
      <li><a href="#35-total-cost">Total Cost.</a></li>
      <li><a href="#36-cost-of-one-imdb-filter">Cost of One IMDB Filter.</a></li>
    </ol>
  </li>
  <li><a href="#4-cost-model-for-a-conjunction-of-filters">Cost Model for a Conjunction of Filters.</a>
    <ol>
      <li><a href="#41-kv-reuse-and-hbm-capacity">KV Reuse and HBM Capacity.</a></li>
      <li><a href="#42-cost-of-a-fixed-filter-order">Cost of a Fixed Filter Order.</a></li>
      <li><a href="#43-choosing-the-filter-order">Choosing the Filter Order.</a></li>
    </ol>
  </li>
  <li><a href="#5-example-imdb-query">Example: IMDB Query.</a>
    <ol>
      <li><a href="#51-filter-ordering">Filter Ordering.</a></li>
      <li><a href="#52-cost-of-the-ordered-conjunction">Cost of the Ordered Conjunction.</a></li>
      <li><a href="#53-speed-of-light-estimate">Speed-of-Light Estimate.</a></li>
    </ol>
  </li>
  <li><a href="#6-conjunction-of-filters-playground">Conjunction of Filters Playground.</a></li>
  <li><a href="#7-conclusion">Conclusion.</a></li>
</ol>
</nav>

# 1. Introduction

We recently released [Quail](https://fsdatalab.github.io/blog/introducing-quail/)<sup><a href="#note-14">14</a></sup>, an execution engine for AI-SQL, which extends SQL with functions that call LLMs<sup><a href="#note-7">7</a></sup><sup>,</sup><sup><a href="#note-12">12</a></sup><sup>,</sup><sup><a href="#note-13">13</a></sup>. To choose between query plans, Quail needs to estimate how long each AI operation will take. This estimate depends on the input length, the model, the GPU, and how much state from earlier calls (the KV cache) can be reused.

**How do we estimate the latency of an AI-SQL query on a given LLM and GPU?**

One approach is to profile the system by running representative queries and fitting a latency model to the measurements. But a profile has to be redone for every new model, GPU, or workload. Instead, we use the roofline model to estimate the latency of an AI operation.<sup><a href="#note-16">16</a></sup> That is, we compute the arithmetic and memory traffic needed for a query, then divide by the GPU's peak compute throughput and memory bandwidth. The resulting latency is a lower bound because it assumes that the GPU sustains these peak rates. Quail uses this *speed-of-light* (SoL) latency to compare query plans.

In this article, we'll discuss:

- The GPU and transformer concepts behind the cost model.
- How to derive the work and latency of a single AI filter.
- How reuse, selectivity, and filter order affect a conjunction of filters.
- An example of applying our cost model to a query running Qwen3-4B on an NVIDIA H100.
- An interactive playground for building conjunctions of filters and comparing their estimated latency.

# 2. Background

We first define AI-powered filters. We then describe the H100 GPU and Qwen3-4B model used in our examples, including the parts of a transformer forward pass that affect cost. Finally, we introduce the roofline model we use to estimate SoL latency.

## 2.1 AI-powered filters

We define AI-powered filters and introduce the single-filter query and the conjunction of filters used throughout the article.

**AI-SQL filter.** An AI-SQL filter is a SQL-style predicate, i.e. a simple comparison like `WHERE price > 10`, where the condition is instead answered by an LLM, which reads each row (or document) and returns a true/false, one-token, judgment on whether it satisfies the predicate. AI-powered filters can handle conditions that SQL can't express, although at the cost of running a model call per row instead of a cheap comparison.

<figure class="figure-full" id="figure-1">
  <img src="{{ '/assets/blog/ai-filter-cost-estimates/figure-1.svg' | relative_url }}" alt="A database instance of four IMDB reviews, an AI_IF conjunction query applying three predicates, and the resulting table of TRUE/FALSE/— outcomes per review.">
  <figcaption>Figure 1. Simplified IMDB instance and query with results for 3 predicates, evaluated in the order written; the reviews are illustrative.</figcaption>
</figure>

[Figure 1](#figure-1) displays the general prompt structure for the AI-powered filter query; its structure is similar to that of a SQL query. Within each invocation of the operator there exists a preamble, "DOCUMENT:\n", the actual document, and the filter instruction "Evaluate TRUE or FALSE for the following question: [...]". The output of each predicate is a single-token output of true or false. Only the surviving documents pass through to subsequent filters. [Figure 1](#figure-1) shows a query with a conjunction of filters: $$F_1$$ (mentions a positive aspect) &rarr; $$F_4$$ (discusses the ending) &rarr; $$F_5$$ (mentions a named actor), all three run over the `reviews` table. We use the first predicate as the single AI-powered filter example in [Section 3](#3-cost-model-for-one-filter).

The formatting of the preamble, document, and filter instruction together influence the cost model, as each predicate carries the same amount of tokens around the variable-length document.

## 2.2 GPU Execution

We run our examples on an NVIDIA H100 SXM GPU. Our cost model depends on three properties of the GPU: its peak arithmetic throughput, HBM bandwidth, and HBM capacity.

<figure class="figure-full" id="gpu">
  <img src="{{ '/assets/blog/ai-filter-cost-estimates/gpu.svg' | relative_url }}" alt="H100 memory hierarchy diagram showing HBM3, L2 cache, and an SM with registers, shared memory, and tensor cores.">
  <figcaption>Figure 2. H100 memory layout and forward-pass data movement.</figcaption>
</figure>

[Figure 2](#gpu) shows the parts of the H100 that affect our cost model. *High-bandwidth memory (HBM)* is the GPU's main memory. It stores the model's weight matrices, embedding table, and KV cache. Data read from HBM passes through the shared L2 cache before reaching one of the GPU's 132 *streaming multiprocessors (SMs)*. Within each SM, registers and the combined shared memory and L1 cache hold small tiles of data close to the Tensor Cores. Each SM has four Tensor Cores, which perform the matrix multiplications used by the model. The H100 SXM has 80 GB of HBM3 and a 50 MB L2 cache shared by all of its SMs.<sup><a href="#note-11">11</a></sup>

Our cost model does not model each level of the cache separately. It counts the bytes read from or written to HBM and divides that amount by the HBM bandwidth. It also counts the *floating-point operations (FLOPs)* performed by the Tensor Cores and divides that amount by their arithmetic throughput.

[Table 1](#table-1) lists the hardware constants used in our examples. *Arithmetic throughput* is the number of FLOPs that the GPU can perform per second, and *memory bandwidth* is the number of bytes that it can transfer from HBM per second. $$\Pi_{bf16}$$ and $$\Pi_{fp8}$$ are the peak arithmetic throughput rates for BF16 and FP8 operations, while $$\beta$$ is the peak HBM bandwidth. We use the FP8 rate for the projection and MLP matrix multiplications, and the BF16 rate for attention in our FlashAttention<sup><a href="#note-3">3</a></sup> setup.

<div class="table-wrap" id="table-1" markdown="1">

| Symbol | Value |
|--------|-------|
| $$\Pi_{bf16}$$ | $$989.5\times10^{12}$$ FLOP/s |
| $$\Pi_{fp8}$$ | $$1.979\times10^{15}$$ FLOP/s |
| $$\beta$$ | $$3.35\times10^{12}$$ bytes/s |
| Capacity | 80 GB |

<p class="table-caption">Table 1. H100 SXM hardware constants.</p>
</div>

## 2.3 Transformer Forward Pass

The architecture of the LLM determines how many operations it performs and how much data it moves. We also need to account for the KV cache, which allows the model to reuse previously processed prefixes. We follow the standard accounting of transformer inference arithmetic.<sup><a href="#note-6">6</a></sup>

We use the Qwen3-4B FP8 model<sup><a href="#note-17">17</a></sup> in our examples.<sup><a href="#note-2b">†</a></sup>

<figure class="figure-full" id="model">
  <img src="{{ '/assets/blog/ai-filter-cost-estimates/model.svg' | relative_url }}" alt="Diagram of a token passing through one Qwen3-4B layer: QKV projection, attention with a KV cache, output projection, and the gate/up/SwiGLU/down MLP block.">
  <figcaption>Figure 3. Path of a single token through one Qwen3-4B layer. Only the attention step reads the KV cache of all <em>T</em> tokens.</figcaption>
</figure>

[Figure 3](#model) follows one token through a layer of Qwen3-4B. Each layer contains four attention projection matrices and three MLP matrices.

The query, key, and value matrices project the token's 2,560-element hidden state into three vectors. The *query vector* contains the information used to search the current and earlier tokens. Each *key vector* contains the information used to match a token, and each *value vector* contains the information returned by that match. During attention, the model compares the query vector with the key vectors and uses the resulting scores to combine their value vectors. The output projection then maps the attention result back to the model's hidden width.

Qwen3-4B uses *grouped query attention (GQA)*.<sup><a href="#note-1">1</a></sup> It has 32 query heads and eight key and value heads, so every four query heads share one key and value head. The key and value matrices are therefore one quarter of the size of the query and output matrices. The smaller key and value matrices also reduce the size of the KV cache.

The MLP processes each token independently through the gate, up, and down matrices. The gate and up matrices operate on the same input. The model combines their outputs with the SwiGLU<sup><a href="#note-15">15</a></sup> activation and uses the down matrix to project the result back to the hidden width. The MLP cost per token does not grow with the number of earlier tokens, while the attention cost does. Qwen3-4B repeats the attention and MLP blocks across all 36 layers.

[Table 2](#table-2) lists the model constants used by our cost model. $$d_{\text{model}}$$ is the *hidden-state width*. $$n_{\text{heads}}$$ and $$n_{kv}$$ are the numbers of query heads and key-value heads, while $$d_{\text{head}}$$ is the width of each attention head. $$d_{\text{mlp}}$$ is the *MLP intermediate width*, and vocab is the *vocabulary size*, the number of distinct tokens the model can read or output. The *embedding table* maps each token id to a $$d_{\text{model}}$$-dimensional vector, and the same table is *tied* to the output head that reads the final hidden state back into a distribution over the vocabulary; its size is $$\text{vocab}\times d_{\text{model}}$$. $$P$$ below counts only the non-embedding parameters, the projection and MLP weights, since the embedding table is priced separately in [Section 4.1](#41-kv-reuse-and-hbm-capacity).

<div class="table-wrap" id="table-2" markdown="1">

| Symbol | Value |
|--------|-------|
| $$P$$ | $$3.6\times10^{9}$$ non-embedding parameters |
| layers | 36 |
| $$d_{\text{model}}$$ | 2560 |
| $$n_{\text{heads}}$$ | 32 |
| $$n_{\text{kv}}$$ | 8 |
| $$d_{\text{head}}$$ | 128 |
| $$d_{\text{mlp}}$$ | 9728 |
| vocab | 151936 |

<p class="table-caption">Table 2. Qwen3-4B architecture constants.</p>
</div>

The *KV cache* stores the key and value vectors of tokens that the model has already processed. When two requests share a prefix, the second request can read the prefix's KV from the cache instead of processing the prefix again. This reuse is central to the cost model for a conjunction of filters in [Section 4](#4-cost-model-for-a-conjunction-of-filters).

## 2.4 Roofline Model & Speed of Light

The transformer forward pass determines how many FLOPs the model performs and how many bytes it moves from HBM. The roofline model converts these two quantities into a latency estimate. The model is due to Williams et al.<sup><a href="#note-16">16</a></sup>, and roofline analyses of LLM serving that we found helpful include earlier work on this cost model.<sup><a href="#note-4">4</a></sup><sup>,</sup><sup><a href="#note-9">9</a></sup><sup>,</sup><sup><a href="#note-10">10</a></sup>

A GPU spends time performing arithmetic and moving data from HBM. We call these *compute time* and *memory transfer time*. The roofline model assumes that the GPU can overlap computation with data transfer, so the slower of the two determines the latency:

$$
T = \max\!\left(\frac{\text{FLOPs}}{\Pi},\; \frac{\text{Bytes Moved}}{\beta}\right).
$$

Here, $$\Pi$$ is the arithmetic throughput in FLOP/s and $$\beta$$ is the HBM bandwidth in bytes/s. Compute time is the number of FLOPs divided by $$\Pi$$, while memory transfer time is the number of bytes moved divided by $$\beta$$. When we use the GPU's peak arithmetic throughput and peak HBM bandwidth, the equation gives the *speed-of-light (SoL) latency*. The SoL latency is a lower bound because an implementation may not sustain both peak rates or overlap computation and data transfer perfectly. We use the peak rates in our examples.

An operation's *operational intensity*, $$I$$, is the number of FLOPs it performs per byte moved from HBM. The *ridge point*, $$I^*$$, is the operational intensity at which compute time and memory transfer time are equal:

$$
\mathrm{Ridge} = I^{*} = \frac{\Pi}{\beta}.
$$

For the H100's peak FP8 throughput and HBM bandwidth, the ridge point is

$$
I^{*} = \frac{\Pi_{fp8}}{\beta} = \frac{1.979\times10^{15}}{3.35\times10^{12}} \approx 590.746\ \text{FLOP/byte}.
$$

An operation below 590.75 FLOP/byte is *memory-bound* because memory transfer takes longer than computation. An operation above 590.75 FLOP/byte is *compute-bound* because computation takes longer than memory transfer.

Attention runs in BF16, whose peak rate is half of FP8's, so it has its own ridge point:

$$
I^{*}_{bf16} = \frac{\Pi_{bf16}}{\beta} = \frac{989.5\times10^{12}}{3.35\times10^{12}} \approx 295.37\ \text{FLOP/byte}.
$$

Attention is memory-bound below 295.37 FLOP/byte and compute-bound above it. The projections and the MLP use the FP8 ridge point of 590.75. Attention is compared against the BF16 ridge point of 295.37 FLOP/byte instead, because it runs at $$\Pi_{bf16}$$; attention commonly stays in BF16 even when the GEMMs run in lower precision.<sup><a href="#note-4">4</a></sup>

[Figure 4](#figure-4) plots attainable compute throughput against operational intensity. The sloped region contains memory-bound operations, whose throughput increases as they perform more FLOPs per byte. The horizontal region contains compute-bound operations, whose throughput cannot exceed the GPU's peak arithmetic rate. The point where the two regions meet is the ridge point.

<figure class="figure-full" id="figure-4">
  <img src="{{ '/assets/blog/ai-filter-cost-estimates/roofline.svg' | relative_url }}" alt="Log-log roofline plot for an NVIDIA H100 SXM showing the memory-bound and compute-bound regions, the FP8 and BF16 ridge points, and the operating points of the example single filter and conjunction.">
  <figcaption>Figure 4. Roofline for an NVIDIA H100 SXM. The BF16 roof applies only to attention; &times; marks the single filter of Section 3, diamond marks the conjunction of Section 5.</figcaption>
</figure>

With the H100 limits and Qwen3-4B architecture defined, we can now derive the cost of one AI-powered filter.

# 3. Cost Model for One Filter

Section 2 described the three components included in our cost model: projections, attention, and the MLP. We now derive the computation and HBM traffic for each component and use the roofline equation to estimate its latency. The sum of the three latency estimates is the cost of one AI-powered filter. We finish by working through the cost of one filter from our example query.

## 3.1 Workload and Notation

Here, we define equations for the workload giving the request length and total token count, and introduce the notion of the chunk budget.

Suppose a batch has $$N$$ documents, where a document, $$i$$, has $$\ell_i$$ tokens. Each request has a prefix and the actual filter instruction around the document: $$q_{\text{pre}} + \text{document} + q_{\text{tail}}$$, so,

$$
\begin{aligned}
L_1 &= \sum_i \ell_i, \\
r_i &= q_{\text{pre}} + \ell_i + q_{\text{tail}}, \\
n_{\text{tok}} &= \sum_i r_i
    = L_1 + N(q_{\text{pre}} + q_{\text{tail}}), \\
L_2' &= \sum_i r_i^2.
\end{aligned}
$$

In case per-document lengths are unavailable, one can approximate $$r_i$$ by the document length mean; Quail does this in its cost model. $$L_2'$$ is necessary for self-attention computations in [Section 3.3](#33-attention-cost). Weights are re-streamed from the HBM once per forward pass, and each pass is bounded by the chunk budget, which is essential to compute the number of passes that will occur. Below, we provide the formula to compute the chunk budget, $$C$$, and as a result, the number of forward passes, $$K$$,

$$
\begin{aligned}
C &= \left\lfloor \frac{2^{31}-1}{2\,d_{\text{mlp}}} \right\rfloor, \\
K &= \left\lceil \frac{n_{\text{tok}}}{C} \right\rceil .
\end{aligned}
$$

The chunk budget is bounded by the widest intermediate activation in the model, $$d_{\text{mlp}}$$, and the slot-indexing counter, $$2^{31} - 1$$. By dividing the total number of tokens in the documents by the chunk budget, we obtain the number of forward passes. [Section 4.1](#41-kv-reuse-and-hbm-capacity) revisits this budget with the memory bound that the full engine also enforces.

## 3.2 Projection Cost

Here we derive the FLOPs for Q, K, V and output projections, their HBM traffic across forward passes, and the projection roofline cost.

Q, K, V, and output projections are matrix multiplications over each token. Each matrix maps a vector of one width to another, so its parameter count is the product of its input and output widths. Per layer we have the following for each projection:

$$
\begin{aligned}
p_Q &= d_{\text{model}}\,n_{\text{heads}}\,d_{\text{head}}, \\
p_K &= d_{\text{model}}\,n_{kv}\,d_{\text{head}}, \\
p_V &= d_{\text{model}}\,n_{kv}\,d_{\text{head}}, \\
p_O &= n_{\text{heads}}\,d_{\text{head}}\,d_{\text{model}}.
\end{aligned}
$$

Thus, in every layer they hold $$2\,d_{\text{model}}\,d_{\text{head}}(n_{\text{heads}}+n_{kv})$$ parameters. The totals for parameters, FLOPs, and memory transfer are given below along with the total projection cost.

$$
\begin{aligned}
P_{\text{proj}} &= \text{layers}\cdot 2\,d_{\text{model}}\,d_{\text{head}}(n_{\text{heads}}+n_{kv}), \\
F_{\text{proj}} &= 2\,P_{\text{proj}}\,n_{\text{tok}}, \\
B_{\text{proj}} &= b_w\,P_{\text{proj}}\,K, \\
T_{\text{proj}} &= \max\!\left(\frac{F_{\text{proj}}}{\Pi_{fp8}}, \frac{B_{\text{proj}}}{\beta}\right).
\end{aligned}
$$

Here $$P_{\text{proj}}$$ is the total projection parameters for all of the layers, $$F_{\text{proj}}$$ is the total FLOPs across the total tokens in $$N$$ documents. The 2 in $$F_{\text{proj}}$$ is one multiply plus one add, and $$b_w$$ is bytes per weight (1 for FP8). The value of $$K$$ represents the number of forward passes to process all $$n_{\text{tok}}$$ tokens. Thus, the total time is given by $$T_{\text{proj}}$$, which takes the maximum of compute time and memory transfer time for projections.

## 3.3 Attention Cost

Here we derive the attention comparisons from the request lengths, the attention FLOPs, the writes and reads for the KV cache, and the attention roofline cost.

Attention is different from the projections, as tokens here interact with each other. Each token attends to itself and to every earlier token in its own request. With an empty KV cache, a request of $$r_i$$ tokens makes $$1 + 2 + \dots + r_i - 1 + r_i$$ comparisons, so across the batch

$$
A = \sum_i \frac{r_i(r_i+1)}{2} = \frac{L_1' + L_2'}{2},
$$

where $$L_1' = \sum_i r_i$$ and $$L_2' = \sum_i r_i^2$$.

We introduced $$L_2'$$ in the prior section because it is needed for the closed form of the self-attention computation.

One comparison, for one head in one layer, costs $$2\,d_{\text{head}}$$ FLOPs for the query--key dot product ($$QK^\top$$), because we do a multiply and addition per number, and another $$2\,d_{\text{head}}$$ for the weighted sum over value vectors. Repeating across all $$n_{\text{heads}}$$ and all layers gives,

$$
F_{\text{attn}} = 4\,n_{\text{heads}}\,d_{\text{head}}\,\text{layers}\cdot A.
$$

With GQA, several query heads share one K/V head, however each query head must compute its own dot products. Each token writes one K row and one V row per layer, and tokens already in the cache must be read back. The KV size per token is

$$
B_{\text{kv}} = 2\,b_{kv}\,\text{layers}\,n_{kv}\,d_{\text{head}},
$$

where the $$2$$ counts K and V, and $$b_{kv}$$ is bytes per element (2 for BF16, as the model uses FlashAttention). Here we use K/V heads rather than query heads because the cache stores K and V heads once, which is why GQA shrinks the cache. With $$W$$ tokens written and $$R$$ cached tokens read,

$$
B_{\text{attn}} = B_{\text{kv}}\,(W + R).
$$

For one filter, $$W = n_{\text{tok}}$$ and $$R = 0$$, because there were no prior K/V vectors in the cache.

FlashAttention runs in BF16, so we must divide by its arithmetic rate.

$$
T_{\text{attn}} = \max\!\left(\frac{F_{\text{attn}}}{\Pi_{bf16}}, \frac{B_{\text{attn}}}{\beta}\right).
$$

## 3.4 MLP Cost

We still need to derive the FLOPs for the gate, up, and down MLP projections, their HBM traffic, and the MLP roofline cost.

The MLP acts on each token independently, so unlike attention its cost per token does not depend on the other tokens in the request. It has three matrices per layer, and the parameter count of each is the product of its input and output widths. $$W_{\text{gate}}$$ and $$W_{\text{up}}$$ map $$d_{\text{model}}$$ to $$d_{\text{mlp}}$$, while $$W_{\text{down}}$$ maps $$d_{\text{mlp}}$$ back to $$d_{\text{model}}$$:

$$
\begin{aligned}
p_{\text{gate}} &= d_{\text{model}}\,d_{\text{mlp}}, \\
p_{\text{up}}   &= d_{\text{model}}\,d_{\text{mlp}}, \\
p_{\text{down}} &= d_{\text{mlp}}\,d_{\text{model}}.
\end{aligned}
$$

Gate and up read the same input; their outputs are combined by SwiGLU (Swish Gated Linear Unit) activation and projected back down. The total parameters in the MLP are,

$$
P_{\text{mlp}} = \text{layers}\cdot(p_{\text{gate}} + p_{\text{up}} + p_{\text{down}}) = 3\,\text{layers}\,d_{\text{model}}\,d_{\text{mlp}}.
$$

The reasoning for the MLP is the same as it is for the projections: every token is multiplied by every weight once, and the weights are streamed from HBM once per forward pass.

$$
\begin{aligned}
F_{\text{mlp}} &= 2\,P_{\text{mlp}}\,n_{\text{tok}}, \\
B_{\text{mlp}} &= b_w\,P_{\text{mlp}}\,K.
\end{aligned}
$$

As GEMMs run in FP8, we obtain the following:

$$
T_{\text{mlp}} = \max\!\left(\frac{F_{\text{mlp}}}{\Pi_{fp8}}, \frac{B_{\text{mlp}}}{\beta}\right).
$$

## 3.5 Total Cost

A forward pass runs the projections, attention, and the MLP in sequence, so the three component times add:

$$
T_{\text{filter}} = T_{\text{proj}} + T_{\text{attn}} + T_{\text{mlp}}.
$$

For the projections and the MLP, every token is multiplied by every weight once per forward pass, so their operational intensity is

$$
I = \frac{F}{B} = \frac{2\,P\,n_{\text{tok}}}{b_w\,P\,K} = \frac{2}{b_w}\cdot\frac{n_{\text{tok}}}{K}.
$$

With FP8 weights ($$b_w = 1$$), the operational intensity is therefore *twice* the average number of tokens per forward pass — the 2 comes from counting each multiply-add as two FLOPs,<sup><a href="#note-6b">‡</a></sup> divided by FP8's one byte per weight. BF16 weights ($$b_w = 2$$) would instead give exactly one times the average tokens per pass, since the two bytes per weight cancel the multiply-add factor. These components are compute-bound once an average pass carries more than $$I^*/2 \approx 295$$ tokens, the same threshold other roofline analyses of LLM decoding derive.<sup><a href="#note-4">4</a></sup> Attention is compared against the BF16 ridge point of 295.37 FLOP/byte instead of the FP8 ridge point of 590.75 FLOP/byte, because it runs at $$\Pi_{bf16}$$; attention commonly stays in BF16 even when the GEMMs run in lower precision.<sup><a href="#note-4">4</a></sup>

## 3.6 Cost of One IMDB Filter

We now estimate the SoL latency of one filter from our example query. We use the H100's peak arithmetic throughput and HBM bandwidth, along with the following batch and prompt values:

$$
N = 5{,}000, \qquad
\bar{\ell} = 298.8466 \ \text{tokens}, \qquad
q_{\text{pre}} = 2 \ \text{tokens}, \qquad
q_{\text{tail}} = 51 \ \text{tokens}.
$$

We approximate every document by the mean length, which gives

$$
L_1 = N\bar{\ell} = 1{,}494{,}233 \ \text{tokens}.
$$

**Request and batch sizes.** Each request contains the prefix, one document, and the filter instruction:

$$
\begin{aligned}
r &= q_{\text{pre}} + \bar{\ell} + q_{\text{tail}}
  = 2 + 298.8466 + 51
  = 351.8466 \ \text{tokens}, \\
n_{\text{tok}} &= N r
   = 5{,}000 \times 351.8466
   = 1{,}759{,}233.
\end{aligned}
$$

**Attention comparisons.** The attention calculation also needs the sum of squared request lengths:

$$
\begin{aligned}
L_2' &= N r^2
    = 5{,}000 \times 351.8466^2
    = 618{,}980{,}149.66, \\
A &= \frac{n_{\text{tok}} + L_2'}{2}
    = 310{,}369{,}691.33 .
\end{aligned}
$$

**Forward passes.** The widest intermediate is the SwiGLU input, with $$2\,d_{\text{mlp}} = 19{,}456$$ slots per token. The slots are indexed by a signed 32-bit counter:

$$
\begin{aligned}
C &= \left\lfloor \frac{2^{31}-1}{19{,}456} \right\rfloor
    = 110{,}376, \\
K &= \left\lceil \frac{1{,}759{,}233}{110{,}376} \right\rceil
    = \lceil 15.94 \rceil = 16 .
\end{aligned}
$$

**Model parameters.** We substitute the Qwen3-4B dimensions from [Table 2](#table-2):

$$
\begin{aligned}
d_{\text{model}} &= 2560, &
d_{\text{mlp}} &= 9728, &
d_{\text{head}} &= 128, \\
n_{\text{heads}} &= 32, &
n_{kv} &= 8, &
\text{layers} &= 36.
\end{aligned}
$$

$$
\begin{aligned}
P_{\text{proj}} &= 36 \cdot 2 \cdot 2560 \cdot 128 \cdot (32+8)
                  = 943{,}718{,}400, \\
P_{\text{mlp}}  &= 3 \cdot 36 \cdot 2560 \cdot 9728
                  = 2{,}689{,}597{,}440 .
\end{aligned}
$$

**Projections.**

$$
\begin{aligned}
F_{\text{proj}} &= 2 \times 943{,}718{,}400 \times 1{,}759{,}233
                  \approx 3.3204\times10^{15}, \\
B_{\text{proj}} &= 1 \times 943{,}718{,}400 \times 16
                  \approx 1.5099\times10^{10}, \\
T_{\text{proj}} &= \max\!\left(
      \frac{3.3204\times10^{15}}{1.979\times10^{15}},\;
      \frac{1.5099\times10^{10}}{3.35\times10^{12}}\right) \\
    &= \max(1.6778,\ 0.0045)
    = 1.6778 \ \text{s}.
\end{aligned}
$$

**Attention.**

$$
\begin{aligned}
F_{\text{attn}} &= 4 \cdot 32 \cdot 128 \cdot 36 \times A
                  = 589{,}824 \times 310{,}369{,}691.33 \\
                  &\approx 1.8306\times10^{14}, \\
B_{\text{kv}}   &= 2 \cdot 2 \cdot 36 \cdot 8 \cdot 128
                  = 147{,}456 \ \text{bytes/token}, \\
B_{\text{attn}} &= 147{,}456 \times (1{,}759{,}233 + 0)
                  \approx 2.5941\times10^{11}, \\
T_{\text{attn}} &= \max\!\left(
      \frac{1.8306\times10^{14}}{989.5\times10^{12}},\;
      \frac{2.5941\times10^{11}}{3.35\times10^{12}}\right) \\
    &= \max(0.1850,\ 0.0774)
    = 0.1850 \ \text{s}.
\end{aligned}
$$

**MLP.**

$$
\begin{aligned}
F_{\text{mlp}} &= 2 \times 2{,}689{,}597{,}440 \times 1{,}759{,}233
                 \approx 9.4633\times10^{15}, \\
B_{\text{mlp}} &= 1 \times 2{,}689{,}597{,}440 \times 16
                 \approx 4.3034\times10^{10}, \\
T_{\text{mlp}} &= \max\!\left(
      \frac{9.4633\times10^{15}}{1.979\times10^{15}},\;
      \frac{4.3034\times10^{10}}{3.35\times10^{12}}\right) \\
    &= \max(4.7818,\ 0.0128)
    = 4.7818 \ \text{s}.
\end{aligned}
$$

**Total.**

$$
\begin{aligned}
T_{\text{filter}} &= T_{\text{proj}} + T_{\text{attn}} + T_{\text{mlp}} \\
    &= 1.6778 + 0.1850 + 4.7818 \\
    &= \boxed{6.64 \ \text{seconds}} .
\end{aligned}
$$

<div class="table-wrap" id="table-3" markdown="1">

| Component | FLOPs | Bytes | $$T_{\text{compute}}$$ | $$T_{\text{memory}}$$ |
|-----------|-------|-------|-----------------------|----------------------|
| Projections | $$3.32\times10^{15}$$ | $$1.51\times10^{10}$$ | 1.6778 s | 0.0045 s |
| Attention | $$1.83\times10^{14}$$ | $$2.59\times10^{11}$$ | 0.1850 s | 0.0774 s |
| MLP | $$9.46\times10^{15}$$ | $$4.30\times10^{10}$$ | 4.7818 s | 0.0128 s |
| **Total** | $$1.30\times10^{16}$$ | $$3.17\times10^{11}$$ | **6.64 s** | 0.09 s |

<p class="table-caption">Table 3. Cost of one IMDB filter (<em>F1</em>: mentions a positive aspect) on Qwen3-4B and an H100. Every component is compute-bound.</p>
</div>

# 4. Cost Model for a Conjunction of Filters

Section 3 estimates the cost of one filter with an empty KV cache. In a conjunction of filters, each successive filter processes only the documents that passed the preceding filters. Successive filters also reuse the prefix KV computed by the first filter. We first describe KV reuse and the HBM capacity needed to support it, then derive the cost of a fixed filter order and explain how to choose the order with the lowest cost.

## 4.1 KV Reuse and HBM Capacity

Taking a closer look at [Figure 1](#figure-1), every filter prompt for a given document contains the same preamble and document. Only the filter instruction changes. The average shared prefix length is therefore

$$
p = q_{\text{pre}} + \bar{\ell},
$$

where $$\bar{\ell}$$ is the mean document length. The first filter processes the prefix and stores its KV. Each later filter that evaluates the document reads the stored KV and processes only its own instruction tokens. Our cost model therefore processes each document's prefix once and reuses its KV for any later filters that evaluate the document.

This only works if every prefix KV is still in HBM when the later filters run. If one were evicted, it would have to be recomputed. We therefore choose a batch small enough that all of its prefix KV fits in HBM at once. To find that size, we need (1) how many bytes one token's KV takes, (2) how much HBM is left for the cache, and (3) how many prefixes fit in it.

**KV size per token.** Each token stores one key vector and one value vector in every layer. Using $$n_{kv}$$ rather than $$n_{\text{heads}}$$ reflects GQA, which stores each K/V head once even though four query heads share it:

$$
B_{\text{kv}} = 2\,b_{kv}\,\text{layers}\,n_{kv}\,d_{\text{head}} = 2 \cdot 2 \cdot 36 \cdot 8 \cdot 128 = 147{,}456 \ \text{bytes}.
$$

The 2 counts K and V, and $$b_{kv} = 2$$ bytes per BF16 element. Full multi-head attention would store 32 heads instead of 8 and need four times as much. An average prefix therefore takes $$B_{\text{kv}}\,p \approx 44.4$$ MB.

**HBM available for the KV cache.** The weights occupy HBM first, and what remains bounds the KV cache and hence the batch size.<sup><a href="#note-9">9</a></sup> Besides weights, the GPU needs scratch space for *activations*: the temporary intermediate vectors (the outputs of the projections, the gate and up matrices, and so on) that exist while a chunk of tokens moves through a layer. Unlike weights, activations depend on the input and are discarded once the next layer has consumed them, but every token in a chunk needs its own at the same time, so this space scales with the chunk size. K and V are the exception: we keep them, and they are priced separately as the KV cache above.

Quail allocates only a fraction $$\rho = 0.95$$ of the capacity to its memory pool. Within that pool it holds the resident weights, $$W_{\text{resident}}$$ (the projection and MLP weights, the embedding table, and the FP8 block scales), and a reserve for activations. The reserve covers $$\gamma = 2$$ chunks of $$C$$ tokens: one chunk is being computed on while the next is being constructed behind it, so the GPU does not stall between chunks, and both chunks' activation buffers must be resident at once. Each token needs $$a = 32\,d_{\text{model}}$$ bytes of activations,<sup><a href="#note-9b">§</a></sup> so one chunk reserves $$M_{\text{chunk}} = C\,a$$ bytes. The HBM left for the KV cache is

$$
\begin{aligned}
M_{\text{KV}} &= \rho\,\text{Capacity} - W_{\text{resident}} - M_{\text{act}}, \\
M_{\text{act}} &= \gamma\,M_{\text{chunk}} = \gamma\,C\,a.
\end{aligned}
$$

**The chunk budget revisited.** The chunk budget $$C$$ of [Section 3.1](#31-workload-and-notation) was bounded only by the slot-indexing counter. The full definition takes the smaller of two upper bounds, the index bound and a memory bound, and raises the result to a floor $$C_{\text{knee}}$$, the chunk size at which the dense projections reach the ridge point. It is found the same way as the ridge point itself in [Section 2.4](#24-roofline-model--speed-of-light): set compute time equal to memory time and solve for the chunk size, using the combined projection and MLP parameters, $$P_{\text{proj}}+P_{\text{mlp}}$$:

$$
\begin{aligned}
C_{\text{idx}} &= \left\lfloor \frac{2^{31}-1}{2\,d_{\text{mlp}}} \right\rfloor, \\
C_{\text{mem}} &= \left\lfloor \frac{\rho\,\text{Capacity} - W_{\text{resident}}}{\sigma\,a} \right\rfloor, \\
C_{\text{knee}} &= \frac{I^{*}\,(P_{\text{proj}}+P_{\text{mlp}})\,b_w}{2\,(P_{\text{proj}}+P_{\text{mlp}}) - I^{*}\,\text{IO}\,b_{\text{act}}}, \\
C &= \max\big(\min(C_{\text{mem}}, C_{\text{idx}}),\ C_{\text{knee}}\big),
\end{aligned}
$$

where $$\sigma = 2$$ is a slack factor. $$\text{IO}$$ is the number of activation elements the projections read in and write out per token: each projection's input width plus its output width, added up across all four projection matrices and all layers. $$b_{\text{act}} = 2$$ is the number of bytes each of those activation elements takes, since activations are stored in BF16. Intuitively, $$C_{\text{mem}}$$ and $$C_{\text{idx}}$$ are both ceilings on how large a chunk can be, one from leftover HBM and one from the kernel's addressing limit, while $$C_{\text{knee}}$$ is a floor on how small a chunk should be, below which the dense projections turn memory-bound; taking the minimum of the two ceilings picks whichever one actually constrains the chunk size, and taking the maximum with the floor then guarantees the engine never picks a chunk small enough to waste the GPU's compute.

**Qwen3-4B on an H100.** [Table 4](#table-4) works through the numbers. Quail measures $$W_{\text{resident}} = 4.50$$ GB as the model's resident footprint: 3.63 GB of FP8 weights and 0.78 GB of BF16 embeddings from the parameter counts of [Section 3.6](#36-cost-of-one-imdb-filter), plus 0.09 GB of FP8 block scales<sup><a href="#note-9c">¶</a></sup>, so $$W_{\text{resident}} = 3.63 + 0.78 + 0.09 = 4.50$$ GB. For Qwen3-4B, $$\text{IO} = 1{,}787{,}904$$, so

$$
C_{\text{knee}} = \frac{590.75 \times 3{,}633{,}315{,}840 \times 1}{2\times3{,}633{,}315{,}840 - 590.75\times1{,}787{,}904\times2} \approx 416.
$$

With $$a = 81{,}920$$ bytes per token, $$C_{\text{mem}} = 436{,}401$$ exceeds $$C_{\text{idx}} = 110{,}376$$, and $$C_{\text{knee}} \approx 416$$ is smaller than both, so $$C = C_{\text{idx}} = 110{,}376$$. One chunk then reserves $$M_{\text{chunk}} = C\,a \approx 9.04$$ GB, and the activation reserve in [Table 4](#table-4) covers $$\gamma = 2$$ of these chunks at once.

**Maximum batch size.** The maximum batch size is the number of average prefixes that fit in $$M_{\text{KV}}$$:

$$
N_b = \left\lfloor \frac{M_{\text{KV}}}{B_{\text{kv}}\,p} \right\rfloor = \left\lfloor \frac{53.42\ \text{GB}}{44.4\ \text{MB}} \right\rfloor = 1{,}204.
$$

We choose $$N \leq N_b$$ so that no prefix KV is evicted and recomputed. We therefore run the 5,000 documents as five batches. The chunk budget in [Section 3.1](#31-workload-and-notation) limits the number of tokens in one forward pass, while KV capacity limits the number of document prefixes that can remain cached between filters.

<div class="table-wrap" id="table-4" markdown="1">

| Item | Size (GB) |
|------|-----------|
| Usable pool, $$\rho$$ Capacity | 76.00 |
| &minus; Resident weights, $$W_{\text{resident}}$$ | 4.50 |
| &minus; Activation reserve, $$M_{\text{act}} = 2\times9.04$$ | 18.08 |
| **Left for KV cache, $$M_{\text{KV}}$$** | **53.42** |

<p class="table-caption">Table 4. HBM budget for Qwen3-4B on an H100 SXM.</p>
</div>

## 4.2 Cost of a Fixed Filter Order

Suppose a query applies $$m$$ filters to $$N$$ documents. Let $$\pi = (\pi_1,\ldots,\pi_m)$$ be an ordering, where $$\pi_j$$ is the filter in position $$j$$. *Selectivity*, $$s_i$$, is the fraction of documents that survive filter $$F_i$$. Each filter runs only on the documents that survived the filters before it, so the expected number of documents entering position $$j$$ is

$$
N_j = N \prod_{k<j} s_{\pi_{k}}.
$$

Each filter depends on four quantities: the tokens processed $$n$$, the attention comparisons $$A$$, the KV tokens written $$W$$, and the KV tokens read $$R$$.

For the first filter, let $$\ell_d$$ be the length of document $$d$$. Its request length is $$r_d = q_{\text{pre}} + \ell_d + q_{\pi_1}$$, and the cost is

$$
\begin{aligned}
n_1 &= \sum_{d=1}^{N} r_d, \\
A_1 &= \sum_{d=1}^{N}\frac{r_d(r_d+1)}{2}, \\
W_1 &= n_1, \\
R_1 &= 0 .
\end{aligned}
$$

For any subsequent filter, only the $$N_j$$ surviving documents run, and only the $$q_{\pi_j}$$ instruction tokens are computed:

$$
\begin{aligned}
n_j &= N_j\,q_{\pi_j}, \\
A_j &= N_j\left[\,q_{\pi_j}\,p + \frac{q_{\pi_j}(q_{\pi_j}+1)}{2}\right], \\
W_j &= n_j, \\
R_j &= N_j\,p .
\end{aligned}
$$

The first term of $$A_j$$ counts comparisons with the cached prefix, and the second counts comparisons among the instruction tokens. $$R_j$$ is the number of cached prefix tokens read from HBM.

## 4.3 Choosing the Filter Order

Reordering the filters can reduce the amount of work because later filters run on fewer documents. We therefore choose the filter order with the lowest estimated cost.

Each filter has two per-document costs because its cost depends on whether the prefix KV is already cached. A *scan cost*, $$\text{scan}_i$$, is the cost of processing an uncached prefix together with filter $$F_i$$'s instruction and writing the prefix KV. An *ask cost*, $$\text{ask}_i$$, is the cost of processing only the instruction tokens for $$F_i$$ and reading the cached prefix KV. For an ordering $$\pi$$, the first filter scans all $$N$$ documents. Each later filter asks only on the documents that passed the preceding filters. The total cost is

$$
C(\pi) = N\,\text{scan}_{\pi_1} + \sum_{j=2}^{m} \text{ask}_{\pi_j}\,N\prod_{k<j} s_{\pi_k}.
$$

The product is the fraction of documents that survive the filters preceding position $$j$$.

The ordering rule follows Hellerstein and Stonebraker's work on ordering expensive predicates<sup><a href="#note-5">5</a></sup>, extended to user-defined predicates by Chaudhuri and Shim<sup><a href="#note-2">2</a></sup>. For filters after the first position, the rule ranks each filter by its ask cost per document removed:

$$
\text{rank}_i = \frac{\text{ask}_i}{1 - s_i}.
$$

A filter with $$s_i \geq 1$$ removes no documents, so we set its rank to $$\infty$$ and place it last among the filters that ask.

Only the first filter pays a scan cost. For each filter $$f$$, we evaluate one ordering with $$f$$ first and the remaining filters sorted by ascending $$\text{rank}_i$$. We compute $$C(\pi)$$ for each ordering and choose the one with the lowest cost. The ranks and scan costs are per document, so the chosen order does not depend on $$N$$; $$N$$ only scales $$C(\pi)$$.

We compute the scan and ask costs with the equations from [Section 3](#3-cost-model-for-one-filter).

Overall, the cost equation estimates the latency of a fixed filter order, and the ordering rule selects the order with the lowest estimated latency.

# 5. Example: IMDB Query

Our example runs the query over $$N = 5{,}000$$ IMDB reviews<sup><a href="#note-8">8</a></sup> using Qwen3-4B on an H100. The weights use FP8, while attention and KV use BF16. The cached prefix contains $$p = 2 + 298.8466 = 300.8466$$ tokens per review.

## 5.1 Filter Ordering

[Table 5](#table-5) lists the three filters. We measure each filter's selectivity as the fraction of the documents reaching it for which the LLM returns True, in a run of Qwen3-4B FP8 that applied the filters in the order $$F_1 \to F_4 \to F_5$$. Each ask and scan cost is the sum of the three components of [Section 3](#3-cost-model-for-one-filter):

$$
\text{cost} = T_{\text{proj}} + T_{\text{mlp}} + T_{\text{attn}},
$$

where projections and the MLP take $$2\,P_{\text{proj}}\,n_{\text{tok}}/\Pi_{fp8}$$ and $$2\,P_{\text{mlp}}\,n_{\text{tok}}/\Pi_{fp8}$$. The inputs are the constants used throughout: $$P_{\text{proj}} = 943{,}718{,}400$$ and $$P_{\text{mlp}} = 2{,}689{,}597{,}440$$ are the parameter counts of [Section 3](#3-cost-model-for-one-filter); $$\Pi_{fp8} = 1.979\times10^{15}$$, $$\Pi_{bf16} = 989.5\times10^{12}$$ and $$\beta = 3.35\times10^{12}$$ are the H100 peak rates of [Table 1](#table-1); $$589{,}824 = 4\,n_{\text{heads}}\,d_{\text{head}}\,\text{layers} = 4 \cdot 32 \cdot 128 \cdot 36$$ is the attention FLOPs per comparison; and $$B_{\text{kv}} = 147{,}456$$ bytes/token is the KV size per token. The cached prefix is $$p = q_{\text{pre}} + \bar{\ell} = 2 + 298.8466 = 300.8466$$ tokens.

For $$F_1$$ ($$q = 51$$, the tail length of its predicate), an ask processes $$n = 51$$ tokens and has projection compute

$$
T_{\text{proj}} = \frac{2 \times 943{,}718{,}400 \times 51}{1.979\times10^{15}} = 0.0486~\text{ms},
$$

MLP compute

$$
T_{\text{mlp}} = \frac{2 \times 2{,}689{,}597{,}440 \times 51}{1.979\times10^{15}} = 0.1386~\text{ms},
$$

and attention cost. The 51 new tokens attend to the $$p$$ cached tokens and to each other, so

$$
A = q\,p + \frac{q(q+1)}{2} = 51 \times 300.8466 + \frac{51 \times 52}{2} = 16{,}669.18.
$$

The KV traffic reads the $$p$$ cached tokens and writes the $$q$$ new ones, so $$W + R = p + q = 351.8466$$ tokens:

$$
\begin{aligned}
T_{\text{attn}}
    &= \max\!\left(
        \frac{589{,}824 \times 16{,}669.18}{989.5\times10^{12}},\;
        \frac{147{,}456 \times 351.8466}{3.35\times10^{12}}\right) \\
    &= \max(0.0099,\ 0.0155)~\text{ms} = 0.0155~\text{ms},
\end{aligned}
$$

the larger of the attention compute and the KV traffic of the cached prefix plus new tokens, so

$$
\text{ask}_1 = T_{\text{proj}} + T_{\text{mlp}} + T_{\text{attn}} = 0.0486 + 0.1386 + 0.0155 = 0.203~\text{ms}.
$$

A scan processes the full request of $$n = r = p + q = 300.8466 + 51 = 351.8466$$ tokens and has projection compute

$$
T_{\text{proj}} = \frac{2 \times 943{,}718{,}400 \times 351.8466}{1.979\times10^{15}} = 0.336~\text{ms},
$$

MLP compute

$$
T_{\text{mlp}} = \frac{2 \times 2{,}689{,}597{,}440 \times 351.8466}{1.979\times10^{15}} = 0.956~\text{ms},
$$

and attention compute. With an empty cache, one request makes

$$
A = \frac{r(r+1)}{2} = \frac{351.8466 \times 352.8466}{2} = 62{,}073.94
$$

comparisons, so

$$
T_{\text{attn}} = \frac{589{,}824 \times 62{,}073.94}{989.5\times10^{12}} = 0.0370~\text{ms},
$$

which exceeds its memory transfer traffic ($$W = r$$, $$R = 0$$) of

$$
\frac{147{,}456 \times 351.8466}{3.35\times10^{12}} = 0.0155~\text{ms},
$$

so

$$
\text{scan}_1 = 0.336 + 0.956 + 0.037 = 1.329~\text{ms}.
$$

The other filters follow the same way with their own $$q$$ ($$45$$ for $$F_4$$, $$49$$ for $$F_5$$).

<div class="table-wrap" id="table-5" markdown="1">

| Filter | Predicate | $$q_{\text{tail}}$$ | $$s_i$$ | $$\text{ask}_i$$ (ms) | $$\text{scan}_i$$ (ms) | $$1-s_i$$ | rank (ms) |
|--------|-----------|--------|-------|------------|-------------|---------|-----------|
| $$F_4$$ | discusses the ending | 45 | 0.2273 | 0.180 | 1.306 | 0.7727 | 0.234 |
| $$F_1$$ | mentions a positive aspect | 51 | 0.4856 | 0.203 | 1.329 | 0.5144 | 0.394 |
| $$F_5$$ | mentions a named actor | 49 | 0.6123 | 0.195 | 1.321 | 0.3877 | 0.504 |

<p class="table-caption">Table 5. Filters sorted by ascending rank ask<sub>i</sub>/(1&minus;s<sub>i</sub>), giving the order F<sub>4</sub> &rarr; F<sub>1</sub> &rarr; F<sub>5</sub>. Each <em>s<sub>i</sub></em> is the fraction kept among documents reaching that filter, in run order F<sub>1</sub> &rarr; F<sub>4</sub> &rarr; F<sub>5</sub>.</p>
</div>

Ranking gives $$F_4 \to F_1 \to F_5$$ &mdash; a different order than the query was written in ($$F_1 \to F_4 \to F_5$$), because $$F_4$$ removes the most documents per token spent even though it isn't listed first. The ask costs are within 12% of each other, since each is processing a similar amount of tokens, so selectivity drives the ranking. $$F_5$$ keeps $$61.2\%$$ of documents and removes little for its cost, so it goes last. Pricing other orders treats each $$s_i$$ as an independent marginal selectivity of the filter itself, not of the position it runs in.

$$F_4$$ also has the cheapest scan, so it is a natural candidate for the first position. [Table 6](#table-6) prices each candidate ordering with the equation from [Section 4.3](#43-choosing-the-filter-order). Starting with $$F_1$$, as the query is written, costs $$0.324$$~s (4.5%) more: $$F_1$$ has a more expensive scan (0.023 ms per document, 0.116 s over 5,000 documents), and it lets $$2{,}428$$ documents through to $$F_4$$, where the chosen order sends only $$1{,}137$$ documents to $$F_1$$. Starting with $$F_5$$ is the most expensive because $$F_5$$ removes the fewest documents before the other two filters run. The number of documents entering each stage under the chosen order is $$N_j = N\prod_{k<j} s_{\pi_k} = 5{,}000,\ 1{,}137,\ 552$$. These counts, unlike those of the run order, are estimates that rely on the independence assumption above.

<div class="table-wrap" id="table-6" markdown="1">

| First | Ordering $$\pi$$ | $$C(\pi)$$ (s) |
|-------|---------------|----------------|
| $$F_4$$ | $$F_4 \to F_1 \to F_5$$ | 6.8666 |
| $$F_1$$ | $$F_1 \to F_4 \to F_5$$ | 7.1906 |
| $$F_5$$ | $$F_5 \to F_4 \to F_1$$ | 7.2995 |

<p class="table-caption">Table 6. Step 2 of the ordering procedure: the cost of each candidate ordering.</p>
</div>

Overall, the cost equation estimates the latency of a fixed filter order, and the ordering rule selects the order with the lowest estimated latency.

## 5.2 Cost of the Ordered Conjunction

The capacity calculation in [Section 4.1](#41-kv-reuse-and-hbm-capacity) shows that only $$1{,}204$$ document prefixes fit in HBM at once, so the $$5{,}000$$ documents run as five batches.

[Table 7](#table-7) lists the per-stage quantities of [Section 4.2](#42-cost-of-a-fixed-filter-order) for the chosen order, together with their totals.

<div class="table-wrap" id="table-7" markdown="1">

| Stage | $$n_j$$ | $$A_j$$ | $$W_j$$ | $$R_j$$ |
|-------|-------|-------|-------|-------|
| 1 ($$F_4$$, prefix) | 1,729,233 | $$2.999\times10^{8}$$ | 1,729,233 | 0 |
| 2 ($$F_1$$) | 57,987 | $$1.895\times10^{7}$$ | 57,987 | 342,063 |
| 3 ($$F_5$$) | 27,048 | $$8.813\times10^{6}$$ | 27,048 | 166,067 |
| **Conjunction** | **1,814,268** | $$3.277\times10^{8}$$ | **1,814,268** | **508,130** |

<p class="table-caption">Table 7. Per-stage work of the ordered IMDB conjunction, and its totals.</p>
</div>

**Pricing the whole conjunction.** Quail prices the conjunction as one piece of work rather than pricing each filter and adding the latencies. It sums the quantities over the filters, $$n = \sum_j n_j$$, $$A = \sum_j A_j$$, $$W = \sum_j W_j$$, and $$R = \sum_j R_j$$, sets $$K = \lceil n/C \rceil$$, and applies the roofline equations of [Section 3](#3-cost-model-for-one-filter) once per component with these totals. FLOPs and bytes are additive over filters, so the compute times and the memory times each add exactly. The only nonlinear step is the max. For a component whose filters have compute times $$a_j$$ and memory times $$b_j$$,

$$
\max\!\Big(\sum_j a_j,\ \sum_j b_j\Big) \ \leq \ \sum_j \max(a_j, b_j),
$$

because each $$\max(a_j, b_j)$$ is at least $$a_j$$ and at least $$b_j$$, with equality when every filter has its component bound by the same resource. Summing first therefore never gives a larger latency than adding per-filter latencies, and the two differ only when the filters have different bottlenecks for a component.

The conjunction is processed in $$K = \lceil n/C \rceil = \lceil 1{,}814{,}268/110{,}376 \rceil = 17$$ forward passes. We apply the roofline to the totals of [Table 7](#table-7), with $$B_{\text{attn}} = B_{\text{kv}}(W+R)$$ and $$W + R = 2{,}322{,}398$$ tokens, once for each component.

Only attention differs from adding per-filter latencies. [Table 9](#table-9) shows why. Stage 1 is compute-bound. Stages 2 and 3 would be memory-bound if priced alone, so per-stage pricing charges their memory times of 17.61 and 8.50 ms, giving $$178.76 + 17.61 + 8.50 = 204.87$$ ms. But the total memory time of the conjunction is only 102.22 ms, less than the 195.31 ms of total compute, so the memory traffic of stages 2 and 3 is hidden behind the compute of the whole conjunction. Attention therefore costs $$\max(195.31, 102.22) = 195.31$$ ms, which is $$9.56$$ ms less than adding per-filter costs. Projections and the MLP are compute-bound in every stage, so both ways of pricing them agree.

<div class="table-wrap" id="table-8" markdown="1">

| Component | FLOPs | Bytes | $$T_{\text{compute}}$$ | $$T_{\text{memory}}$$ |
|-----------|-------|-------|-----------------------|----------------------|
| Projections | $$3.42\times10^{15}$$ | $$1.60\times10^{10}$$ | 1.7303 s | 0.0048 s |
| Attention | $$1.93\times10^{14}$$ | $$3.42\times10^{11}$$ | 0.1953 s | 0.1022 s |
| MLP | $$9.76\times10^{15}$$ | $$4.57\times10^{10}$$ | 4.9314 s | 0.0136 s |
| **Total** | $$1.34\times10^{16}$$ | $$4.04\times10^{11}$$ | **6.857 s** | 0.12 s |

<p class="table-caption">Table 8. Cost of the ordered IMDB conjunction, with the roofline applied once per component. Every component is compute-bound.</p>
</div>

<div class="table-wrap" id="table-9" markdown="1">

| Stage | $$a_j$$ (ms) | $$b_j$$ (ms) | $$\max(a_j,b_j)$$ (ms) |
|-------|------------|------------|------------------------|
| 1 ($$F_4$$) | 178.76 | 76.12 | 178.76 |
| 2 ($$F_1$$) | 11.30 | 17.61 | 17.61 |
| 3 ($$F_5$$) | 5.25 | 8.50 | 8.50 |
| **Total** | **195.31** | **102.22** | **204.87** |

<p class="table-caption">Table 9. Attention compute time <em>a<sub>j</sub></em> and memory time <em>b<sub>j</sub></em> per stage.</p>
</div>

## 5.3 Speed-of-Light Estimate

[Table 8](#table-8) provides the values used to compute the SoL estimate for the conjunction of filters. At peak arithmetic throughput and peak memory bandwidth on an H100, with ideal scheduling and no avoidable KV recomputation,

$$
\begin{aligned}
\text{SoL} &= T_{\text{proj}} + T_{\text{attn}} + T_{\text{mlp}} \\
    &= 1.7303 + 0.1953 + 4.9314 \\
    &\approx \boxed{6.86~\text{seconds}} .
\end{aligned}
$$

The ordering cost $$C(\pi) = 6.8666$$~s of the chosen order in [Table 6](#table-6) is slightly higher than this because it adds per-filter costs; we use $$C(\pi)$$ only to compare orders, not as the final latency estimate.

For an implementation of the example query on Qwen3-4B and an H100, the observed runtime can now be compared with $$6.86$$~s. A large gap points to inefficiency such as lost KV reuse or idle tensor cores. Roofline estimates are also most optimistic when fixed overheads such as scheduling dominate, as at small batch sizes.<sup><a href="#note-9">9</a></sup><sup>,</sup><sup><a href="#note-10">10</a></sup>

# 6. Conjunction of Filters Playground

We built an interactive playground that lets a reader build a conjunction of filters and see its speed-of-light latency directly. In the playground, filters are dragged to reorder them, and each filter's selectivity $$s_i$$ and instruction length $$q_{\text{tail}}$$ are tunable. The filters share the prefix $$q_{\text{pre}}$$ because they reuse one prefix KV, as described in [Section 4.1](#41-kv-reuse-and-hbm-capacity). The playground applies the ordering rule from [Section 4.3](#43-choosing-the-filter-order) and, for up to six filters, compares the result with every possible order. The playground uses Qwen3-4B on an H100, and its default values come from the IMDB query in [Section 5](#5-example-imdb-query). All numbers are lower bounds at peak rates, with expected survivors rounded to whole documents and every document at the mean length.

<div class="pg" id="filter-chain-playground" data-filter-chain-calculator>
  <noscript>The playground needs JavaScript. The worked example in Section 5 gives the same numbers for the IMDB query.</noscript>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Sortable/1.15.6/Sortable.min.js" defer></script>
<script src="{{ '/assets/js/filter-chain-calculator.js' | relative_url }}" defer></script>

The playground is available online at [https://fsdatalab.github.io/blog/ai-filter-cost-estimates/#6-conjunction-of-filters-playground](https://fsdatalab.github.io/blog/ai-filter-cost-estimates/#6-conjunction-of-filters-playground).

# 7. Conclusion

The cost model estimates latency for a given workload, model, and GPU. The resulting SoL estimate is an optimistic lower bound on latency. We applied our cost model to one filter and a conjunction of filters from an IMDB query. Comparing the observed runtime with the SoL estimate shows how much room remains for improvements such as preserving KV reuse and keeping the tensor cores busy.<sup><a href="#note-9">9</a></sup>

# Acknowledgements

We thank [Modal](https://modal.com/) for sponsoring the compute used in
this research.

# Notes

<span id="note-2b"><strong>†.</strong></span> We use Qwen3-4B FP8 because its grouped query attention reduces KV-cache storage and its FP8 weights use the H100's higher FP8 throughput. Its 32 query heads share eight key-value heads, making the KV cache one quarter the size of full multi-head attention. FP8 also halves raw weight storage relative to BF16, while the KV cache remains in BF16. See [NVIDIA's FP8 primer](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.5/user-guide/examples/fp8_primer.html).

<span id="note-6b"><strong>‡.</strong></span> NVIDIA's matrix multiplication guide counts each fused multiply&ndash;add as two operations, so a product of $$M\times K$$ and $$K \times N$$ matrices takes $$2MKN$$ FLOPs. See [NVIDIA's Matrix Multiplication Background](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#math-mem).

<span id="note-9b"><strong>§.</strong></span> The constant 32 is an estimate chosen in Quail's implementation, not a measured value.

<span id="note-9c"><strong>¶.</strong></span> FP8 weights are quantized in small blocks, each with its own scale factor stored at higher precision so the block can be rescaled correctly when read back. Those scale factors are extra bytes beyond the raw FP8 weights, and they stay resident in HBM for as long as the weights do.

# References

<span id="note-1"><strong>1.</strong></span> Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebron, and Sumit Sanghai. 2023. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, 4895&ndash;4901.

<span id="note-2"><strong>2.</strong></span> Surajit Chaudhuri and Kyuseok Shim. 1999. Optimization of queries with user-defined predicates. *ACM Transactions on Database Systems* 24, 2, 177&ndash;228. [doi:10.1145/320248.320249](https://doi.org/10.1145/320248.320249)

<span id="note-3"><strong>3.</strong></span> Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher R&eacute;. 2022. FlashAttention: fast and memory-efficient exact attention with IO-awareness. In *Advances in Neural Information Processing Systems 35 (NeurIPS 2022)*.

<span id="note-4"><strong>4.</strong></span> Fergus Finn. 2026. The economics of speculative decoding. Blog post. [https://fergusfinn.com/blog/economics-of-speculative-decoding/](https://fergusfinn.com/blog/economics-of-speculative-decoding/)

<span id="note-5"><strong>5.</strong></span> Joseph M. Hellerstein and Michael Stonebraker. 1993. Predicate Migration: Optimizing Queries with Expensive Predicates. In *Proceedings of the 1993 ACM SIGMOD International Conference on Management of Data*, 267&ndash;276. [doi:10.1145/170035.170078](https://doi.org/10.1145/170035.170078)

<span id="note-6"><strong>6.</strong></span> Kipply. 2022. Transformer Inference Arithmetic. [https://kipp.ly/transformer-inference-arithmetic/](https://kipp.ly/transformer-inference-arithmetic/)

<span id="note-7"><strong>7.</strong></span> Pawe&#322; Liskowski, Benjamin Han, Paritosh Aggarwal, Bowei Chen, Boxin Jiang, Nitish Jindal, Zihan Li, Aaron Lin, Kyle Schmaus, Jay Tayade, Anupam Datta, Nathan Wiegand, and Dimitrios Tsirogiannis. 2026. Cortex AISQL: A Production SQL Engine for Unstructured Data. In *Companion of the International Conference on Management of Data (SIGMOD Companion '26)*.

<span id="note-8"><strong>8.</strong></span> Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. 2011. Learning Word Vectors for Sentiment Analysis. In *Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies*, 142&ndash;150. [https://aclanthology.org/P11-1015/](https://aclanthology.org/P11-1015/)

<span id="note-9"><strong>9.</strong></span> Ben Mayer. 2026. HTDYM (How To Deploy Your Model). Sail Research blog. [https://www.sailresearch.com/blog/htdym](https://www.sailresearch.com/blog/htdym)

<span id="note-10"><strong>10.</strong></span> Modal. LLM Engineer's Almanac (Spec Dec Roofline Model / Speedup ratio). Web page. [https://modal.com/llm-almanac/spec-dec-roofline](https://modal.com/llm-almanac/spec-dec-roofline)

<span id="note-11"><strong>11.</strong></span> The H100 SXM memory capacity, bandwidth, and Tensor Core throughput come from [NVIDIA's H100 specifications](https://resources.nvidia.com/en-us-gpu-resources/h100-datasheet-24306). NVIDIA reports Tensor Core throughput with structured sparsity, while Table 1 uses the dense rates, which are half of the reported sparse rates. The L2 cache size, SM count, and number of Tensor Cores per SM come from [NVIDIA's Hopper architecture overview](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/).

<span id="note-12"><strong>12.</strong></span> Liana Patel, Siddharth Jha, Melissa Pan, Harshit Gupta, Parth Asawa, Carlos Guestrin, and Matei Zaharia. 2025. Semantic Operators and Their Optimizations in LOTUS. *Proc. VLDB Endow.* 18, 11, 4171&ndash;4184.

<span id="note-13"><strong>13.</strong></span> Shreya Shankar, Tristan Chambers, Tarak Shah, Aditya G. Parameswaran, and Eugene Wu. 2025. DocETL: Agentic Query Rewriting and Evaluation for Complex Document Processing. *Proc. VLDB Endow.* 18, 9, 3035&ndash;3048. [doi:10.14778/3746405.3746426](https://doi.org/10.14778/3746405.3746426)

<span id="note-14"><strong>14.</strong></span> Shreya Shankar, Charles Frye, Fergus Finn, Arnav Dhariya, Joseph Barrow, and Meryem Arik. 2026. Building an Ultra-High Throughput AI-SQL Engine. [https://fsdatalab.github.io/blog/introducing-quail/](https://fsdatalab.github.io/blog/introducing-quail/)

<span id="note-15"><strong>15.</strong></span> Noam Shazeer. 2020. GLU Variants Improve Transformer. [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)

<span id="note-16"><strong>16.</strong></span> Samuel Williams, Andrew Waterman, and David Patterson. 2009. Roofline: an insightful visual performance model for multicore architectures. *Commun. ACM* 52, 4, 65&ndash;76. [doi:10.1145/1498765.1498785](https://doi.org/10.1145/1498765.1498785)

<span id="note-17"><strong>17.</strong></span> An Yang et al. 2025. Qwen3 Technical Report. [arXiv:2505.09388](https://arxiv.org/abs/2505.09388)

# Cite this post

<div class="bibtex-block" markdown="1">
<button class="copy-bibtex" type="button">Copy BibTeX</button>

```bibtex
@misc{dhariya2026aifilter,
  title = {Estimating Costs for AI-Powered Filters},
  author = {Dhariya, Arnav A. and Shankar, Shreya},
  year = {2026},
  month = sep,
  url = {https://fsdatalab.github.io/blog/ai-filter-cost-estimates/}
}
```

</div>
