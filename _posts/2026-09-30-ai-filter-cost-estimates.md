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
      <li><a href="#36-cost-of-one-biodex-filter">Cost of One BioDEX Filter.</a></li>
    </ol>
  </li>
  <li><a href="#4-cost-model-for-a-conjunction-of-filters">Cost Model for a Conjunction of Filters.</a>
    <ol>
      <li><a href="#41-kv-reuse-and-hbm-capacity">KV Reuse and HBM Capacity.</a></li>
      <li><a href="#42-cost-of-a-fixed-filter-order">Cost of a Fixed Filter Order.</a></li>
      <li><a href="#43-choosing-the-filter-order">Choosing the Filter Order.</a></li>
    </ol>
  </li>
  <li><a href="#5-example-biodex-query">Example: BioDEX Query.</a>
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

We recently released [Quail](/blog/introducing-quail/), an execution engine for AI-SQL, which extends SQL with functions that call LLMs. To compare query plans, Quail needs to estimate the latency of each AI operation. The latency depends on the input length, the model, the GPU, and how much of the KV cache can be reused.

**How can we estimate the latency of an AI operation on a given LLM and GPU?**

In this post, we focus on AI-powered filters, which use an LLM to decide whether each document satisfies a natural language predicate.

One approach is to profile the system by running representative queries and fitting a latency model to the measurements. A profile has to be redone for every new model, GPU, or workload. Instead, we use the roofline model to estimate filter latency.<sup><a href="#note-1">1</a></sup> We count the arithmetic and HBM traffic required by a filter, then divide by the GPU's peak compute throughput and memory bandwidth. The roofline estimate is a lower bound because it assumes that the GPU sustains these peak rates. Quail uses this *speed-of-light* (SoL) latency to compare query plans.

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

**AI-SQL filter.** An AI-SQL filter is a SQL-style predicate, i.e a simple comparison like `WHERE price > 10`, where the condition is instead answered by an LLM, which reads each row (or document) and returns a true/false, one token, judgment on whether it satisfies the predicate. AI-powered filters can handle conditions that SQL can't express, although at the cost of running a model call per row instead of a cheap comparison.

<figure class="figure-full" id="figure-1">
  <img src="{{ '/assets/blog/ai-filter-cost-estimates/figure-1.svg' | relative_url }}" alt="A database instance of four BioDEX reports, an AI_IF query applying a conjunction of three predicates, and the resulting table of TRUE/FALSE outcomes per report.">
  <figcaption>Figure 1. Simplified BioDEX instance and query with results for 3 predicates.</figcaption>
</figure>

[Figure 1](#figure-1) displays the general prompt structure for the AI-powered filter query, its structure is similar to that of a SQL query. Within each invocation of the operator there exists a preamble, "Document: ", the actual document, and the filter instruction "question: [...]". The output of each predicate is a single token output of true or false. Only the surviving documents pass through to subsequent filters. [Figure 1](#figure-1) shows a query with a conjunction of filters. We use the first predicate as the single AI-powered filter example in [Section 3](#3-cost-model-for-one-filter).

The formatting of the preamble, document, and filter instruction together, influence the cost model as each predicate carries the same amount of tokens around the variable-length document.

## 2.2 GPU Execution

We run our examples on an NVIDIA H100 SXM GPU. Our cost model depends on three properties of the GPU: its peak arithmetic throughput, HBM bandwidth, and HBM capacity.

<figure class="figure-full" id="gpu">
  <img src="{{ '/assets/blog/ai-filter-cost-estimates/gpu.svg' | relative_url }}" alt="H100 memory hierarchy diagram showing HBM3, L2 cache, and an SM with registers, shared memory, and tensor cores.">
  <figcaption>Figure 2. H100 memory layout and forward-pass data movement.</figcaption>
</figure>

[Figure 2](#gpu) shows the parts of the H100 that affect our cost model. *High-bandwidth memory (HBM)* is the GPU's main memory. It stores the model's weight matrices, embedding table, and KV cache. Data read from HBM passes through the shared L2 cache before reaching one of the GPU's 132 *streaming multiprocessors (SMs)*. Within each SM, registers and the combined shared memory and L1 cache hold small tiles of data close to the Tensor Cores. Each SM has four Tensor Cores, which perform the matrix multiplications used by the model. The H100 SXM has 80 GB of HBM3 and a 50 MB L2 cache shared by all of its SMs.<sup><a href="#note-2">2</a></sup>

Our cost model does not model each level of the cache separately. It counts the bytes read from or written to HBM and divides that amount by the HBM bandwidth. It also counts the *floating-point operations (FLOPs)* performed by the Tensor Cores and divides that amount by their arithmetic throughput.

[Table 1](#table-1) lists the hardware constants used in our examples. *Arithmetic throughput* is the number of FLOPs that the GPU can perform per second, and *memory bandwidth* is the number of bytes that it can transfer from HBM per second. $$\Pi_{fp16}$$ and $$\Pi_{fp8}$$ are the peak arithmetic throughput rates for FP16 and FP8 operations, while $$\beta$$ is the peak HBM bandwidth. We use the FP8 rate for the projection and MLP matrix multiplications, and the FP16 rate for attention in our FlashAttention setup.

<div class="table-wrap" id="table-1" markdown="1">

| Symbol | Value |
|--------|-------|
| $$\Pi_{fp16}$$ | $$989.5\times10^{12}$$ FLOP/s |
| $$\Pi_{fp8}$$ | $$1.979\times10^{15}$$ FLOP/s |
| $$\beta$$ | $$3.35\times10^{12}$$ bytes/s |
| Capacity | 80 GB |

<p class="table-caption">Table 1. H100 SXM hardware constants.</p>
</div>

## 2.3 Transformer Forward Pass

The architecture of the LLM determines how many operations it performs and how much data it moves. We also need to account for the KV cache, which allows the model to reuse previously processed prefixes.

We use the Qwen3-4B FP8 model in our examples.<sup><a href="#note-3">3</a></sup>

<figure class="figure-full" id="model">
  <img src="{{ '/assets/blog/ai-filter-cost-estimates/model.svg' | relative_url }}" alt="Diagram of a token passing through one Qwen3-4B layer: QKV projection, attention with a KV cache, output projection, and the gate/up/SwiGLU/down MLP block.">
  <figcaption>Figure 3. Path of a single token through one Qwen3-4B layer. The QKV and output projections are per-token matrix multiplications; only the attention step reads the KV cache of all <em>T</em> tokens.</figcaption>
</figure>

[Figure 3](#model) follows one token through a layer of Qwen3-4B. Each layer contains four attention projection matrices and three MLP matrices.

The query, key, and value matrices project the token's 2,560-element hidden state into three vectors. The *query vector* contains the information used to search the current and earlier tokens. Each *key vector* contains the information used to match a token, and each *value vector* contains the information returned by that match. During attention, the model compares the query vector with the key vectors and uses the resulting scores to combine their value vectors. The output projection then maps the attention result back to the model's hidden width.

Qwen3-4B uses *grouped query attention (GQA)* {% include shreya-comment.html text="cite" %}. It has 32 query heads and eight key and value heads, so every four query heads share one key and value head. The key and value matrices are therefore one quarter of the size of the query and output matrices. The smaller key and value matrices also reduce the size of the KV cache.

The MLP processes each token independently through the gate, up, and down matrices. The gate and up matrices operate on the same input. The model combines their outputs with the SwiGLU activation and uses the down matrix to project the result back to the hidden width. The MLP cost per token does not grow with the number of earlier tokens, while the attention cost does. Qwen3-4B repeats the attention and MLP blocks across all 36 layers.

[Table 2](#table-2) lists the model constants used by our cost model. $$d_{model}$$ is the *hidden-state width*. $$n_{heads}$$ and $$n_{kv}$$ are the numbers of query heads and key-value heads, while $$d_{head}$$ is the width of each attention head. $$d_{mlp}$$ is the *MLP intermediate width*, and vocab is the *vocabulary size*.

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

The transformer forward pass determines how many FLOPs the model performs and how many bytes it moves from HBM. The roofline model converts these two quantities into a latency estimate.

A GPU spends time performing arithmetic and moving data from HBM. We call these *compute time* and *memory transfer time*. The roofline model assumes that the GPU can overlap computation with data transfer, so the slower of the two determines the latency:

$$
T = \max\!\left(\frac{\text{FLOPs}}{\Pi},\; \frac{\text{Bytes Moved}}{\beta}\right),
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

[Figure 4](#figure-4) plots attainable compute throughput against operational intensity. The sloped region contains memory-bound operations, whose throughput increases as they perform more FLOPs per byte. The horizontal region contains compute-bound operations, whose throughput cannot exceed the GPU's peak arithmetic rate. The point where the two regions meet is the ridge point.

<figure class="figure-full" id="figure-4">
  <img src="{{ '/assets/blog/ai-filter-cost-estimates/figure-4.svg' | relative_url }}" alt="Log-log roofline plot for an NVIDIA H100 SXM showing the memory-bound and compute-bound regions, the ridge point, and the operating point of the example AI filter query.">
  <figcaption>Figure 4. Roofline for an NVIDIA H100 SXM (&Pi; = 1.979&times;10<sup>15</sup> FLOP/s at FP8, &beta; = 3.35&times;10<sup>12</sup> bytes/s). The red marker shows the operational intensity of the query in Section 2.1 and its derivation is given in Section 3.6.</figcaption>
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
n_{\text{tok}} &= \sum_i r_i \\
    &= L_1 + N(q_{\text{pre}} + q_{\text{tail}}), \\
L_2' &= \sum_i r_i^2.
\end{aligned}
$$

In case that per-document lengths are unavailable, one can approximate $$r_i$$ by the document length mean. $$L_2'$$ is necessary for self-attention computations in [Section 3.3](#33-attention-cost). Weights are re-streamed from the HBM once per forward pass, and each pass is bounded by the chunk budget which is essential to compute the amount of passes that will occur. Below, we provide the formula to compute the chunk budget, $$C$$, and as a result, the number of forward passes, $$K$$,

$$
\begin{aligned}
C &= \left\lfloor \frac{2^{31}-1}{2\,d_{\text{mlp}}} \right\rfloor, \\
K &= \left\lceil \frac{n_{\text{tok}}}{C} \right\rceil .
\end{aligned}
$$

The chunk budget is bounded by the widest intermediate activation in the model, $$d_\text{mlp}$$, and the slot-indexing counter, $$2^{31} - 1$$. By dividing the total number of tokens in the documents by the chunk budget, we obtain the number of forward passes.

## 3.2 Projection Cost

Here we derive the FLOPs for Q, K, V and output projections, their HBM traffic across forward passes and compute the projection roofline cost.

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

Here $$P_{\text{proj}}$$ is the total projection parameters for all of the layers, $$F_{\text{proj}}$$ is the total FLOPs across the total tokens in $$N$$ documents. The 2 in $$F_{\text{proj}}$$ is one multiply plus one add, and $$b_w$$ is bytes per weight (1 for FP8). The value of $$K$$ represents the number of forward passes to process all $$n_{\text{tok}}$$ tokens. Thus, the total time is given by the $$T_{\text{proj}}$$ which takes the maximum of compute time and memory transfer time for projections.

## 3.3 Attention Cost

Here we derive the attention comparisons from the request lengths, derive the attention FLOPs, the writes and reads for the KV cache, and compute the attention roofline cost.

Attention is different from the projections as tokens here interact with each other. Each token attends to itself and to every earlier token in its own request. With an empty KV cache, a request of $$r_i$$ tokens makes $$1 + 2 + \dots +r_i -1 + r_i$$ comparisons, so across the batch

$$
A = \sum_i \frac{r_i(r_i+1)}{2} = \frac{L_1' + L_2'}{2},
$$

where $$L_1' = \sum_i r_i$$ and $$L_2' = \sum_i r_i^2$$.

We introduced $$L_2'$$ in the prior section because it is needed for the closed form of the self-attention computation.

One comparison, for one head in one layer, costs $$2\,d_{\text{head}}$$ FLOPs for the query--key dot product ($$QK^\top$$), because we do a multiply and addition per number and another $$2\,d_{\text{head}}$$ for the weighted sum over value vectors. Repeating across all $$n_{\text{heads}}$$ and all layers gives,

$$
F_{\text{attn}} = 4\,n_{\text{heads}}\,d_{\text{head}}\,\text{layers}\cdot A.
$$

With GQA, several query heads share one K/V head, however each query head must compute its own dot products. Each token writes one K row and one V row per layer, and tokens already in the cache must be read back. The KV size per token is

$$
B_{\text{kv}} = 2\,b_{kv}\,\text{layers}\,n_{kv}\,d_{\text{head}},
$$

where the $$2$$ counts K and V, and $$b_{kv}$$ is bytes per element (2 for FP16 as the model uses FlashAttention). Here we use K/V heads rather than query heads because the cache stores K and V heads once, which is why GQA shrinks the cache. With $$W$$ tokens written and $$R$$ cached tokens read,

$$
B_{\text{attn}} = B_{\text{kv}}\,(W + R).
$$

For one filter, $$W = n_{\text{tok}}$$ and $$R = 0$$, because there were no prior K/V vectors in the cache.

FlashAttention runs in FP16 so we must divide by its arithmetic rate.

$$
T_{\text{attn}} = \max\!\left(\frac{F_{\text{attn}}}{\Pi_{fp16}}, \frac{B_{\text{attn}}}{\beta}\right).
$$

## 3.4 MLP Cost

We still need to derive the FLOPs for the gate, up and down MLP projections, their HBM traffic, and compute the MLP roofline cost.

The MLP acts on each token independently, so unlike attention its cost per token does not depend on the other tokens in the request. It has three matrices per layer, and the parameter count of each is the product of its input and output widths. $$W_{\text{gate}}$$ and $$W_{\text{up}}$$ map $$d_{\text{model}}$$ to $$d_{\text{mlp}}$$, while $$W_{\text{down}}$$ maps $$d_{\text{mlp}}$$ back to $$d_{\text{model}}$$:

$$
\begin{aligned}
p_{\text{gate}} &= d_{\text{model}}\,d_{\text{mlp}}, \\
p_{\text{up}}   &= d_{\text{model}}\,d_{\text{mlp}}, \\
p_{\text{down}} &= d_{\text{mlp}}\,d_{\text{model}}.
\end{aligned}
$$

Gate and up read the same input; their outputs are combined by SwiGLU (Swish Gated Linear Unit) activation and projected back down. The total parameters in MLP are,

$$
P_{\text{mlp}} = \text{layers}\cdot(p_{\text{gate}} + p_{\text{up}} + p_{\text{down}}) = 3\,\text{layers}\,d_{\text{model}}\,d_{\text{mlp}}.
$$

The reasoning for MLP is the same as it is for the projections: every token is multiplied by every weight once, and the weights are streamed from HBM once per forward pass.

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

Each component is compute-bound or memory-bound depending on its operational intensity, as described in Section 2.4.

## 3.6 Cost of One BioDEX Filter

We now estimate the SoL latency of one filter from our example query. We use the H100's peak arithmetic throughput and HBM bandwidth, along with the following batch and prompt values:

$$
N = 200, \qquad
\bar{\ell} = 4{,}145.96 \ \text{tokens}, \qquad
q_{\text{pre}} = 2 \ \text{tokens}, \qquad
q_{\text{tail}} = 47 \ \text{tokens}.
$$

We approximate every document by the mean length, which gives

$$
L_1 = N\bar{\ell} = 829{,}192 \ \text{tokens}.
$$

**Request and batch sizes.** Each request contains the prefix, one document, and the filter instruction:

$$
\begin{aligned}
r &= q_{\text{pre}} + \bar{\ell} + q_{\text{tail}} \\
  &= 2 + 4{,}145.96 + 47 \\
    &= 4{,}194.96 \ \text{tokens}, \\
n_{\text{tok}} &= N r \\
    &= 200 \times 4{,}194.96 \\
    &= 838{,}992.
\end{aligned}
$$

**Attention comparisons.** The attention calculation also needs the sum of squared request lengths:

$$
\begin{aligned}
L_2' &= N r^2 \\
    &= 200 \times 4{,}194.96^2 \\
    &= 3{,}519{,}537{,}880.32, \\
A &= \frac{n_{\text{tok}} + L_2'}{2} \\
    &= 1{,}760{,}188{,}436.16 .
\end{aligned}
$$

**Forward passes.** The widest intermediate is the SwiGLU input, with $$2\,d_{\text{mlp}} = 19{,}456$$ slots per token. The slots are indexed by a signed 32-bit counter:

$$
\begin{aligned}
C &= \left\lfloor \frac{2^{31}-1}{19{,}456} \right\rfloor \\
    &= 110{,}376, \\
K &= \left\lceil \frac{838{,}992}{110{,}376} \right\rceil \\
    &= \lceil 7.60 \rceil = 8 .
\end{aligned}
$$

**Model parameters.** We substitute the Qwen3-4B dimensions from Table 2:

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
P_{\text{proj}} &= 36 \cdot 2 \cdot 2560 \cdot 128 \cdot (32+8) \\
                  &= 943{,}718{,}400, \\
P_{\text{mlp}}  &= 3 \cdot 36 \cdot 2560 \cdot 9728 \\
                  &= 2{,}689{,}597{,}440 .
\end{aligned}
$$

**Projections.**

$$
\begin{aligned}
F_{\text{proj}} &= 2 \times 943{,}718{,}400 \times 838{,}992 \\
                  &\approx 1.5835\times10^{15}, \\
B_{\text{proj}} &= 1 \times 943{,}718{,}400 \times 8 \\
                  &\approx 7.550\times10^{9}, \\
T_{\text{proj}} &= \max\!\left(
      \frac{1.5835\times10^{15}}{1.979\times10^{15}},\;
      \frac{7.550\times10^{9}}{3.35\times10^{12}}\right) \\
    &= \max(0.8002,\ 0.0023) \\
    &= 0.8002 \ \text{s}.
\end{aligned}
$$

**Attention.**

$$
\begin{aligned}
F_{\text{attn}} &= 4 \cdot 32 \cdot 128 \cdot 36 \times A \\
                  &= 589{,}824 \times 1{,}760{,}188{,}436.16 \\
                  &\approx 1.0382\times10^{15}, \\
B_{\text{kv}}   &= 2 \cdot 2 \cdot 36 \cdot 8 \cdot 128 \\
                  &= 147{,}456 \ \text{bytes/token}, \\
B_{\text{attn}} &= 147{,}456 \times (838{,}992 + 0) \\
                  &\approx 1.2371\times10^{11}, \\
T_{\text{attn}} &= \max\!\left(
      \frac{1.0382\times10^{15}}{989.5\times10^{12}},\;
      \frac{1.2371\times10^{11}}{3.35\times10^{12}}\right) \\
    &= \max(1.0492,\ 0.0369) \\
    &= 1.0492 \ \text{s}.
\end{aligned}
$$

**MLP.**

$$
\begin{aligned}
F_{\text{mlp}} &= 2 \times 2{,}689{,}597{,}440 \times 838{,}992 \\
                 &\approx 4.5131\times10^{15}, \\
B_{\text{mlp}} &= 1 \times 2{,}689{,}597{,}440 \times 8 \\
                 &\approx 2.1517\times10^{10}, \\
T_{\text{mlp}} &= \max\!\left(
      \frac{4.5131\times10^{15}}{1.979\times10^{15}},\;
      \frac{2.1517\times10^{10}}{3.35\times10^{12}}\right) \\
    &= \max(2.2805,\ 0.0064) \\
    &= 2.2805 \ \text{s}.
\end{aligned}
$$

**Total.**

$$
\begin{aligned}
T_{\text{filter}} &= T_{\text{proj}} + T_{\text{attn}} + T_{\text{mlp}} \\
    &= 0.8002 + 1.0492 + 2.2805 \\
    &= \boxed{4.13 \ \text{seconds}} .
\end{aligned}
$$

<div class="table-wrap" id="table-3" markdown="1">

| Component | FLOPs | Bytes | $$T_{\text{compute}}$$ | $$T_{\text{memory}}$$ |
|-----------|-------|-------|-----------------------|----------------------|
| Projections | $$1.58\times10^{15}$$ | $$7.55\times10^{9}$$ | 0.8002 s | 0.0023 s |
| Attention | $$1.04\times10^{15}$$ | $$1.24\times10^{11}$$ | 1.0492 s | 0.0369 s |
| MLP | $$4.51\times10^{15}$$ | $$2.15\times10^{10}$$ | 2.2805 s | 0.0064 s |
| **Total** | | | | **4.13 s** |

<p class="table-caption">Table 3. Cost of one BioDEX filter on Qwen3-4B and an H100. Every component is compute-bound.</p>
</div>

# 4. Cost Model for a Conjunction of Filters

Section 3 estimates the cost of one filter with an empty KV cache. In a conjunction of filters, each successive filter processes only the documents that passed the preceding filters. Successive filters also reuse the prefix KV computed by the first filter. We first describe KV reuse and the HBM capacity needed to support it, then derive the cost of a fixed filter order and explain how to choose the order with the lowest cost.

## 4.1 KV Reuse and HBM Capacity

Taking a closer look at [Figure 1](#figure-1), every filter prompt for a given document contains the same preamble and document. Only the filter instruction changes. The average shared prefix length is therefore

$$
p = q_{\text{pre}} + \bar{\ell},
$$

where $$\bar{\ell}$$ is the mean document length. The first filter processes the prefix and stores its KV. Each later filter that evaluates the document reads the stored KV and processes only its own instruction tokens. Our cost model therefore processes each document's prefix once and reuses its KV for any later filters that evaluate the document.

To reuse the prefixes without recomputing them, we choose a batch small enough to keep every prefix KV in HBM while the later filters run. The HBM left after storing the model weights and embedding table is

$$
M_{\text{KV}} = \text{Capacity} - b_w(P_{\text{proj}} + P_{\text{mlp}}) - E.
$$

Here, $$E$$ is the size of the embedding table in bytes. {% include shreya-comment.html text="Is this true?" %} An average prefix uses $$B_{\text{kv}}p$$ bytes, so the maximum batch size is

$$
N_b = \left\lfloor \frac{M_{\text{KV}}}{B_{\text{kv}}\,p} \right\rfloor .
$$

We choose $$N \leq N_b$$ so that no prefix KV is evicted and recomputed.

For our Qwen3-4B and H100 example, KV capacity limits a batch to about $$123$$ documents. We therefore run the $$200$$ documents as two batches. The chunk budget in Section 3 limits the number of tokens in one forward pass, while KV capacity limits the number of document prefixes that can remain cached between filters.

## 4.2 Cost of a Fixed Filter Order

Suppose a query applies $$m$$ filters to $$N$$ documents. Let $$\pi = (\pi_1,\ldots,\pi_m)$$ be an ordering, where $$\pi_j$$ is the filter in position $$j$$. *Selectivity*, $$s_i$$, is the fraction of documents that survive filter $$F_i$$, and $$q_i$$ is the number of instruction tokens for $$F_i$$. Each filter runs only on the documents that survived the filters before it, so the expected number of documents entering position $$j$$ is

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

The ordering rule follows Hellerstein and Stonebraker's work on ordering expensive predicates.<sup><a href="#note-4">4</a></sup> For filters after the first position, the rule ranks each filter by its ask cost per document removed:

$$
\text{rank}_i = \frac{\text{ask}_i}{1 - s_i}.
$$

Only the first filter pays a scan cost. For each filter $$f$$, we evaluate one ordering with $$f$$ first and the remaining filters sorted by ascending $$\text{rank}_i$$. We compute $$C(\pi)$$ for each ordering and choose the one with the lowest cost.

We compute the scan and ask costs with the equations from [Section 3](#3-cost-model-for-one-filter).

Overall, the cost equation estimates the latency of a fixed filter order, and the ordering rule selects the order with the lowest estimated latency.

# 5. Example: BioDEX Query

Our example runs the query over $$N = 200$$ documents using Qwen3-4B on an H100. The weights use FP8, while attention and KV use FP16. The cached prefix contains $$p = 2 + 4{,}145.96 = 4{,}147.96$$ tokens per document.

## 5.1 Filter Ordering

[Table 4](#table-4) lists the three filters. We measure each filter's selectivity as the fraction of documents for which the LLM returns True. Each ask and scan cost is the sum of the three components of [Section 3](#3-cost-model-for-one-filter):

$$
\text{cost} = T_{\text{proj}} + T_{\text{mlp}} + T_{\text{attn}},
$$

where projections and the MLP take $$2\,P_{\text{proj}}\,n_{\text{tok}}/\Pi_{fp8}$$ and $$2\,P_{\text{mlp}}\,n_{\text{tok}}/\Pi_{fp8}$$. The inputs are the constants used throughout: $$P_{\text{proj}} = 943{,}718{,}400$$ and $$P_{\text{mlp}} = 2{,}689{,}597{,}440$$ are the parameter counts of [Section 3](#3-cost-model-for-one-filter); $$\Pi_{fp8} = 1.979\times10^{15}$$, $$\Pi_{fp16} = 989.5\times10^{12}$$ and $$\beta = 3.35\times10^{12}$$ are the H100 peak rates of [Table 1](#table-1); $$589{,}824 = 4\,n_{\text{heads}}\,d_{\text{head}}\,\text{layers} = 4 \cdot 32 \cdot 128 \cdot 36$$ is the attention FLOPs per comparison; and $$B_{\text{kv}} = 147{,}456$$ bytes/token is the KV size per token. The cached prefix is $$p = q_{\text{pre}} + \bar{\ell} = 2 + 4{,}145.96 = 4{,}147.96$$ tokens.

For $$F_7$$ ($$q = 47$$, the tail length of its predicate), an ask processes $$n = 47$$ tokens and has projection compute

$$
T_{\text{proj}} = \frac{2 \times 943{,}718{,}400 \times 47}{1.979\times10^{15}} = 0.0448~\text{ms},
$$

MLP compute

$$
T_{\text{mlp}} = \frac{2 \times 2{,}689{,}597{,}440 \times 47}{1.979\times10^{15}} = 0.1278~\text{ms},
$$

and attention cost. The 47 new tokens attend to the $$p$$ cached tokens and to each other, so

$$
A = q\,p + \frac{q(q+1)}{2} = 47 \times 4{,}147.96 + \frac{47 \times 48}{2} = 196{,}082.12.
$$

The KV traffic reads the $$p$$ cached tokens and writes the $$q$$ new ones, so $$W + R = p + q = 4{,}194.96$$ tokens:

$$
\begin{aligned}
T_{\text{attn}}
    &= \max\!\left(
        \frac{589{,}824 \times 196{,}082.12}{989.5\times10^{12}},\;
        \frac{147{,}456 \times 4{,}194.96}{3.35\times10^{12}}\right) \\
    &= \max(0.1169,\ 0.1846)~\text{ms} = 0.1846~\text{ms},
\end{aligned}
$$

the larger of the attention compute and the KV traffic of the cached prefix plus new tokens, so

$$
\text{ask}_7 = T_{\text{proj}} + T_{\text{mlp}} + T_{\text{attn}} = 0.0448 + 0.1278 + 0.1846 = 0.357~\text{ms}.
$$

A scan processes the full request of $$n = r = p + q = 4{,}147.96 + 47 = 4{,}194.96$$ tokens and has projection compute

$$
T_{\text{proj}} = \frac{2 \times 943{,}718{,}400 \times 4{,}194.96}{1.979\times10^{15}} = 4.00~\text{ms},
$$

MLP compute

$$
T_{\text{mlp}} = \frac{2 \times 2{,}689{,}597{,}440 \times 4{,}194.96}{1.979\times10^{15}} = 11.40~\text{ms},
$$

and attention compute. With an empty cache, one request makes

$$
A = \frac{r(r+1)}{2} = \frac{4{,}194.96 \times 4{,}195.96}{2} = 8{,}800{,}942
$$

comparisons, so

$$
T_{\text{attn}} = \frac{589{,}824 \times 8{,}800{,}942}{989.5\times10^{12}} = 5.25~\text{ms},
$$

which exceeds its memory transfer traffic ($$W = r$$, $$R = 0$$) of

$$
\frac{147{,}456 \times 4{,}194.96}{3.35\times10^{12}} = 0.185~\text{ms},
$$

so

$$
\text{scan}_7 = 4.00 + 11.40 + 5.25 = 20.65~\text{ms}.
$$

The other filters follow the same way with their own $$q$$ ($$41$$ for $$F_8$$, $$49$$ for $$F_9$$).

<div class="table-wrap" id="table-4" markdown="1">

| Filter | Predicate | $$q_{\text{tail}}$$ | $$s_i$$ | $$\text{ask}_i$$ (ms) | $$\text{scan}_i$$ (ms) | $$1-s_i$$ | rank (ms) | Docs in |
|--------|-----------|--------|-------|------------|-------------|---------|-----------|---------|
| $$F_7$$ | female patient | 47 | 0.555 | 0.357 | 20.649 | 0.445 | 0.803 | 200 |
| $$F_8$$ | combination therapy | 41 | 0.658 | 0.335 | 20.612 | 0.342 | 0.979 | 111 |
| $$F_9$$ | serious adverse event | 49 | 0.863 | 0.365 | 20.662 | 0.137 | 2.662 | 73 |

<p class="table-caption">Table 4. Filters sorted by ascending rank ask<sub>i</sub>/(1&minus;s<sub>i</sub>), giving the order F<sub>7</sub> &rarr; F<sub>8</sub> &rarr; F<sub>9</sub> in step 1. "Docs in" is the number of documents entering each stage under the chosen order.</p>
</div>

Ranking gives $$F_7 \to F_8 \to F_9$$. The ask costs are within 10% of each other, since each is processing a similar amount of tokens, so selectivity drives the ranking. $$F_9$$ keeps $$86.3\%$$ of documents and removes little for its cost, so it goes last.

$$F_8$$ has the cheapest scan, so it is a candidate for the first position. [Table 5](#table-5) prices each candidate ordering with the equation from [Section 4.3](#43-choosing-the-filter-order). Starting with $$F_8$$ saves $$7.4$$~ms of scan cost but makes $$F_7$$ run later on more documents, for a net cost of $$2.4$$~ms, so $$F_7$$ stays first. The number of documents entering each stage is $$N_j = N\prod_{k<j} s_{\pi_k} = 200,\ 111,\ 73$$.

<div class="table-wrap" id="table-5" markdown="1">

| First | Ordering $$\pi$$ | $$C(\pi)$$ (s) |
|-------|---------------|----------------|
| $$F_7$$ | $$F_7 \to F_8 \to F_9$$ | 4.1937 |
| $$F_8$$ | $$F_8 \to F_7 \to F_9$$ | 4.1961 |
| $$F_9$$ | $$F_9 \to F_7 \to F_8$$ | 4.2261 |

<p class="table-caption">Table 5. Step 2 of the ordering procedure: the cost of each candidate ordering.</p>
</div>

## 5.2 Cost of the Ordered Conjunction

The capacity calculation in [Section 4.1](#41-kv-reuse-and-hbm-capacity) shows that only $$123$$ document prefixes fit in HBM at once, so the $$200$$ documents run as two batches.

<div class="table-wrap" id="table-6" markdown="1">

| Stage | $$n_j$$ | $$A_j$$ | $$R_j$$ | $$T_{\text{proj}}$$ | $$T_{\text{attn}}$$ | $$T_{\text{mlp}}$$ | Stage total |
|-------|-------|-------|-------|--------------------|--------------------|--------------------|-------------|
| 1 ($$F_7$$, prefix) | 838,992 | $$1.760\times10^{9}$$ | 0 | 0.8002 s | 1.0492 s (compute) | 2.2805 s | 4.1299 s |
| 2 ($$F_8$$) | 4,551 | $$1.897\times10^{7}$$ | 460,424 | 4.34 ms | 20.47 ms (memory) | 12.37 ms | 37.18 ms |
| 3 ($$F_9$$) | 3,577 | $$1.493\times10^{7}$$ | 302,801 | 3.41 ms | 13.49 ms (memory) | 9.72 ms | 26.62 ms |
| **Conjunction** | 847,120 | | 763,225 | **0.8079 s** | **1.0832 s** | **2.3026 s** | **4.1937 s** |

<p class="table-caption">Table 6. Cost of the ordered BioDEX conjunction.</p>
</div>

## 5.3 Speed-of-Light Estimate

[Table 6](#table-6) provides the values used to compute the SoL estimate for the conjunction of filters. At peak arithmetic throughput and peak memory bandwidth on an H100, with ideal scheduling and no avoidable KV recomputation,

$$
\begin{aligned}
\text{SoL} &= T_{\text{proj}} + T_{\text{attn}} + T_{\text{mlp}} \\
    &= 0.8079 + 1.0832 + 2.3026 \\
    &= \boxed{4.19~\text{seconds}} .
\end{aligned}
$$

For an implementation of the example query on Qwen3-4B and an H100, the observed runtime can now be compared with $$4.19$$~s. A large gap points to inefficiency such as lost KV reuse or idle tensor cores.

# 6. Conjunction of Filters Playground

Build a conjunction of filters below and see its speed-of-light latency. Drag the filters to reorder them, and tune each filter's selectivity $$s_i$$ and instruction length $$q_{\text{tail}}$$. The filters share the prefix $$q_{\text{pre}}$$ because they reuse one prefix KV, as described in [Section 4.1](#41-kv-reuse-and-hbm-capacity). The playground applies the ordering rule from [Section 4.3](#43-choosing-the-filter-order) and, for up to six filters, compares the result with every possible order. The playground uses Qwen3-4B on an H100. Its default values come from the BioDEX query in [Section 5](#5-example-biodex-query), entered in reverse order.

<div class="pg" id="filter-chain-playground" data-filter-chain-calculator>
  <noscript>The playground needs JavaScript. The worked example in Section 5 gives the same numbers for the BioDEX query.</noscript>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Sortable/1.15.6/Sortable.min.js" defer></script>
<script src="{{ '/assets/js/filter-chain-calculator.js' | relative_url }}" defer></script>

All numbers are lower bounds at peak rates, with expected survivors rounded to whole documents and every document at the mean length.

# 7. Conclusion

The cost model estimates latency for a given workload, model, and GPU. The resulting SoL estimate is an optimistic lower bound on latency. We applied our cost model to one filter and a conjunction of filters from a BioDEX query. Comparing the observed runtime with the SoL estimate shows how much room remains for improvements such as preserving KV reuse and keeping the tensor cores busy.

# Acknowledgements

We thank [Modal](https://modal.com/) for sponsoring the compute used in
this research.

# Notes

<span id="note-1"><strong>1.</strong></span> Roofline models are commonly used to analyze the computation and memory limits of LLM inference. See [LLM Inference Unveiled: Survey and Roofline Model Insights](https://arxiv.org/abs/2402.16363). We follow [NVIDIA](https://developer.nvidia.com/blog/unleashing-the-power-of-nvidia-ampere-architecture-with-nvidia-nsight-developer-tools/) in calling theoretical peak performance the "Speed of Light."

<span id="note-2"><strong>2.</strong></span> The H100 SXM memory capacity, bandwidth, and Tensor Core throughput come from [NVIDIA's H100 specifications](https://www.nvidia.com/en-us/data-center/h100/). NVIDIA reports Tensor Core throughput with structured sparsity, while Table 1 uses the dense rates, which are half of the reported sparse rates. The L2 cache size, SM count, and number of Tensor Cores per SM come from [NVIDIA's Hopper architecture overview](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/).

<span id="note-3"><strong>3.</strong></span> We use [Qwen3-4B FP8](https://huggingface.co/Qwen/Qwen3-4B) because its grouped query attention reduces KV-cache storage and its FP8 weights use the H100's higher FP8 throughput. Its 32 query heads share eight key-value heads, making the KV cache one quarter the size of full multi-head attention. FP8 also halves raw weight storage relative to FP16, while the KV cache remains in FP16. See [NVIDIA's FP8 primer](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.5/user-guide/examples/fp8_primer.html).

<span id="note-4"><strong>4.</strong></span> Joseph M. Hellerstein and Michael Stonebraker, ["Predicate Migration: Optimizing Queries with Expensive Predicates"](https://doi.org/10.1145/170035.170078), *Proceedings of the 1993 ACM SIGMOD International Conference on Management of Data*, 1993.

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
