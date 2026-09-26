---
layout: post
title: "Estimating Costs for AI-Powered Filters"
date: 2026-09-30
author: "Arnav Dhariya, Shreya Shankar"
permalink: /blog/ai-filter-cost-estimates/
math: true
description: "A roofline-based cost model for AI-powered SQL filters that estimates FLOPs, HBM traffic, KV-cache storage, and latency, and extends to chains of filters with an optimal ordering rule."
---

<aside class="tldr"><strong>TL;DR:</strong> Running AI-SQL queries efficiently on a given LLM and GPU is crucial, but can we know the lower bound on latency? We present a cost model for a single filter and for a chain of filters, and implement it in <a href="https://github.com/fsdatalab/quail">Quail</a>. Combined with peak memory-transfer and peak arithmetic-throughput rates, the model yields Speed-of-Light (SoL) latency estimates (<a href="#3-cost-model-for-one-filter">Sec. 3</a> and <a href="#5-example-biodex-query">Sec. 5</a>). We also provide an interactive visualization for building AI filter chains and computing their optimal ordering, demonstrated on Qwen3-4B running on an H100 GPU.</aside>

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
      <li><a href="#31-cost-model">Cost Model.</a></li>
      <li><a href="#32-projection-cost">Projection Cost.</a></li>
      <li><a href="#33-attention-cost">Attention Cost.</a></li>
      <li><a href="#34-mlp-cost">MLP Cost.</a></li>
      <li><a href="#35-total-cost">Total Cost.</a></li>
      <li><a href="#36-example-single-filter-query-sol">Example: Single Filter Query SoL.</a></li>
    </ol>
  </li>
  <li><a href="#4-cost-model-for-a-conjunction-of-filters">Cost Model for a Conjunction of Filters.</a>
    <ol>
      <li><a href="#41-execution-model-and-kv-reuse">Execution Model and KV Reuse.</a></li>
      <li><a href="#42-cost-of-a-fixed-filter-order">Cost of a Fixed Filter Order.</a></li>
      <li><a href="#43-filter-ordering">Filter Ordering.</a></li>
    </ol>
  </li>
  <li><a href="#5-example-biodex-query">Example: BioDEX Query.</a>
    <ol>
      <li><a href="#51-filter-ordering">Filter Ordering.</a></li>
      <li><a href="#52-cost-of-the-ordered-conjunction">Cost of the Ordered Conjunction.</a></li>
      <li><a href="#53-speed-of-light-estimate">Speed-of-Light Estimate.</a></li>
    </ol>
  </li>
  <li><a href="#6-filter-chain-playground">Filter Chain Playground.</a></li>
  <li><a href="#7-conclusion">Conclusion.</a></li>
</ol>
</nav>

# 1. Introduction

Recent work has motivated the use of LLMs to analyze documents, where a query can filter rows using an arbitrary natural language predicate rather than a simple comparison. Consider the BioDEX dataset, a corpus of biomedical papers, each annotated with the adverse drug reactions mentioned within the paper. An analyst may want papers that report female patients. AI SQL can be used for such an analysis task, `AI_IF("The paper reports a female patient", fulltext)`, however, it raises a practical question: *how fast can such a query run?* Developers require a cost model to understand the theoretical latency of a similar query because without one they cannot determine whether hardware or poor configurations bound the query latency. We use an NVIDIA H100 SXM GPU and a Qwen3-4B LLM for our cost model. Our cost model estimates four quantities for a query on a given LLM and GPU: computations (FLOPs), HBM (High Bandwidth Memory) traffic, KV-cache storage, and latency.

The evaluation of our cost model with peak hardware performance and ideal execution yields the *Speed of Light* (SoL) estimate: an optimistic lower bound that an implementation on a particular GPU cannot beat. The SoL need not be achievable as its value lies in showing room for improvement. If there is room for improvement, it informs whether data movement or computation can be improved.

We outline a cost model to compute SoL estimates specifically for AI-powered filter queries, extend it to filter chains where we have multiple predicates in a sequence which we need to optimally order to avoid latency overestimation. [Section 2](#2-background) provides a background on AI-powered filters, GPU, and LLM, [Section 3](#3-cost-model-for-one-filter) introduces the cost model for a single AI-powered filter query, [Section 4](#4-cost-model-for-a-conjunction-of-filters) extends the cost model to a conjunction of AI-powered filter query, [Section 5](#5-example-biodex-query) demonstrates the conjunction query on a given example query, and finally [Section 6](#6-filter-chain-playground) lets you build your own filter chain in an interactive playground and compute its SoL.

# 2. Background

We now define the AI-powered filter and the general hardware. We specifically focus on the hardware and model specific to our examples and demonstrate a forward pass on them. Lastly, we describe the roofline model for SoL estimates.

## 2.1 AI-powered filters

Here we define AI-powered filters, our example's structure for both cost models, and describe the cost models.

**AI_SQL Filter.** An AI_SQL filter is a SQL-style predicate, i.e a simple comparison like `WHERE price > 10`, where the condition is instead answered by an LLM, which reads each row (or document) and returns a true/false, one token, judgment on whether it satisfies the predicate. AI-powered filters can handle conditions that SQL can't express, although at the cost of running a model call per row instead of a cheap comparison.

<figure class="figure-full" id="figure-1">
  <img src="{{ '/assets/blog/ai-filter-cost-estimates/figure-1.svg' | relative_url }}" alt="A database instance of four BioDEX reports, an AI_IF chain query applying three predicates, and the resulting table of TRUE/FALSE/— outcomes per report.">
  <figcaption>Figure 1. Simplified BioDEX instance and query with results for 3 predicates.</figcaption>
</figure>

[Figure 1](#figure-1) displays the general prompt structure for the AI-powered filter query, its structure is similar to that of a SQL query. Within each invocation of the operator there exists a preamble, "Document: ", the actual document, and the filter instruction "question: [...]". The output of each predicate is a single token output of true or false. Only the surviving documents pass through to subsequent filters. We display a chain query in [Figure 1](#figure-1). The first predicate within the chained query example is used as a single AI-powered filter query example in [Section 3](#3-cost-model-for-one-filter).

The formatting of the preamble, document, and filter instruction together, influence the cost model as each predicate carries the same amount of tokens around the variable-length document.

## 2.2 GPU Execution

We introduce the hardware associated with the cost model, its relevance, and the configuration we will later use for SoLs in the examples.

<figure class="figure-full" id="gpu">
  <img src="{{ '/assets/blog/ai-filter-cost-estimates/gpu.svg' | relative_url }}" alt="H100 memory hierarchy diagram showing HBM3, L2 cache, and an SM with registers, shared memory, and tensor cores.">
  <figcaption>Figure 2. H100 memory layout and forward-pass data movement.</figcaption>
</figure>

[Figure 2](#gpu) displays an abstraction of the parts in a GPU, specifically the NVIDIA H100. The HBM is the GPU's main memory where model weight matrices and the embedding table reside. It's huge and slower in comparison to the Streaming Multiprocessor's (SM) shared memory (L1) and L2 cache. The L2 cache sits between the HBM and SMs. It caches the data which is reused, thereby reducing the back and forth from the HBM. An SM has registers which are the fastest possible storage that hold values throughout computation and L1 memory where smaller tiles are staged for tensor core accesses. Each SM has 4 tensor cores that actually perform the matrix multiplications (matmuls). Tensor cores are where floating point operations (FLOPs) happen and everything upstream is tasked with feeding the data fast enough into them. In sizes, the HBM for an H100 is the largest at 80 GB, and the L2 cache follows with 50 MB shared for all 132 of the SMs. Each SM has a register and L1 which are up to 256 KB in size.

The GPU constants that matter for our cost model are outlined in [Table 1](#table-1). $$A_{f\_fp16}$$ and $$A_{f\_fp8}$$ are the peak arithmetic throughput rates for the two quantizations within the GPU, i.e. how many FLOPs per second the GPU can perform. The differences in quantization throughput rates is attributed to precision, as $$A_{f\_fp16}$$ is double the precision, 16-bit, as $$A_{f\_fp8}$$, which is 8-bit. We need both quantizations because we model attention in FP16, as in FlashAttention. However, General Matrix Multiplications (GEMMs) run in FP8.

<div class="table-wrap" id="table-1" markdown="1">

| Symbol | Value |
|--------|-------|
| $$A_{f\_fp16}$$ | $$989.5\times10^{12}$$ FLOP/s |
| $$A_{f\_fp8}$$ | $$1.979\times10^{15}$$ FLOP/s |
| $$A_{bm}$$ | $$3.35\times10^{12}$$ bytes/s |
| Capacity | 80 GB |

<p class="table-caption">Table 1. H100 SXM hardware constants.</p>
</div>

## 2.3 Transformer Forward Pass

Here, we explain how an LLM processes input tokens, and where it is stored within the GPU. We also introduce KV-caches to reuse previously processed prefixes.

Each model has a different architecture, for our purposes we pick the Qwen3-4B FP8 model.

<figure class="figure-full" id="model">
  <img src="{{ '/assets/blog/ai-filter-cost-estimates/model.svg' | relative_url }}" alt="Diagram of a token passing through one Qwen3-4B layer: QKV projection, attention with a KV cache, output projection, and the gate/up/SwiGLU/down MLP block.">
  <figcaption>Figure 3. Path of a single token through one Qwen3-4B layer. The QKV and output projections are per-token matrix multiplications; only the attention step reads the KV cache of all <em>T</em> tokens.</figcaption>
</figure>

[Figure 3](#model) helps visualize the projection and MLP weight matrices and their dimensions for the Qwen3-4B model.

We have 7 matrices in total. A token's hidden size is projected through matrices, here that is 2560. The query weight matrix projects the token's hidden state into a query vector that describes what a token is looking for in other tokens, the key weight matrix does the same and projects it to a key vector that represents what a token offers for other tokens being matched, and the value matrix projects on the actual content that would be retrieved when a match is found. The query matrix dimensions are different than the key and value matrix dimensions because of grouped query attention (GQA) where 4 query heads share one K/V head instead of having their own, so the KV matrices are a quarter of the size of the query and output matrices.

The attention matrices help understand information across tokens and what information is relevant. The MLP on the other hand is concerned with the information within a single token, and not cross token interactions. The gate, up, and down matrices have a set cost to process a token which does not increase due to the past tokens like in attention. The gate and up run in parallel on the same input producing two vectors that are combined using the SwiGLU activation and then projected back using the down matrix. Both, attention and MLP blocks, repeat identically for all 36 layers passing the output of one layer to the next, representing one forward pass.

[Table 2](#table-2) represents the constants for the Qwen3-4B model. Here $$d_{model}$$ is the width of each token's hidden-state vector as it flows through the stream; $$n_{heads}$$, $$n_{kv}$$, and $$d_{head}$$ describe how attention splits that vector that goes through the weight matrices for query, key, and value. $$d_{mlp}$$ is the intermediate width of the MLP, the largest that we expand to internally, and vocab is the number of distinct tokens the model can embed and predict over.

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

The KV cache allows for state reuse from a previously processed prefixes. It is crucial to our cost model and especially for conjunction queries.

## 2.4 Roofline Model & Speed of Light

We define compute and memory transfer time, the roofline equation, and discuss boundedness. We also define how a cost model produces a SoL estimate.

A GPU has two main responsibilities: (1) to perform operations, and (2) to transfer the data necessary from HBM for those operations, referred to as compute and memory transfer time. Both compute and memory transfer are carried out in parallel, but differ in completion times. Hence, we are either bounded by the memory transfer or compute. Our cost model can be characterized by the equation,

$$
T = \max\!\left(\frac{\text{FLOPs}}{\Pi},\; \frac{\text{Bytes Moved}}{\beta}\right),
$$

where $$\Pi$$ is the arithmetic throughput in FLOP/s, that is, how many operations we can do per second, and $$\beta$$ is the memory bandwidth that captures how fast the HBM delivers data. When both $$\Pi$$ and $$\beta$$ are the peak arithmetic throughput and peak memory bandwidth rates, we achieve the SoL estimate. In our example evaluations of this cost model we use the peak rates.

The ridge point, $$I^*$$ is the operational intensity at which a workload transitions from being memory-bound to being compute-bound:

$$
\mathrm{Ridge} = I^{*} = \frac{\Pi}{\beta}.
$$

By substituting peak arithmetic throughput and peak memory bandwidth rates, we achieve the ridge point,

$$
I^{*} = \frac{A_{f\_fp8}}{A_{bm}} = \frac{1.979\times10^{15}}{3.35\times10^{12}} \approx 590.746\ \text{FLOP/byte}.
$$

For every byte pulled out of the HBM we can theoretically do 590.75 FLOPs before the tensor cores would be idle, and wait for the next byte. If the operational intensity, $$I$$ is less than 590.75 we do fewer FLOPs per byte than the H100 GPU can sustain, meaning that the data transfer is the bottleneck, while if it is greater than 590.75 FLOPs, the H100 has enough data in queue to not be idle.

In associated literature the ridge point is represented by rooflines that we illustrate in [Figure 4](#figure-4): on the left-side we are memory-bound, while on the right-side we are compute-bound. The compute-bound is a roofline because we can never exceed the attainable throughput, even if our operational intensity increases.

<figure class="figure-full" id="figure-4">
  <img src="{{ '/assets/blog/ai-filter-cost-estimates/figure-4.svg' | relative_url }}" alt="Log-log roofline plot for an NVIDIA H100 SXM showing the memory-bound and compute-bound regions, the ridge point, and the operating point of the example AI filter query.">
  <figcaption>Figure 4. Roofline for an NVIDIA H100 SXM (&Pi; = 1.979&times;10<sup>15</sup> FLOP/s at FP8, &beta; = 3.35&times;10<sup>12</sup> bytes/s). The red marker shows the operational intensity of the query in Section 2.1 and its derivation is given in Section 3.6.</figcaption>
</figure>

# 3. Cost Model for One Filter

We provide an overview of the cost model and derive the associated costs for projection, attention and MLP. All three of them together help build the total filter costs. We demonstrate the filter costs through the single AI-powered filter query for the BioDEX dataset.

## 3.1 Cost Model

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

In case that per-document lengths are unavailable, one can approximate $$r_i$$ by the document length mean. $$L_2$$ is necessary for self-attention computations in [Section 3.3](#33-attention-cost). Weights are re-streamed from the HBM once per forward pass, and each pass is bounded by the chunk budget which is essential to compute the amount of passes that will occur. Below, we provide the formula to compute the chunk budget, $$C$$, and as a result, the number of forward passes, $$K$$,

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

We introduced $$L_2$$ in the prior section as it is crucial for the closed form of self-attention computations.

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

A forward pass runs projections, attention, and the MLP one after another, so their times add, and each component takes its own maximum of compute and memory time:

$$
T_{\text{filter}} = T_{\text{proj}} + T_{\text{attn}} + T_{\text{mlp}}.
$$

The particular bound, memory or compute, depends on operational intensity. For projections and the MLP, $$F/B = 2\,n_{\text{tok}}P/(b_w P K)$$, which is the number of tokens per forward pass. If $$F/B$$ exceeds the ridge point, $$I^*$$, we will be compute bound, otherwise we will be memory transfer bound. Similarly for attention, $$F/B$$, being either below or above the, $$I^*$$, threshold tells us whether what we will be bound by. $$I^*$$ can also help us compute the $$n_{\text{tok}}$$ at which we transition from memory transfer boundedness to compute boundedness.

## 3.6 Example: Single Filter Query SoL

Below we substitute the BioDEX, H100, and Qwen3-4B values and report the optimistic latency lower bound for a single AI-powered filter query. We use the first predicate from [Figure 1](#figure-1) which has $$q_{\text{pre}}=2$$ and $$q_{\text{tail}}=47$$. For the arithmetic throughput and memory transfer rate we use peak rates in the above cost model, which in return makes our computation a SoL estimate.

We batch $$N = 200$$ documents averaging $$\bar{\ell} = 4{,}145.96$$ tokens, so $$L_1 = 829{,}192$$. We approximate every document by the average length. Each request is then

$$
\begin{aligned}
r &= 4{,}145.96 + 2 + 47 \\
    &= 4{,}194.96 \ \text{tokens}, \\
n_{\text{tok}} &= L_1 + N(q_{\text{pre}} + q_{\text{tail}}) \\
    &= 829{,}192 + 200 \times 49 \\
    &= 838{,}992, \\
L_2' &= N r^2 \\
    &= 200 \times 4{,}194.96^2 \\
    &= 3{,}519{,}537{,}880.32, \\
A &= \frac{n_{\text{tok}} + L_2'}{2} \\
    &= 1{,}760{,}188{,}436.16 .
\end{aligned}
$$

The widest intermediate is the SwiGLU input, $$2\,d_{\text{mlp}} = 19{,}456$$ slots per token, and slots are indexed by a signed 32-bit counter:

$$
\begin{aligned}
C &= \left\lfloor \frac{2^{31}-1}{19{,}456} \right\rfloor \\
    &= 110{,}376, \\
K &= \left\lceil \frac{838{,}992}{110{,}376} \right\rceil \\
    &= \lceil 7.60 \rceil = 8 .
\end{aligned}
$$

With $$d_{\text{model}}=2560$$, $$n_{\text{heads}}=32$$, $$n_{kv}=8$$, $$d_{\text{head}}=128$$, $$d_{\text{mlp}}=9728$$ and 36 layers:

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

Here we explain the execution model for a conjunction of filters, give the cost of a fixed filter order, and present the rule for choosing the order that minimizes it.

## 4.1 Execution Model and KV Reuse

Taking a closer look at [Figure 1](#figure-1), the filters in a chain all read the same document, so all the requests share the prefix, $$q_{\text{pre}} + \text{document}$$, and only the filter instruction differs. The first filter pays for computing and storing the prefix KV, in addition to processing its own filter instruction. All of the subsequent filters operate on the first filter's selected set, pay for fetching the documents' KV from the HBM, and process only their filter instructions. Their instruction tokens attend to the cached prefix.

Documents can be pipelined through the filters in batches, which lets their KV remain in the cache until all filters have executed. The KV cache is limited by HBM capacity and model weights, so a batch must fit within the remaining space, otherwise evictions may happen resulting in prefix recomputations, thereby increasing cost. We compute the KV footprint and the batch size permitted by the HBM capacity below.

The KV cost per token, $$B_{\text{kv}}$$, was derived as a part of the attention cost in the prior section. For a document with an average prefix of $$p = q_{\text{pre}} + \bar{\ell}$$ tokens, where $$\bar{\ell}$$ is the mean document length, the KV footprint is $$B_{\text{kv}}\,p$$ bytes. HBM holds the weights and the embedding table, and the remainder is available for KV:

$$
\begin{aligned}
M_{\text{KV}} &= \text{Capacity} \\
                &\quad - b_w(P_{\text{proj}} + P_{\text{mlp}}) - E, \\
N_b &= \left\lfloor \frac{M_{\text{KV}}}{B_{\text{kv}}\,p} \right\rfloor .
\end{aligned}
$$

$$N_b$$ is the number of documents whose prefix KV fits at once. Pipelining in batches of $$N_b$$ never evicts KV, and therefore our cost model assumes an "infinite" KV cache. Keep in mind that we use mean document lengths so in practice the above KV cache size is an estimate.

## 4.2 Cost of a Fixed Filter Order

Any ordering of the filters, for example, any permutation of the already present order in [Figure 1](#figure-1), is valid for a query and will return the same result, because each filter judges a document independently of the others.

Suppose a query applies $$m$$ filters to $$N$$ documents. Let $$\pi = (\pi_1,\ldots,\pi_m)$$ be an ordering, where $$\pi_j$$ is the filter in position $$j$$. Selectivity $$s_i$$ is the fraction of documents that survive filter $$F_i$$. Each filter runs only on the documents that survived the filters before it, so the expected number of documents entering position $$j$$ is

$$
N_j = N \prod_{k<j} s_{\pi_{k}}.
$$

Each filter depends on four quantities: the tokens processed $$n$$, the attention comparisons $$A$$, the KV tokens written $$W$$, and the KV tokens read $$R$$.

The first filter processes each request as in the single-filter case, with $$r_i = q_{\text{pre}} + \ell_i + q_{\pi_1}$$:

$$
\begin{aligned}
n_1 &= \sum_i r_i, \\
A_1 &= \sum\frac{r_i(r_i+1)}{2}, \\
W_1 &= n_1, \\
R_1 &= 0 .
\end{aligned}
$$

For any subsequent filter, only the $$N_j$$ surviving documents run, and only their $$q = q_{\pi_j}$$ instruction tokens are computed:

$$
\begin{aligned}
n_j &= N_j\,q, \\
A_j &= N_j\left[\,q\,p + \frac{q(q+1)}{2}\right], \\
W_j &= n_j, \\
R_j &= N_j\,p .
\end{aligned}
$$

The first term of $$A_j$$ counts prefix comparisons and the second counts self comparisons. $$R_j$$ is the tokens to fetch documents from the KV cache.

## 4.3 Filter Ordering

Let's consider applying $$m$$ filters to $$N$$ documents. Let $$\pi = (\pi_1,\ldots,\pi_m)$$ denote an arbitrary ordering of the filters, where $$\pi_j$$ is the filter in position $$j$$. Each filter $$F_i$$ has a selectivity $$s_i$$, the fraction of documents it keeps, and can be priced in two ways per document, depending on whether the document is already in the KV cache. The selectivity for each predicate present in [Table 4](#table-4) is measured empirically as the fraction of documents for which the LLM returns True. A *scan* runs on an uncached document and it processes the prefix together with the filter's question and writes the prefix KV, at a cost of $$\text{scan}_i$$. An *ask* runs on a cached document and it processes only the question tokens, which attend to the cached prefix, at a cost of $$\text{ask}_i$$. In an ordering $$\pi$$, the first filter scans all $$N$$ documents and every later filter asks on the documents that survive the filters before it given by the formula below for the total cost.

$$
C(\pi) = N\,\text{scan}_{\pi_1} + \sum_{j=2}^{m} \text{ask}_{\pi_j}\,N\prod_{k<j} s_{\pi_k}.
$$

The product is the fraction of documents that survive the filters preceding position $$j$$.

The scan cost only prices the choice of first filter; it does not define a second ordering. We choose an ordering, $$\pi$$, in three steps.

1. Sort all filters by ascending $$\text{rank}_i$$.

   $$
   \text{rank}_i = \frac{\text{ask}_i}{1 - s_i}.
   $$

   It is the cost paid per document removed, however it ignores scans in the ranking.
2. Evaluate all filter chains with different filters first.

   For each filter $$f$$, form the chain $$\pi^f$$ with $$f$$ first and the other filters in the order of step 1, and compute $$C(\pi^f)$$ with the equation above: the scan of $$f$$ on all $$N$$ documents, plus the ask of each other filter weighted by the remaining documents.
3. Select the cheapest chain. The filter with the smallest $$C(\pi^f)$$ goes first, and the others follow in their step 1 order.

We evaluate $$m$$ chains, not all $$m!$$ orderings, because we only choose which filter goes first and keep the rest in the ask-cost rank order.

We assign ask and scan cost both with the cost model of [Section 3](#3-cost-model-for-one-filter). The cost is sum of three parts, the projections, the MLP, and attention. Each part takes the maximum of its compute time and its memory time. Projections and the MLP compute at the FP8 rate and read their weights every pass. Attention computes at the FP16 rate and reads and writes KV.

# 5. Example: BioDEX Query

We apply the model to the BioDEX query of [Figure 1](#figure-1): choose the filter order, cost the ordered conjunction filter by filter, and read off the SoL estimate. All values are for Qwen3-4B on an H100, with weights in FP8 and attention and KV in FP16, $$N = 200$$ documents, and a cached prefix of $$p = 2 + 4{,}145.96 = 4{,}147.96$$ tokens per document.

## 5.1 Filter Ordering

[Table 4](#table-4) lists the three filters. Each ask and scan cost is the sum of the three components of [Section 3](#3-cost-model-for-one-filter):

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

$$F_8$$ has the cheapest scan, so it is a real candidate for the first slot. [Table 5](#table-5) prices each chain with the equation from [Section 4.3](#43-filter-ordering). Starting with $$F_8$$ saves $$7.4$$~ms of scan cost but makes $$F_7$$ run later on more documents, for a net cost of $$2.4$$~ms, so $$F_7$$ stays first. The number of documents entering each stage is $$N_j = N\prod_{k<j} s_{\pi_k} = 200,\ 111,\ 73$$.

<div class="table-wrap" id="table-5" markdown="1">

| First | Chain $$\pi^f$$ | $$C(\pi^f)$$ (s) |
|-------|---------------|----------------|
| $$F_7$$ | $$F_7 \to F_8 \to F_9$$ | 4.1937 |
| $$F_8$$ | $$F_8 \to F_7 \to F_9$$ | 4.1961 |
| $$F_9$$ | $$F_9 \to F_7 \to F_8$$ | 4.2261 |

<p class="table-caption">Table 5. Step 2 of the ordering procedure: the cost of each candidate chain.</p>
</div>

## 5.2 Cost of the Ordered Conjunction

The prefix KV of a document takes $$147{,}456 \times 4{,}147.96 \approx 0.612$$~GB and must stay resident until its last filter runs. After weights and embeddings, $$75.59$$~GB of HBM remain for KV, so only $$\lfloor 75.59 / 0.612 \rfloor = 123$$ documents fit and the $$200$$ are pipelined as two batches.

<div class="table-wrap" id="table-6" markdown="1">

| Stage | $$n_j$$ | $$A_j$$ | $$R_j$$ | $$T_{\text{proj}}$$ | $$T_{\text{attn}}$$ | $$T_{\text{mlp}}$$ | Stage total |
|-------|-------|-------|-------|--------------------|--------------------|--------------------|-------------|
| 1 ($$F_7$$, prefix) | 838,992 | $$1.760\times10^{9}$$ | 0 | 0.8002 s | 1.0492 s (compute) | 2.2805 s | 4.1299 s |
| 2 ($$F_8$$) | 4,551 | $$1.897\times10^{7}$$ | 460,424 | 4.34 ms | 20.47 ms (memory) | 12.37 ms | 37.18 ms |
| 3 ($$F_9$$) | 3,577 | $$1.493\times10^{7}$$ | 302,801 | 3.41 ms | 13.49 ms (memory) | 9.72 ms | 26.62 ms |
| **Chain** | 847,120 | | 763,225 | **0.8079 s** | **1.0832 s** | **2.3026 s** | **4.1937 s** |

<p class="table-caption">Table 6. Cost of the ordered BioDEX conjunction.</p>
</div>

## 5.3 Speed-of-Light Estimate

[Table 6](#table-6) cleanly provides all of the values computed for the filter chain to achieve its SoL estimate. At peak arithmetic throughput and peak memory bandwidth on an H100, with ideal scheduling and no avoidable KV recomputation,

$$
\begin{aligned}
\text{SoL} &= T_{\text{proj}} + T_{\text{attn}} + T_{\text{mlp}} \\
    &= 0.8079 + 1.0832 + 2.3026 \\
    &= \boxed{4.19~\text{seconds}} .
\end{aligned}
$$

For an implementation of the chained example query on Qwen3-4B and an H100, the observed runtime can now be compared with $$4.19$$~s where a large gap points to inefficiency such as lost KV reuse or idle tensor cores.

# 6. Filter Chain Playground

Build a filter chain below and see its Speed-of-Light latency. Drag filters into the chain, reorder them, and tune each filter's selectivity $$s_i$$ and instruction length $$q_{\text{tail}}$$. The prefix $$q_{\text{pre}}$$ is shared by the whole chain because the filters reuse one prefix KV ([Section 4.1](#41-execution-model-and-kv-reuse)). The playground applies the ordering rule of [Section 4.3](#43-filter-ordering) and, for up to six filters, compares it against every possible order. It is fixed to Qwen3-4B on an H100, and the default is the BioDEX query of [Section 5](#5-example-biodex-query), entered in reverse.

<div class="pg" id="filter-chain-playground" data-filter-chain-calculator>
  <noscript>The playground needs JavaScript. The worked example in Section 5 gives the same numbers for the BioDEX query.</noscript>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Sortable/1.15.6/Sortable.min.js" defer></script>
<script src="{{ '/assets/js/filter-chain-calculator.js' | relative_url }}" defer></script>

All numbers are lower bounds at peak rates, with expected survivors rounded to whole documents and every document at the mean length.

# 7. Conclusion

The cost model allows the estimation of the latency for a given hardware, model specification and workload. An instance of the cost model is SoL estimates which are optimistic lower bounds that are derived from the model. We demonstrated the above cost model to compute the SoL for a single filter and a filter chain query on the BioDEX dataset. Observed runtime can be compared with the SoL to identify any headroom, giving way for further optimizations to inch closer to the SoL. Understanding cost models for different AI-powered operators and over different models and accelerators is crucial for building a holistic understanding of how to optimize systems for any workload.

# Acknowledgements

We thank [Modal](https://modal.com/) for sponsoring the compute used in
this research.

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