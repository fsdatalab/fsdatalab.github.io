---
layout: post
title: "Supporting Hybrid Models in Quail"
date: 2026-11-02
author: "Arnav Dhariya, Shreya Shankar"
permalink: /blog/hybrid-models-in-quail/
unlisted: true
sitemap: false
math: true
typora-root-url: ..
description: "How Quail supports hybrid models that mix full attention with Gated DeltaNet layers: checkpoints of the recurrent state, retention and eviction, and the cost-model changes."
---

<style>
.prose > h1[id] {
  font-size: 1.5rem;
}
.prose > h2[id] {
  font-size: 1.125rem;
  line-height: 1.3;
  margin: 1.25rem 0 .375rem;
}
.prose sup:has(> a[href^="#note-"]) {
  font-size: .75em;
  top: -.45em;
}
.prose .hybrid-table.table-wrap {
  margin-block: 1.25rem;
}
.prose .hybrid-table.table-wrap table,
.prose #table-3.hybrid-table table,
.prose #table-5.hybrid-table table {
  width: fit-content;
  min-width: 0;
  table-layout: auto;
}
.prose .hybrid-table.table-wrap :is(th, td),
.prose #table-5.hybrid-table :is(th, td) {
  min-width: 0;
  width: auto;
  padding: .3rem .55rem;
}
.prose #table-3.hybrid-table :is(th, td) {
  width: auto;
  text-align: left;
}
</style>

<aside class="tldr"><strong>TL;DR:</strong> We now support hybrid models such as <a href="https://qwen.ai/blog?id=qwen3.5">Qwen3.5-4B</a>, which use less memory for long documents, letting Quail keep more documents in memory for reuse. On <a href="https://github.com/fsdatalab/quail-bench">QUAIL-B</a>, we observe a 3.2&times; speedup over vLLM (geometric mean over queries) on Qwen3.5-4B.</aside>

<nav class="post-toc" aria-label="Table of contents">
<strong>Contents</strong>
<ol>
  <li><a href="#1-introduction">Introduction.</a></li>
  <li><a href="#2-gated-deltanet-vs-attention">Gated DeltaNet vs. Attention.</a>
    <ol>
      <li><a href="#21-attention">Attention.</a></li>
      <li><a href="#22-linear-attention-layer-gated-deltanet">Linear Attention Layer (Gated DeltaNet).</a></li>
      <li><a href="#23-short-convolution">Short Convolution.</a></li>
      <li><a href="#24-gated-delta-rule">Gated Delta Rule.</a></li>
    </ol>
  </li>
  <li><a href="#3-checkpoints-in-quail">Checkpoints in Quail.</a>
    <ol>
      <li><a href="#31-checkpoint-size">Checkpoint Size.</a></li>
      <li><a href="#32-retaining-kv-and-checkpoints">Retaining KV and Checkpoints.</a></li>
      <li><a href="#33-eviction">Eviction.</a></li>
    </ol>
  </li>
  <li><a href="#4-updated-cost-model">Updated Cost Model.</a>
    <ol>
      <li><a href="#41-the-gated-delta-rule">The Gated Delta Rule.</a></li>
      <li><a href="#42-example-sales-4-on-qwen35-4b">Example: SALES-4 on Qwen3.5-4B.</a></li>
    </ol>
  </li>
  <li><a href="#5-results">Results.</a></li>
  <li><a href="#6-hybrid-playgrounds">Hybrid Playgrounds.</a>
    <ol>
      <li><a href="#61-checkpoint-playground">Checkpoint Playground.</a></li>
      <li><a href="#62-cost-model-playground">Cost-Model Playground.</a></li>
    </ol>
  </li>
  <li><a href="#7-conclusion">Conclusion.</a></li>
</ol>
</nav>

# 1. Introduction

So far, [Quail](https://fsdatalab.github.io/blog/introducing-quail/)<sup><a href="#note-1">1</a></sup>, our inference engine for AI functions in SQL, has only supported transformers like Qwen3<sup><a href="#note-2">2</a></sup>, which uses grouped-query attention to reduce the size of its key-value (KV) cache. The cache still grows linearly with document length and quickly fills GPU memory. Many new open-weight models, like Qwen3.5, are hybrids, which replace most attention layers with Gated DeltaNet (GDN)<sup><a href="#note-3">3</a></sup> layers, a type of linear attention. GDN layers keep a fixed-size state regardless of document length, which dramatically reduces memory for long documents (e.g., a 16,000-token document requires about 2.36 GB of cached state with Qwen3-4B and 576 MB with Qwen3.5-4B, using bf16 KV). With less memory per document, Quail can keep more documents in memory.

To support hybrids, we changed what Quail saves after reading a document. Along with the KV cache, Quail now saves a *checkpoint* of the GDN layers' state so it can answer another question without processing the document again.

In this post, we describe how we changed Quail's memory management and cost model to account for checkpoints. With Qwen3.5-4B<sup><a href="#note-4">4</a></sup>, Quail achieved a 3.2&times; speedup over vLLM (geometric mean across the 46 [QUAIL-B](https://github.com/fsdatalab/quail-bench) queries we compared).

# 2. Gated DeltaNet vs. Attention

Gated DeltaNet (GDN) and attention behave in vastly different ways. Below, we describe the differences in what they retain, and briefly give an overview of both.

## 2.1 Attention

Attention compares each token with all earlier tokens, so the KV cache grows linearly with document length. Though grouped-query attention (GQA)<sup><a href="#note-5">5</a></sup> shrinks the KV cache, it is only by a constant factor.

After an operator finishes, Quail “rewinds” the cache by discarding the keys and values for the question tokens. It keeps the entries for the preamble and document so the next operator can ask a different question without processing the document again.

## 2.2 Linear Attention Layer (Gated DeltaNet)

Next, we turn to linear attention layers, specifically Gated DeltaNet (GDN) and its recurrence formula. We illustrate the layer for Qwen3.5-4B.

Hybrid architectures combine both full-attention layers and linear attention layers. An example of linear attention layers is GDN layers in Qwen3.5-4B, which has 32 layers of which 8 are full-attention and 24 are GDN. [Figure 1](#figure-1) demonstrates the architecture of the GDN layer with the two parts we will cover, convolution and the gated delta rule, paneled in teal and coral. In the figure, $$h$$ is the input width. The projections produce query, key and value vectors ($$q,k,v$$), an output gate $$z$$, and the inputs to the decay and write gates ($$\alpha,\beta$$). Each head updates its state matrix $$S$$ and uses the query to read out an output vector $$o$$. In the update, $$I$$ is the identity matrix and $$^{\top}$$ denotes transpose.

<figure class="figure-full figure-readable" id="figure-1">
  <img src="/assets/blog/hybrid-models-in-quail/gdn-layer.svg?v=2" alt="Diagram of the state-carrying part of a Qwen3.5-4B Gated DeltaNet layer, in four panels: input, projections, convolution, and the gated delta rule.">
  <figcaption>Figure 1. The state-carrying part of a Qwen3.5-4B GDN layer. SiLU is the activation applied after the convolution.</figcaption>
</figure>

Instead of linear KV growth per token as in full-attention layers, GDN maintains a fixed-size recurrent state, $$S_t$$, after token $$t$$. Each head updates its state for every token, so the state size per document stays constant as the document grows.

## 2.3 Short Convolution

Each GDN layer applies a width-4 convolution over the concatenated $$q$$, $$k$$ and $$v$$ channels ($$16\times128\times2+32\times128=8{,}192$$ channels), which lets a token blend information from its predecessors. It keeps the last 3 inputs per channel, and this buffer becomes part of the checkpoint ([Section 3.1](#31-checkpoint-size)).

## 2.4 Gated Delta Rule

For one head at token $$t$$, let $$k_t$$ and $$v_t$$ be the key and value vectors, with dimensions $$d_k$$ and $$d_v$$. The state $$S_t$$ is a $$d_v \times d_k$$ matrix. The decay gate $$\alpha_t$$ scales the previous state, and the write gate $$\beta_t$$ sets how strongly to correct its prediction for the current key. The gated delta rule is<sup><a href="#note-6">6</a></sup>

$$
S_t = \alpha_t S_{t-1} + \beta_t\bigl(v_t - \alpha_t S_{t-1}k_t\bigr) k_t^{\top}.
$$

The term in parentheses is the difference between the current value and the value predicted by the decayed state. The last factor writes the correction along the current key.

# 3. Checkpoints in Quail

A *checkpoint* is a copy of every GDN layer's state $$S$$ and convolution buffer, taken at the end of a document's reusable prefix: the preamble and the document, before any question. When a later operator asks about that document, Quail loads the checkpoint and the attention KV and computes only the question, instead of recomputing the document. A question starts from the checkpoint and leaves it unchanged, so any number of questions can read it.

We size Quail's memory for Qwen3.5-4B on one H100. [Table 1](#table-1) gives the model dimensions used below.

<div class="table-wrap hybrid-table" id="table-1" markdown="1">

| Quantity | Value | Description |
|----------|-------|-------------|
| Layers | $$L = 32 = 8 \times (3 \text{ GDN} + 1 \text{ attention})$$ | Splits the layers into $$L_g = 24$$ GDN layers, which keep a checkpoint, and $$L_a = 8$$ attention layers, which keep KV. |
| Hidden size $$h$$ | 2,560 | Width of a token vector; sets the floating-point operation counts (FLOPs) for projections and the multilayer perceptron (MLP) ([Section 4](#4-updated-cost-model)). |
| MLP intermediate width $$d_{\text{mlp}}$$ | 9,216 | Sets MLP FLOPs ([Section 4](#4-updated-cost-model)). |
| Attention heads | $$n_h = 16$$ query, $$n_{kv} = 4$$ KV, head dimension $$d_h = 256$$ | Each attention layer stores $$2\,n_{kv}d_h$$ values per token, which sets KV per token. |
| GDN heads | $$H_k = 16$$ key, $$H_v = 32$$ value | The state has one matrix per value head. The key heads also set the convolution channels. |
| GDN head dimension | $$d_k = d_v = 128$$ | Each state matrix is $$d_v \times d_k$$. |
| Convolution width $$w$$ | 4 | The convolution buffer keeps the last $$w-1 = 3$$ inputs. |

<p class="table-caption">Table 1. Qwen3.5-4B shape and description.</p>
</div>

## 3.1 Checkpoint Size

Each GDN layer saves its state $$S$$ and its convolution buffer $$B_{\text{cv}}$$ (the last $$w-1=3$$ inputs over $$D_c = 2H_kd_k + H_vd_v = 8{,}192$$ channels). With the shapes in [Table 1](#table-1), $$S$$ is stored in fp32 (4 bytes per value) and $$B_{\text{cv}}$$ in bf16 (2 bytes per value). Writing $$|S|$$ and $$|B_{\text{cv}}|$$ for their respective sizes in bytes gives

$$
\begin{aligned}
|S| &= H_v\, d_v\, d_k \cdot 4 = 32\cdot128\cdot128\cdot4 = 2{,}048 \text{ KiB}, \\
|B_{\text{cv}}| &= D_c\,(w-1)\cdot 2 = 8{,}192\cdot3\cdot2 = 48 \text{ KiB}.
\end{aligned}
$$

Across all $$L_g = 24$$ GDN layers, the checkpoint size in bytes, $$B_{\text{ckpt}}$$, is

$$
\begin{aligned}
B_{\text{ckpt}} &= L_g\left(|S| + |B_{\text{cv}}|\right) = 24 \cdot 2{,}096 \text{ KiB} \\
&= 49.125 \text{ MiB} = 51{,}511{,}296 \text{ B}.
\end{aligned}
$$

## 3.2 Retaining KV and Checkpoints

As we retain both KV and checkpoints in the high-bandwidth memory (HBM), we must evict a document's KV and checkpoint together. Reusing a document needs both, so the half that survived saves minimal recomputations.

We first convert both checkpoint and KV into a common unit, 16-token pages<sup><a href="#note-7">7</a></sup>, that we can use to represent the space required by checkpoints easily.

**Pages.** With $$L_a = 8$$ attention layers, let $$\kappa$$ be the KV bytes per token and $$P_{\text{page}}$$ the bytes per 16-token page. Each layer stores a key and a value in bf16:

$$
\begin{aligned}
\kappa &= L_a \cdot 2 \cdot n_{kv}\, d_h \cdot 2 = 8\cdot2\cdot4\cdot256\cdot2 = 32{,}768 \text{ B}, \\
P_{\text{page}} &= 16\kappa = 524{,}288 \text{ B}.
\end{aligned}
$$

Let $$c$$ be the checkpoint size in KV pages. Converting its size in bytes gives

$$
\frac{B_{\text{ckpt}}}{\kappa} = 1{,}572 \text{ tokens}, \qquad c = \frac{B_{\text{ckpt}}}{P_{\text{page}}} = 98.25 \text{ pages}.
$$

A retained document of $$\ell$$ tokens costs

$$
\lceil \ell/16 \rceil + c \text{ pages.}
$$

**Available space.** After the model weights and the temporary buffers needed to run a pass are set aside, Quail has $$\text{cap} = 67{,}803$$ pages (35.55 GB) on one H100 for retained documents, following the budget rules of our [previous post](/blog/ai-filter-cost-estimates/).

**Pages served.** Quail shares the pool between whole KV pages and checkpoint slots. A document of $$\ell$$ tokens occupies $$\lceil \ell/16 \rceil + c$$ pages, so the number of documents that fit, $$n_{\text{slots}}$$, and the remaining capacity for KV are

$$
n_{\text{slots}} = \left\lfloor \frac{\text{cap}}{\lceil \ell/16 \rceil + c} \right\rfloor, \qquad \text{KV pages} = \left\lfloor \text{cap} - c\, n_{\text{slots}} \right\rfloor.
$$

These counts assume every document has length $$\ell$$ and is retained. [Table 2](#table-2) evaluates them for four document lengths. The last column is the checkpoint's share of a document's pages.

<div class="table-wrap hybrid-table" id="table-2" markdown="1">

| Tokens | KV pages/doc | Pages/doc | Held | Ckpt % |
|-------:|---------:|----------:|-----:|-----------:|
| 200 | 13 | 111.25 | 609 | 88% |
| 1,000 | 63 | 161.25 | 420 | 61% |
| 4,000 | 250 | 348.25 | 194 | 28% |
| 16,000 | 1,000 | 1,098.25 | 61 | 9% |

<p class="table-caption">Table 2. Documents retained on one H100 with Qwen3.5-4B.</p>
</div>

For example, with 1,000-token documents, $$n_{\text{slots}} = \lfloor 67{,}803 / 161.25 \rfloor = 420$$ checkpoint slots, which is 21.63 GB of checkpoints, and the remaining 26,538 pages leave 13.91 GB available for KV. You can try other document lengths in the [checkpoint playground](#61-checkpoint-playground).

## 3.3 Eviction

Our eviction policy has been to evict the lowest priority first, where the priority of a document $$d$$ of $$\ell_d$$ tokens is the work it is expected to save per page, with $$p_d$$ the probability that a later operator reads it and $$s(\ell_d)$$ the time to recompute it. For a hybrid model, the checkpoint simply counts as $$c = 98.25$$ extra pages of the document, and Quail evicts a document's KV and checkpoint together:

$$
\text{priority}(d) = \frac{p_d\, s(\ell_d)}{\lceil \ell_d/16 \rceil + c}.
$$

# 4. Updated Cost Model

Here, we describe the updates to the cost model since the [last post](/blog/ai-filter-cost-estimates/), as hybrid models break the assumption in the speed-of-light (SoL) estimates that each layer is full-attention. An SoL estimate is the time the roofline formula below<sup><a href="#note-8">8</a></sup> gives for a query: each operation would take the larger of its arithmetic time and memory-transfer time, and the times are summed over the query. For component $$j$$, let $$T_j$$ be its estimated time in seconds and $$T$$ the total. Here $$F_j$$ and $$B_j$$ are the FLOPs and HBM bytes of one piece of the forward pass, $$\Pi_j$$ the arithmetic rate for that component in floating-point operations per second, and $$\beta_{\text{hbm}} = 3.35$$ TB/s the HBM bandwidth, taking the H100 values from its datasheet<sup><a href="#note-9">9</a></sup>.

$$
T_j = \max\!\left(\frac{F_j}{\Pi_j},\ \frac{B_j}{\beta_{\text{hbm}}}\right), \qquad T = \sum_j T_j.
$$

We keep the notation of the previous post: $$n$$ tokens computed, $$A$$ causal query–key token pairs (including self-pairs, counted before multiplying by heads and layers), $$W$$ and $$R$$ tokens whose KV is written and read, $$C = 116{,}508$$ the token budget of one pass (set by 32-bit kernel indexing), and $$K = \lceil n/C \rceil$$ the number of passes. For hybrids we add the split $$L_a = 8$$ attention and $$L_g = 24$$ GDN layers, and the checkpoints written and read, $$U_W$$ and $$U_R$$, each of $$B_{\text{ckpt}}$$ bytes.

[Table 3](#table-3) lists the only terms that change relative to a transformer. The projection and MLP terms keep their form but the projection parameters now depend on the layer type; the attention terms keep their form but sum over $$L_a = 8$$ layers, not 32; and GDN, the short convolution and checkpoint movement are new. In the table, $$P_{\text{attn}}$$ and $$P_{\text{gdn}}$$ count projection weights per attention and GDN layer. $$P_{\text{proj}}$$ and $$P_{\text{mlp}}$$ count projection and feed-forward (MLP) weights across all layers. For either total $$P$$, $$F$$ counts two FLOPs per weight per token and $$B$$ counts two bytes per weight per pass in bf16.

<div class="table-wrap hybrid-table" id="table-3" markdown="1">

| Term | What changes | Formula |
|------|--------------|---------|
| $$T_{\text{attn}}$$ | only $$L_a = 8$$ layers | $$F_{\text{attn}} = 4L_a n_h d_h A = 131{,}072\,A$$; $$B_{\text{attn}} = \kappa(W+R)$$, $$\kappa = 32{,}768$$ B |
| $$T_{\text{proj}}$$, $$T_{\text{mlp}}$$ | per-layer-type projections | $$P_{\text{proj}} = L_aP_{\text{attn}} + L_gP_{\text{gdn}}$$ $$= 1{,}304{,}166{,}400$$<br>$$P_{\text{mlp}} = 3L\,h\,d_{\text{mlp}}$$ $$= 2{,}264{,}924{,}160$$<br>$$F = 2Pn$$, $$B = 2PK$$ |
| $$T_{\text{gdn}}$$ | new | $$F_{\text{gdn}} = 140{,}509{,}184\,n$$ (below), no byte term |
| $$T_{\text{conv}}$$ | new | $$F_{\text{conv}} = 2wD_cL_g\,n = 1{,}572{,}864\,n$$ at the fp32 rate, no byte term |
| $$T_{\text{state}}$$ | new | $$B_{\text{ckpt}}(U_W + U_R)/\beta_{\text{hbm}}$$: each checkpoint written or read moves $$B_{\text{ckpt}}$$ bytes |

<p class="table-caption">Table 3. Cost-model terms for Qwen3.5-4B.</p>
</div>

The projections and the MLP are matrix multiplications, and we price them at 857 TFLOP/s (the rate we measure for bf16 matrix multiplies on our H100). We price attention and the delta rule at the dense bf16 peak, $$\Pi_{\text{bf16}} = 989.5$$ TFLOP/s.

## 4.1 The Gated Delta Rule

The recurrence of [Section 2.4](#24-gated-delta-rule) updates $$S_t$$ per token, but a GPU kernel splits the sequence into chunks of $$C_g = 64$$ tokens and rewrites most of the work as matrix multiplies on tensor cores. The cost model estimates $$f_{\text{gdn}}$$ FLOPs per token and value head. With key and value dimensions $$d_k = d_v = 128$$,

$$
\begin{aligned}
f_{\text{gdn}} &= 6C_gd_k + 4C_gd_v + 6d_kd_v + \tfrac{2}{3}C_g^2 \\
&= 49{,}152 + 32{,}768 + 98{,}304 + 8{,}192/3 \\
&\approx 182{,}955.
\end{aligned}
$$

The terms with $$C_g$$ are the work inside a chunk. The $$6d_kd_v$$ term is the work on the state. Summing over $$H_v = 32$$ value heads and $$L_g = 24$$ layers:

$$
F_{\text{gdn}} = f_{\text{gdn}}\, H_v L_g\, n = 140{,}509{,}184\,n, \qquad T_{\text{gdn}} = \frac{F_{\text{gdn}}}{\Pi_{\text{bf16}}}.
$$

## 4.2 Example: SALES-4 on Qwen3.5-4B

To understand how the above cost model would be used to evaluate the latency of a query, we exemplify it below on the SALES-4 query from QUAIL-B. SALES-4 comes from QUAIL-B at scale factor 0.1. We use it because its two filters make the arithmetic easy to follow. [Figure 2](#figure-2) shows the discount filter $$\phi_d$$ and the commitment filter $$\phi_c$$.

<figure class="figure-full figure-readable" id="figure-2">
  <img src="/assets/blog/hybrid-models-in-quail/sales4-query.svg?v=2" alt="A database instance of two sales calls, an AI_IF conjunction query applying a discount filter and a commit filter, and the expected TRUE/FALSE output; a dash marks a filter skipped.">
  <figcaption>Figure 2. SALES-4 applies both filters to the sales-call transcripts. A dash marks a filter skipped because the transcript already failed an earlier one. The figure shows the filters in the order written, and Quail may reorder them.</figcaption>
</figure>

**Ordering.** A query is built from two kinds of calls, as in the [filter post](/blog/ai-filter-cost-estimates/). Let $$p$$ be the prefix length in tokens (preamble plus transcript), $$q$$ the question length and $$r=p+q$$ the full prompt length. A *scan* computes the whole prompt and writes one checkpoint, so $$n=W=r$$ and $$U_W=1$$. An *ask* computes only the question and reads the prefix and checkpoint, so $$n=W=q$$, $$R=p$$ and $$U_R=1$$. An ask has $$A=qp+q(q+1)/2$$ causal attention pairs. The mean prefix length is about $$18+1{,}162=1{,}180$$ tokens.

For filter $$i$$, Table 4 uses question length $$q_i$$ and selectivity $$s_i$$ (the fraction of transcripts that pass). Its ask cost is the estimated time per document in a full batch, charging $$q_i/C$$ of a pass for weight reads. The rank is the ask cost divided by $$1-s_i$$, the fraction the filter eliminates.<sup><a href="#note-10">10</a></sup> The planner runs filters in ascending rank.

<div class="table-wrap hybrid-table" id="table-4" markdown="1">

| Filter | $$q_i$$ | $$s_i$$ | ask | rank |
|--------|--------:|--------:|----:|-----:|
| $$\phi_c$$ (commits) | 83 | 33/985 | 733.9 | 759.3 |
| $$\phi_d$$ (discount) | 78 | 104/985 | 690.6 | 772.1 |

<p class="table-caption">Table 4. SALES-4 filters, sorted by ascending rank. ask and rank are in &micro;s per document.</p>
</div>

We compute the total $$T$$ for each order by summing the roofline terms over that order's exact work counts, so the totals are

$$
T(\phi_c,\phi_d) = 10.485 \text{ s}, \qquad T(\phi_d,\phi_c) = 10.496 \text{ s}.
$$

The planner runs the commit question first. The choice changes the estimate by only 0.1%, because both orders compute nearly the same tokens ($$n = 1{,}218{,}236$$ against $$1{,}219{,}369$$). [Table 5](#table-5) gives the commit-first breakdown.

**Work counts.** The estimate for the whole query is broken into two stages, one per filter. Stage 1 asks $$\phi_c$$ of all 985 transcripts and 33 pass. Stage 2 asks $$\phi_d$$ of those 33 and 10 pass. Of the 985 transcripts, 984 share a prefix with an earlier one (10,853 tokens in all, beyond the preamble), so each starts from the saved state of that shared part and computes the rest. The first transcript is a full scan. Stage 2 starts from each survivor's checkpoint and computes only the 78-token question. The transcripts contain 1,144,742 tokens in total, including 38,968 in the 33 survivors. The estimator counts attention pairs using individual prompt lengths. Summing gives

$$
\begin{aligned}
n &= 18 + 1{,}144{,}742 - 10{,}853 + 985\cdot83 + 33\cdot78 = 1{,}218{,}236, \\
A &= 795{,}539{,}705, \\
W &= 1{,}218{,}236, \\
R &= 984\cdot18 + 10{,}853 + 33\cdot18 + 38{,}968 = 68{,}127, \\
U_W &= 985, \qquad U_R = 984 + 33 = 1{,}017, \\
K &= \lceil 1{,}218{,}236 / 116{,}508 \rceil = 11.
\end{aligned}
$$

We write a checkpoint per transcript to the HBM, as the [prior post](/blog/ai-filter-cost-estimates/) keeps saved vectors in HBM until the document fails a filter or passes the final filter, and we continue that assumption for simplicity.

<div class="table-wrap hybrid-table" id="table-5" markdown="1">

| Term | $$F_j$$ | $$\Pi_j$$ (TFLOP/s) | $$F_j/\Pi_j$$ | $$B_j$$ | $$B_j/\beta_{\text{hbm}}$$ | $$T_j$$ |
|------|--------:|----------:|--------------:|--------:|---------------------------:|--------:|
| $$T_{\text{mlp}}$$ | $$5.52\times10^{15}$$ | 857 | 6.439 s | 49.8 GB | 0.015 s | 6.439 s |
| $$T_{\text{proj}}$$ | $$3.18\times10^{15}$$ | 857 | 3.708 s | 28.7 GB | 0.009 s | 3.708 s |
| $$T_{\text{gdn}}$$ | $$1.71\times10^{14}$$ | 989.5 | 0.173 s | 0 | 0 | 0.173 s |
| $$T_{\text{attn}}$$ | $$1.04\times10^{14}$$ | 989.5 | 0.105 s | 42.2 GB | 0.013 s | 0.105 s |
| $$T_{\text{state}}$$ | 0 | --- | 0 | 103.1 GB | 0.031 s | 0.031 s |
| $$T_{\text{conv}}$$ | $$1.92\times10^{12}$$ | 67 | 0.029 s | 0 | 0 | 0.029 s |
| **$$T$$ (total)** | | | | | | **10.485 s** |

<p class="table-caption">Table 5. SoL estimate for SALES-4 at scale factor 0.1 on Qwen3.5-4B bf16 and one H100. Quail's estimator uses individual transcript lengths. The <a href="#62-cost-model-playground">cost-model playground</a> uses mean document and shared-prefix lengths and expected pass counts.</p>
</div>

In [Table 5](#table-5), $$T_{\text{mlp}}$$ and $$T_{\text{proj}}$$ are 96.8% of the total. The new terms the hybrid adds are $$T_{\text{gdn}}$$, $$T_{\text{conv}}$$ and $$T_{\text{state}}$$. The short convolution adds 0.029 s (0.3%) at the fp32 rate.

Without the checkpoint, stage 2 would recompute 39,562 tokens (3.2% of $$n$$), or about 0.33 s for projection and MLP work alone. Checkpoint reads and writes across both stages cost an estimated 0.031 s.

# 5. Results

We compare Quail with vLLM on Qwen3.5-4B (bf16). The question is whether the changes of the previous sections pay off on real queries. We run QUAIL-B at scale factor 0.1 on one H100. vLLM runs each query two ways, stock (one operator at a time) and pipelined, and we compare Quail with the faster of the two on each query. Four queries do not run on vLLM, so 46 queries are compared. Model startup is excluded from every time.

vLLM is version 0.29.0 with prefix caching on, so requests that share a prefix reuse its KV. For the GDN layers, vLLM saves the recurrent state at block boundaries, so it can also resume from a shared prefix. We set `max_num_seqs=2047`, the most sequences its GDN kernel can launch for this model, `max_num_batched_tokens=25305` and `gpu_memory_utilization=0.852`, the same on every query. Pipelined vLLM submits an operator's requests as soon as rows reach it, instead of after the previous operator finishes. For each query, $$n_{\text{fresh}}$$ counts the tokens the engine computes, and $$n_{\min}$$ is the benchmark's estimated minimum with unlimited cache. KV regret is the fraction of fresh tokens above that minimum:

$$
\text{KV regret} = \frac{n_{\text{fresh}} - n_{\min}}{n_{\text{fresh}}}.
$$

For agreement, we use each query's predicate-answer accuracy against Qwen3-32B-FP8, or its classification-label accuracy when the query has no predicate answers. Each query counts equally.

<div class="table-wrap hybrid-table" id="table-6" markdown="1">

| Metric | Quail | vLLM |
|--------|------:|-----:|
| Queries where faster | 45/46 | 1/46 |
| Total runtime (s) | 2,650 | 16,854 |
| KV regret (%) | 6.8 | 57.1 |
| Agreement | .889 | .888 |

<p class="table-caption">Table 6. Quail against the faster vLLM configuration on the 46 queries vLLM can run, Qwen3.5-4B bf16. Totals sum query time; regret and agreement are means over queries.</p>
</div>

Quail is 3.2&times; faster than vLLM: the geometric mean, over the 46 queries, of the faster vLLM configuration's time divided by Quail's. Quail's mean KV regret is 6.8% against 57.1% for vLLM. Agreement with the reference labels differs by less than 0.1 percentage points between engines.

Quail is faster on 45 of the 46 queries and on par on the last one. On REL-AGENT-1, a single filter over 186 traces, vLLM takes 12.93 s against 13.20 s for Quail (2.1%). Quail spends 12.52 s running the model and about 0.68 s preparing the input.

**Where the speedup comes from.** Per query, the speedup is the product of two ratios: how many fewer tokens Quail computes (vLLM's fresh tokens over Quail's) and how fast it computes each one (Quail's fresh tokens per second over vLLM's). Over the 46 queries, the geometric means are 3.219 for the fresh-token ratio and 0.984 for the per-token rate ratio (their product is 3.168), so the speedup comes from recomputing less. In total, Quail computes 165M fresh tokens against 1,002M for vLLM.

**Limitations and follow-ups.** The cost model currently counts only compute time for GDN. In practice, the delta rule and convolution are limited by memory bandwidth. We plan to account for the time spent moving data and cover GPU profiling in a separate post. Other hybrids have different checkpoint sizes. LFM2, for example, keeps only the convolution buffer for each recurrent layer.

# 6. Hybrid Playgrounds

## 6.1 Checkpoint Playground

Change the document length to see how many documents each model can keep on one H100.

<div class="pg" id="hybrid-checkpoint-playground" data-hybrid-capacity>
  <noscript>The playground needs JavaScript. Table 2 gives the same numbers for four document lengths.</noscript>
</div>

## 6.2 Cost-Model Playground

Start with SALES-4 on Qwen3.5-4B. Change a filter to see how it affects the estimated query time.

<div class="pg" id="hybrid-cost-playground" data-hybrid-sales4>
  <noscript>The playground needs JavaScript. Table 5 gives the same breakdown for the commit-first order.</noscript>
</div>
<script src="{{ '/assets/js/hybrid-playground.js' | relative_url }}" defer></script>

# 7. Conclusion

<aside aria-label="Comment from Shreya" style="padding: .75rem 1rem; border-radius: 6px; background: #edf3fc; color: #183b68;"><strong style="color: inherit;">Shreya:</strong> Add a conclusion?</aside>

# Acknowledgements

We thank [Modal](https://modal.com/) for sponsoring the compute used in this research.

# Notes

<span id="note-1"><strong>1.</strong></span> Shreya Shankar, Charles Frye, Fergus Finn, Arnav Dhariya, Joseph Barrow, and Meryem Arik. 2026. Building an Ultra-High Throughput AI-SQL Engine. [https://fsdatalab.github.io/blog/introducing-quail/](https://fsdatalab.github.io/blog/introducing-quail/)

<span id="note-2"><strong>2.</strong></span> An Yang et al. 2025. Qwen3 Technical Report. [arXiv:2505.09388](https://arxiv.org/abs/2505.09388).

<span id="note-3"><strong>3.</strong></span> Songlin Yang, Jan Kautz, and Ali Hatamizadeh. 2025. [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464). In *International Conference on Learning Representations (ICLR)*.

<span id="note-4"><strong>4.</strong></span> Qwen Team. 2026. Qwen3.5: Towards Native Multimodal Agents. [Qwen3.5 announcement](https://qwen.ai/blog?id=qwen3.5).

<span id="note-5"><strong>5.</strong></span> Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebron, and Sumit Sanghai. 2023. [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://aclanthology.org/2023.emnlp-main.298/). In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, 4895&ndash;4901.

<span id="note-6"><strong>6.</strong></span> Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, and Yoon Kim. 2024. [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484). In *Advances in Neural Information Processing Systems (NeurIPS)*.

<span id="note-7"><strong>7.</strong></span> Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. 2023. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180). In *Proceedings of the 29th ACM Symposium on Operating Systems Principles (SOSP)*.

<span id="note-8"><strong>8.</strong></span> Samuel Williams, Andrew Waterman, and David Patterson. 2009. Roofline: an insightful visual performance model for multicore architectures. *Commun. ACM* 52, 4, 65&ndash;76. [doi:10.1145/1498765.1498785](https://doi.org/10.1145/1498765.1498785)

<span id="note-9"><strong>9.</strong></span> The H100 SXM memory capacity, bandwidth, and Tensor Core throughput come from [NVIDIA's H100 specifications](https://resources.nvidia.com/en-us-gpu-resources/h100-datasheet-24306). NVIDIA reports Tensor Core throughput for structured sparse matrices; we use the dense throughput, which is half the reported sparse throughput.

<span id="note-10"><strong>10.</strong></span> Joseph M. Hellerstein and Michael Stonebraker. 1993. Predicate Migration: Optimizing Queries with Expensive Predicates. In *Proceedings of the 1993 ACM SIGMOD International Conference on Management of Data*, 267&ndash;276. [doi:10.1145/170035.170078](https://doi.org/10.1145/170035.170078)

# Cite this article

<div class="bibtex-block" markdown="1">
<button class="copy-bibtex" type="button">Copy BibTeX</button>

```bibtex
@misc{dhariya2026hybrid,
  title = {Supporting Hybrid Models in Quail},
  author = {Dhariya, Arnav A. and Shankar, Shreya},
  year = {2026},
  month = nov,
  url = {https://fsdatalab.github.io/blog/hybrid-models-in-quail/}
}
```

</div>
