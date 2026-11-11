---
layout: default
title: Get Involved
description: "Want to do research with the Full Stack Data Lab? Start by using Quail, our open-source AI-SQL engine, to answer a real question over a large dataset. Here are project ideas and how to get started on Modal's free tier."
permalink: /get-involved/
---
<article class="content-narrow prose" markdown="1">

<div class="post-header"><h1>Get involved</h1></div>

Many people reach out wanting to do research with us, and we can't meet
with everyone. So here is how to get started without waiting on us. **Use
[Quail](https://github.com/fsdatalab/quail), our open-source AI-SQL
engine, to answer a real question over a large dataset, and send us what
you found.**

A project like this helps everyone. You learn how LLM-powered data
processing works in practice, and you end up with something concrete to
show anyone, not just us. We learn which workloads Quail handles badly,
where its answers are wrong, and what to build next. A good project tells
us more about how you think than a cover letter does.

You don't need to be a systems person. Some of the most useful projects
will come from people who know a domain (medicine, law, policy, history,
finance) and know which questions in it are worth answering.

## How it works

1. **Pick a problem.** Choose one of the [ideas below](#project-ideas),
   or bring your own. A good problem has a real question that someone cares
   about, and a dataset too big to read by hand. Thousands to millions of
   rows is the sweet spot: big enough that calling an LLM API on every row
   gets expensive.
2. **Get compute.** Sign up for [Modal](https://modal.com/pricing). Its
   free Starter plan includes $30 of compute each month, which is about 7.5
   hours on an H100 at the price we used in our [Quail
   post]({{ '/blog/introducing-quail/' | relative_url }}) (check Modal's
   pricing page for current rates). That is a lot of AI-SQL: in that post,
   two AI filters over all 100,000 IMDB reviews cost $0.37, including model
   startup. Quail's repo has a [Modal
   quickstart](https://github.com/fsdatalab/quail/blob/d8d31f14f9d5c40d6cf683a74ba860788ce0d500/demos/quickstart_modal.py)
   that runs a query inside a Modal GPU function.
3. **Start small.** Load a sample of 500 to 1,000 rows and iterate on your
   prompts there. Only run the full dataset once your queries do what you
   want.
4. **Check the answers.** Label 100 or so rows by hand, or use existing
   labels if the dataset has them, and measure how often Quail's answers
   agree with yours. Read the disagreements. Your prompts and criteria will
   change once you read real outputs, and that is normal.
5. **Write it up.** Put your code in a public repo with a short report:
   the question, the data, your queries, the answer, how accurate it was,
   and what it cost and how long it took. Tell us what was slow, confusing,
   or broken in Quail, and file [GitHub
   issues](https://github.com/fsdatalab/quail/issues) for bugs.
6. **Send it to us.** Email [shreyashankar@cmu.edu](mailto:shreyashankar@cmu.edu)
   with a link to the repo, your CV, and (for students) your transcript.
   Please don't write the email with AI.

Quail supports AI functions in SQL such as `AI.IF` (filters and joins),
`AI.CLASSIFY`, `AI.EXTRACT`, and `AI.MAP`, running open-weight models on
GPUs. Our [AI-SQL page]({{ '/ai-sql/' | relative_url }}) and
[blog]({{ '/blog/' | relative_url }}) explain how each one works.

## What makes a project good

- **The question matters to someone.** A finding that a journalist,
  clinician, regulator, or researcher would use beats a toy demo. If you
  can, talk to someone who works with the data.
- **You measure quality.** "The LLM said so" is not a result. Report
  agreement with labels, and show examples of where it fails.
- **You report cost and runtime.** How many rows, how many tokens, how
  long, and how many dollars. Compare against an LLM API if you can.
- **You push on Quail.** Multi-step queries, joins, long documents, and
  many AI functions over the same rows stress the engine in interesting
  ways. Finding where Quail is slow or wrong is a contribution.

## Project ideas

These are real public datasets with open questions. Ideas marked
**has labels** come with human annotations that you can use to measure
accuracy. Before you start, read each dataset's license and terms of use.

### Health and science

#### Screening papers for systematic reviews (has labels)

A systematic review starts with people reading thousands of paper
abstracts to decide which ones meet the review's inclusion criteria. The
[SYNERGY dataset](https://github.com/asreview/synergy-dataset) has
169,288 papers from 26 published reviews, with the authors' include or
exclude decisions; only about 1.7% were included. Turn each review's
criteria into `AI.IF` filters and measure recall: for screening, missing
a relevant paper is much worse than including an extra one. Can a cascade
of cheap filters find every included paper while cutting the reading load
by 10x?

#### Matching patients to clinical trials (has labels)

Finding a trial a patient is eligible for means checking their history
against each trial's inclusion and exclusion criteria.
[ClinicalTrials.gov](https://clinicaltrials.gov/) has close to 600,000
registered studies, and the [TREC Clinical Trials
track](https://trec-cds.org/2022.html) provides patient descriptions
with judgments of which trials each patient is eligible for, excluded
from, or not relevant to. This is a natural AI join between patients and
trials. Negation makes it hard ("non-smokers" in inclusion criteria versus
"smokers" in exclusion criteria), and the join is large, which makes it a
good stress test for Quail's planner.

#### Mining the biomedical literature

[PubMed's annual baseline](https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/)
contains tens of millions of citations, most with abstracts. Pick a
question that keyword search answers badly, and use `AI.EXTRACT` and
`AI.CLASSIFY` to answer it: which trials in a field report null results,
how sample sizes have changed over time, or which studies use a method you
care about.

### Safety and consumer protection

#### Coding workplace injury narratives (has labels)

OSHA publishes [Severe Injury Reports](https://www.osha.gov/severe-injury-reports),
employer reports of amputations, hospitalizations, and loss of an
eye since 2015. Each report includes a free-text narrative, and OSHA staff code it
with the Occupational Injury and Illness Classification System (OIICS)
for the injury's nature, body part, event, and source. Use `AI.CLASSIFY`
to code the narratives and compare against OSHA's codes. Then look for
patterns the codes miss.

#### Early warning for vehicle defects

NHTSA's Office of Defects Investigation publishes [owner complaints about
vehicles](https://catalog.data.gov/dataset/nhtsas-office-of-defects-investigation-odi-complaints-complaints-flat-file),
along with recalls and investigations. Could an AI-SQL query have flagged
a defect before the recall? Classify complaints by failure mode, then join
complaints against later recall descriptions to see how early the signal
appeared.

#### Aviation safety reports

NASA's [Aviation Safety Reporting System](https://asrs.arc.nasa.gov/search/database.html)
holds de-identified, voluntary reports from pilots, controllers, and crew
going back to 1988. Extract contributing factors (fatigue, miscommunication,
automation confusion) and track how they change over time. The online database exports search results as CSV.

#### Consumer finance complaints

The CFPB's [Consumer Complaint
Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
includes consumers' own descriptions of their problems with banks,
lenders, and debt collectors, for those who consented to publish them.
Each complaint also has a product and issue category chosen at submission
time. How often do those categories match the narrative? Which complaints
describe the same practice at different companies?

### Government and policy

#### Public comments on federal regulations

Agencies must read and respond to public comments on proposed rules, and
popular rules get hundreds of thousands, many from organized campaigns.
[Regulations.gov](https://www.regulations.gov/) publishes them. For a few
dockets, separate form letters from substantive comments, classify each
comment's position, and extract its arguments. Does the agency's final rule
respond to the arguments that appear most often? (The API is
rate-limited, so pick dockets early and download their comments slowly.)

#### Tracking legislation across states

[Open States](https://open.pluralpolicy.com/data/) publishes bulk data on
bills in all 50 state legislatures. Find every bill on a topic (AI, housing,
data privacy) across states, and extract what each one actually does. A
harder version is an AI join between bills to find model legislation:
bills in different states that share language or provisions.

#### Police misconduct records

The [Police Records Access
Project](https://bids.berkeley.edu/california-police-records-access-project)
at UC Berkeley and Stanford collected about 1.5 million pages of records
from 12,000 misconduct and use-of-force cases across nearly 700 California
agencies. Extract each case's allegations, findings, and discipline, and
compare across agencies. The records are redacted PDFs, so you will need to
turn them into text first, and bulk access may need coordination with the
project, so please ask us before starting.

### History, law, and finance

#### A century and a half of American newspapers

[American Stories](https://huggingface.co/datasets/dell-research-harvard/AmericanStories)
extracts article text from nearly 20 million newspaper page scans in the
Library of Congress's Chronicling America collection, from 1774 to 1963.
Pick a historical question, such as how papers covered the 1918 flu, how
coverage of an event spread across regions, or how job advertisements
changed, and answer it at full scale instead of on a sample.

#### Court opinions

CourtListener offers [bulk downloads of court
opinions](https://www.courtlistener.com/help/api/bulk-data/) from federal
and state courts. Find every opinion that addresses a legal question and
extract how it was decided and why. You can check your answers on a sample
against legal researchers' summaries.

#### Risk disclosures in 10-K filings

[EDGAR-CORPUS](https://huggingface.co/datasets/eloukas/edgar-corpus)
splits the annual 10-K reports of all US public companies from 1993 to 2020
into sections. Use the risk factors section (Item 1A) to track how
companies describe a risk, such as climate, cybersecurity, or supply chain,
and whether they describe it concretely or in boilerplate.

### AI and data

#### What do people actually use chatbots for?

[WildChat-4.8M](https://huggingface.co/datasets/allenai/WildChat-4.8M)
contains millions of real conversations between users and ChatGPT,
collected with consent. Classify what users are trying to do, where the
assistant fails them, and how usage changes over time. This dataset is
large enough that the cost of an LLM API really adds up.

#### Filtering web data for pretraining

To build [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
Hugging Face had Llama-3-70B score about 460,000 web pages for educational
value, then trained a small classifier on those scores to label the rest.
With a fast AI-SQL engine, how far can you get with an LLM judging every
page directly? Compare its labels with the classifier's, and estimate the
cost of each approach at web scale.

#### Why do coding agents fail?

Nebius released [80,036 SWE-agent
trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories)
of an agent attempting to fix real GitHub issues, with whether each attempt
succeeded. Use `AI.EXTRACT` and `AI.MAP` to find failure modes (looping,
editing the wrong file, ignoring test output) and how they relate to
success. Agent traces are long and Quail is currently slower than vLLM on
some agent queries in our benchmark, so you will help us improve
Quail, too.

### Bring your own

If you have a dataset and a question that fit the description above, go for
it. We especially want to hear from people outside computer science who have
unstructured data (documents, transcripts, reports) and no engineering team
to analyze it.

</article>
