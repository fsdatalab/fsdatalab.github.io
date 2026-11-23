---
title: "Why does my LLM agent run so many SQL queries for one question, and how do I cut them down?"
section: database-systems
order: 21
description: "Agents explore schemas and try partial queries before the final one, and more attempts do raise accuracy. In Supporting Our AI Overlords (CIDR 2026), more attempts raised success on the BIRD text-to-SQL benchmark by 14% to 70%, but across 50 attempts per task, distinct sub-plans were often under 10-20% of the total, so much of the work repeats. We recommend giving the agent grounding hints, such as which column holds the needed data; that cut queries by more than 20%."
sources:
  - title: "Supporting Our AI Overlords: Redesigning Data Systems to be Agent-First (CIDR 2026)"
    url: https://arxiv.org/abs/2509.00997
asked:
  - https://reddit.com/r/learnmachinelearning/comments/1shx8ag/help_curious_if_anyone_here_has_tackled/
  - https://news.ycombinator.com/item?id=48399793
---
