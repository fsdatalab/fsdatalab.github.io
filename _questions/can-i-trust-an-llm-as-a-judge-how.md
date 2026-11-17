---
title: "Can I trust an LLM-as-a-judge? How do I know it agrees with human reviewers?"
section: human-behavior
order: 2
description: "Only after you check it against your own labels on a sample of real outputs. Our EvalGen paper describes a tool for this: you grade a subset of outputs, and it uses your grades to pick the candidate judge prompts and code checks that best match them. Our user study found that alignment is subjective and iterative, so keep checking as outputs change. The FAQ entry \"How do I know if I can trust my automated eval?\" gives a workflow."
sources:
  - title: "Who Validates the Validators? Aligning LLM-Assisted Evaluation of LLM Outputs with Human Preferences (UIST 2024)"
    url: https://arxiv.org/abs/2404.12272
  - title: "AI Evals: Everything You Need to Know (Hamel Husain and Shreya Shankar)"
    url: https://hamel.dev/blog/posts/evals-faq/
asked:
  - https://www.reddit.com/r/LLMDevs/comments/1uacjzp/can_you_actually_trust_llmasjudge/
  - https://www.reddit.com/r/LLMDevs/comments/1tocdg1/my_llmasjudge_had_cohens_kappa_of_047_promptfoo/
  - https://www.reddit.com/r/AI_Agents/comments/1sx9vn1/how_are_people_actually_trusting_llm_eval_scores/
---
