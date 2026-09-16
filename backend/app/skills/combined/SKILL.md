---
name: combined
description: >-
  questions that need BOTH the employee's personal data (via Workday) AND a
  policy document to answer (e.g. eligibility questions that depend on the
  employee's own country/tenure/profile)
category: hr
---

## Answer guidance

Apply the policy rules to the employee's actual profile data rather than
restating the policy generically -- this mirrors `COMBINE_PROMPT` in
`llm/prompts.py`, the prompt actually used for the final combine step. This
route's `category: hr` matches every combined-route example in this project
(see `docs/example_queries.md`) and is what `agents/rag_agent.py`'s
`ROUTE_TO_CATEGORY` derives from the registry.
