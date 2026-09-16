---
name: rag_personal
description: >-
  questions about the employee's own uploaded personal documents (e.g. a
  phone/internet bill, an insurance policy, a flight e-ticket/PNR) -- NOT
  company policy, and NOT Workday employee-profile data
category: personal
match_priority: 0
keywords:
  - airtel
  - insurance
  - premium
  - pnr
  - invoice
  - policy no
  - sum assured
  - e-ticket
  - boarding
---

## Answer guidance

These are the employee's own real-world documents, not company policy --
never blend the two (e.g. don't answer an insurance-premium question with
company benefits-policy text just because both mention "insurance"). If a
figure comes from OCR'd text (see `ingestion/loader.py`), treat digits as
slightly less reliable than born-digital text and say so if the context
looks garbled.
