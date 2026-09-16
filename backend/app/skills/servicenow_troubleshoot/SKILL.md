---
name: servicenow_troubleshoot
description: >-
  the employee wants IT troubleshooting AND/OR wants a support ticket
  created or checked
---

## Notes

Handled by `agents/it_agent.py`'s multi-turn troubleshoot/check-resolution
flow, not by the RAG pipeline. No `category`, no body consumed at
runtime -- present here purely so this route's description lives in the
registry alongside every other route.
