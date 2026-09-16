---
name: workday
description: >-
  questions about the employee's OWN Workday HR data (their vacation
  balance, their benefits enrollment, their profile) with NO document
  lookup needed
---

## Notes

No document retrieval happens on this route; `agents/employee_agent.py`
calls Workday tools directly (`get_employee`, `get_leave_balance`,
`get_benefits`). This SKILL.md carries no `category` and its body is never
loaded into a prompt -- it exists purely so the routing description lives
in the same registry as every other route, per
`docs/concepts/skills_pattern.md`.
