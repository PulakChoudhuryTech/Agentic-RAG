# Example Queries

Try these against the Streamlit "Chat" tab, or directly:

```bash
curl -s -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id": "demo-1", "message": "What is the parental leave policy in India?"}' | jq
```

| # | Query | Route | What happens |
|---|-------|-------|--------------|
| 1 | "What is the parental leave policy in India?" | `rag_hr` | Metadata router detects `category=hr, country=IN`; the RAG pipeline runs scoped to those filters and answers from the India section of `hr/parental_leave_policy.md`. |
| 2 | "How many vacation days do I have?" (as employee `E1002`) | `workday` | The employee agent calls the `get_leave_balance` tool against the mock Workday API -- no document retrieval involved. |
| 3 | "Am I eligible for parental leave based on my profile?" (as `E1001`) | `combined` | The employee agent looks up the employee's country/tenure via Workday, the RAG agent retrieves the parental leave policy, and `combine_node` asks Gemini to apply the policy to the actual profile data. |
| 4 | "My VPN isn't working. What should I do?" | `rag_it` | Pure RAG over `it/vpn_troubleshooting.md` -- no tool calls, just documentation. |
| 5 | "My VPN isn't working. Try troubleshooting and create a ServiceNow ticket if it still doesn't work." | `servicenow_troubleshoot` | Turn 1: RAG-sourced troubleshooting steps, asks for confirmation. Turn 2 (if you reply that it's still broken): the LLM judges the issue unresolved and calls the `create_ticket` tool against the mock ServiceNow API. See `docs/architecture.md` for the full conditional-workflow diagram. |

## A few more to try

- "What benefits am I enrolled in?" (as any employee) → `workday`, calls `get_benefits`.
- "What is the hotel nightly cap for business travel in Bangalore?" → `rag_travel`.
- "Given my tenure, do I qualify for paid parental leave?" (as `E1003`, 3 months tenure) → `combined`; a good case to compare against query 3, since `E1003` is *below* the 6-month eligibility threshold in `hr/parental_leave_policy.md` -- the answer should reflect that, not just restate the general policy.
- "What is the status of ticket INC0010001?" → `servicenow_troubleshoot`, calls `get_ticket_status` (after you've created at least one ticket via query 5's flow).

## Personal documents (`data/raw_docs/personal/`)

If you've dropped your own PDFs into `data/raw_docs/personal/` (bills,
insurance policies, e-tickets -- see that folder's README), they're ingested
into a `personal` category, separate from the synthetic company docs. Try:

- "What is my current internet/phone bill amount and due date?" → `rag_personal`
- "What is my insurance policy number?" → `rag_personal`
- "What is the PNR for my flight ticket?" → `rag_personal`

These are a good stress test for the pipeline precisely because they're
*real* documents, not written for this project: bills and insurance PDFs
extract as dense, table-heavy text (not the clean paragraphs the sample HR
docs have), and scanned/image-based e-tickets go through the OCR fallback
path in `ingestion/loader.py` rather than clean text extraction. Compare
`ENABLE_HYBRID_SEARCH=true` vs `false` on a query like the PNR one -- exact
keyword matching (PostgreSQL FTS) can miss OCR'd text that vector search
still finds semantically, which is exactly the kind of case hybrid search
and RRF fusion exist for. See `docs/concepts/fts_vs_bm25.md`.

## Comparing configurations

For any of the above, flip a flag in `.env` (e.g. `ENABLE_RERANKING=false`,
`ENABLE_HYBRID_SEARCH=false`, `ENABLE_HYDE=true`), restart the backend, and
resend the same query. The debug trace (visible in the Streamlit expander,
or the `trace` field of the API response) will show you exactly which
pipeline stages ran, what they returned, and how that changed the final
context and answer.
