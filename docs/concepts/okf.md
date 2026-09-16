# OKF (Open Knowledge Format) -- layered, not migrated

**Where it's implemented:** [`backend/app/ingestion/loader.py`](../../backend/app/ingestion/loader.py) (`_parse_okf_frontmatter`)

## What OKF is

[Open Knowledge Format](https://cloud.google.com/blog/products/data-analytics/how-the-open-knowledge-format-can-improve-data-sharing)
is Google Cloud's open spec (v0.1, published June 2026) for packaging
knowledge as a directory of markdown files with YAML frontmatter -- nothing
more. The only required field is `type`; everything else (`tags`, `related`,
`stale_after`, `sources`, `resource`, ...) is producer-defined. Relationships
between concepts are just markdown links, which makes the knowledge base a
literal graph you can traverse, distinct from vector similarity.

## Why layered, not migrated

OKF is very new -- its canonical repository already moved once
(`GoogleCloudPlatform/knowledge-catalog` -> `GoogleCloudPlatform/open-knowledge-format`)
in the months after v0.1 shipped. Rather than restructure all of
`data/raw_docs/` around a spec that's still settling, this project makes
OKF frontmatter **opt-in per file**:

- A markdown file with no frontmatter loads exactly as it always has --
  title/effective-date regex extraction, per-section country tagging via
  `tag_country()`, all unchanged.
- A markdown file that *does* start with a `---\n...\n---\n` block gets that
  block parsed for `type`, `tags`, `related` (paths to other files under
  `data/raw_docs/`), and `stale_after`, then stripped before the body is
  used for title/date extraction, chunking, or embedding -- the frontmatter
  never pollutes retrieval content.
- The parsed fields land in the existing `documents.metadata` JSONB column
  (`backend/app/ingestion/ingest.py`) -- no schema migration needed.

## Example

```yaml
---
type: Runbook
tags: [it, vpn, network]
related: [it/password_reset_policy.md]
stale_after: 2027-01-01
---
# VPN Troubleshooting Guide
...
```

Three real documents in this project carry frontmatter today, chosen
because they have a genuine relationship a human would recognize but
vector similarity alone might miss:

- `it/vpn_troubleshooting.md` <-> `it/password_reset_policy.md` (VPN
  authentication prompts are often actually a password/account-lockout
  issue -- see the "Repeated Authentication Prompts" section)
- `hr/parental_leave_policy.md` -> `hr/benefits_overview.md` (leave
  continuation depends on benefits enrollment)

## What this doesn't do yet

The `related` links are captured but nothing currently *traverses* them
during retrieval -- today's pipeline (`rag/pipeline.py`) is vector +
keyword search only. A graph-aware expansion step (pull in a hit's related
documents alongside its parent-chunk expansion) is the natural next step
once more of the corpus carries real `related` links, gated behind a new
`ENABLE_RELATED_DOC_EXPANSION` flag like every other optional stage in this
pipeline.

## Try it yourself

Add a frontmatter block to any other doc in `data/raw_docs/` and re-run
`make ingest`, then check `documents.metadata` for that row in Postgres --
you should see exactly the fields you added and nothing else changed.
