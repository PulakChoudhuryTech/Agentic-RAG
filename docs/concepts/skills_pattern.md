# Skills pattern (custom, Gemini-compatible)

**Where it's implemented:** [`backend/app/skills/`](../../backend/app/skills/)

## The problem it solves

Before this pattern, everything the supervisor needed to know about a
domain (its routing description, its category, its keyword-matching rules,
any domain-specific answer guidance) was smeared across four separate
Python files: `llm/prompts.py`'s hardcoded `SUPERVISOR_ROUTING_PROMPT`
string, `agents/supervisor.py`'s `VALID_ROUTES` set, `agents/rag_agent.py`'s
`ROUTE_TO_CATEGORY` dict, and `rag/metadata_router.py`'s `CATEGORY_KEYWORDS`
dict. Adding or tuning a domain meant editing all four in sync -- easy to
get wrong, and it means a non-engineer (an HR or IT SME who wants to fix a
routing keyword or add a caveat to answers) has to touch Python.

## The pattern

This borrows the *shape* of Anthropic's Agent Skills (see
[Anthropic's docs](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview)):
a filesystem-based capability packaged as a markdown file with YAML
frontmatter, loaded in layers of increasing cost:

| Layer | Source | Loaded | Consumed by |
|---|---|---|---|
| `name` + `description` | frontmatter | always, to build the routing prompt | `agents/supervisor.py` |
| `category` + `keywords` | frontmatter | once per process, cached | `agents/rag_agent.py`, `rag/metadata_router.py` |
| body | markdown after frontmatter | only for the triggered skill | `agents/rag_agent.py` -> `rag/pipeline.py`'s `RAG_ANSWER_PROMPT` |

It is **not** Anthropic's actual Agent Skills runtime -- that only exists
inside Claude's API/Code-execution container, and this project is 100%
Gemini (`google-genai`, `langchain-google-genai`). What's replicated here is
the pattern (cheap always-on metadata, expensive content loaded only when
triggered, filesystem-based and human-editable), implemented from scratch
in [`skills/registry.py`](../../backend/app/skills/registry.py).

## How it wires into the graph

1. `agents/supervisor.py` builds `SUPERVISOR_ROUTING_PROMPT`'s route list and
   `VALID_ROUTES` from `load_skills()` instead of a hardcoded string/set.
2. `rag/metadata_router.py` builds `CATEGORY_KEYWORDS` from each skill's
   `keywords`, sorted by `match_priority` (lower = checked first --
   `rag_personal` sets `match_priority: 0` for the same reason the original
   hardcoded dict put `personal` first: its vocabulary is specific enough
   to rarely collide with the policy categories, and should win when it
   does).
3. `agents/rag_agent.py` and `agents/it_agent.py` build `ROUTE_TO_CATEGORY`
   from each skill's `category`, and pass the matched skill's `body` into
   `run_rag_pipeline()` as `skill_guidance`, which lands in
   `RAG_ANSWER_PROMPT` only for that request.

## Try it yourself

Open `backend/app/skills/rag_it/SKILL.md` and edit the "Answer guidance"
section, then ask an IT question through the UI -- the new guidance shows
up in that turn's answer with no other code changes. Add a brand-new
`backend/app/skills/<name>/SKILL.md` with a `name`/`description` and it
becomes a selectable route the very next time the supervisor prompt is
built (add the corresponding LangGraph node/edge in `agents/graph.py` to
actually handle it).
