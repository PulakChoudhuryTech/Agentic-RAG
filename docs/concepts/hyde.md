# HyDE (Hypothetical Document Embeddings)

**Where it's implemented:** [`backend/app/rag/hyde.py`](../../backend/app/rag/hyde.py)

## The idea

Normally, vector search embeds the user's raw *question* and compares it
against embeddings of document *chunks*. But questions and answers are
written very differently -- a question is often short, informal, and framed
as "what/how/why...", while a policy document reads like a declarative
statement of fact. That mismatch in writing style can put a question and
its own correct answer further apart in embedding space than you'd expect.

HyDE's fix: instead of embedding the question, ask an LLM to first write a
short **hypothetical answer** -- a passage that plausibly (not necessarily
factually) answers the question, in the style of the target documents (in
this project's prompt, "as if it were an excerpt from a company policy
document") -- and embed *that* instead. The idea is that a fake-but-
plausibly-worded answer sits closer, in embedding space, to the real
answer than the bare question does.

## Why it's optional and off by default

HyDE is a genuinely mixed-results technique in practice:

- It adds one extra LLM call (latency + cost) per query.
- The hypothetical passage can drift from the actual question if the LLM
  "hallucinates" a confident-sounding but wrong-shaped answer, which can
  *hurt* retrieval compared to just embedding the question.
- It tends to help most when the corpus's writing style is very different
  from how users phrase questions, and help least (or hurt) when questions
  are already close in phrasing to the source documents -- which is
  somewhat true of this project's fairly plainly-written HR/IT/travel docs.

This is exactly the kind of technique this project exists to let you
measure rather than take on faith: toggle `ENABLE_HYDE=true` in `.env`,
run the same query with it on and off, and compare the `vector_search`
trace step's results (and ultimately the final answer) between the two
runs. The `hyde` trace step shows you the exact hypothetical passage that
was generated and embedded, so you can judge for yourself whether it's
actually closer to the real document language than the original question
was.
