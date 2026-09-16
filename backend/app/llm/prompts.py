"""
All prompt templates in one file, so you can see exactly what's sent to
Gemini for each pipeline stage without hunting through the codebase.
"""

QUERY_REWRITE_PROMPT = """You rewrite user questions into a single, clearer \
search query for a document retrieval system. Fix ambiguity, expand \
pronouns/abbreviations, and keep it concise. Return ONLY the rewritten \
query, nothing else.

Original question: {query}

Rewritten query:"""

QUERY_EXPANSION_PROMPT = """Generate {n} alternative phrasings of the \
following search query, each capturing a different way someone might ask \
about the same underlying need. Return ONLY the {n} alternatives, one per \
line, no numbering, no extra commentary.

Query: {query}

Alternatives:"""

HYDE_PROMPT = """Write a short, plausible passage (3-5 sentences) that \
would directly and completely answer the following question, as if it were \
an excerpt from a company policy document. This is a hypothetical document \
used only to improve search -- it does not need to be factually accurate, \
just written in the style of a real policy answer.

Question: {query}

Hypothetical passage:"""

RAG_ANSWER_PROMPT = """You are an internal Employee AI Assistant. Answer \
the employee's question using ONLY the context below. If the context does \
not contain enough information to answer, say so plainly rather than \
guessing. Cite sources using the bracketed [Source N] labels already \
present in the context when you use information from them.
{skill_guidance}
Context:
{context}

Question: {query}

Answer:"""

SUPERVISOR_ROUTING_PROMPT = """You are the routing supervisor for an internal \
Employee AI Assistant. Read the employee's message and decide which single \
route should handle it. Respond with EXACTLY ONE of these route labels, \
nothing else:

{route_descriptions}

{intent_hint}
Employee's message: {query}

Route:"""

COMBINE_PROMPT = """Answer the employee's question by combining their \
personal Workday data with the relevant company policy below. Be specific: \
apply the policy rules to their actual profile data rather than restating \
the policy generically.

Employee's question: {query}

Employee's Workday data:
{workday_data}

Relevant policy (from company documents):
{policy_context}

Answer:"""

TROUBLESHOOTING_RESOLVED_PROMPT = """An employee reported this issue and \
was given troubleshooting steps from IT documentation. Based on their most \
recent message, decide whether the issue is now RESOLVED or still \
UNRESOLVED. Respond with exactly one word: RESOLVED or UNRESOLVED.

Original issue: {issue}
Troubleshooting steps given: {steps}
Employee's latest message: {latest_message}

Answer (one word):"""
