"""LLM prompt templates for each stage of the STORM pipeline.

Each constant is a string template with ``{placeholders}`` that get
filled in by the pipeline.  These replace the DSPy Signature classes
from the original STORM codebase.
"""

# ---------------------------------------------------------------------------
# Anti-reasoning suffix — appended to every prompt to prevent models (e.g.
# Gemma) from dumping their chain-of-thought into the output.
# ---------------------------------------------------------------------------

_NO_REASONING = """

IMPORTANT: Output ONLY your final answer. Do NOT include any internal \
reasoning, planning, drafting notes, bullet-point outlines, self-correction \
comments, or "thinking out loud" text. No meta-commentary about what you \
are about to write. Just the finished content, nothing else."""

# ---------------------------------------------------------------------------
# Stage 1 — Find related topics
# ---------------------------------------------------------------------------

FIND_RELATED_TOPICS = """\
I'm writing a comprehensive, Wikipedia-quality article about the following topic:

Topic: {topic}

Please identify 3-5 closely related topics (they can be real Wikipedia articles \
or well-known subjects) that would help me understand:
1. What aspects are commonly covered in articles about this topic
2. What structure and depth is typical
3. What perspectives might be relevant

For each related topic, provide the topic name and a brief note about why \
it is relevant.  Format your response as a numbered list:

1. <Topic Name> — <why it's relevant>
2. ...
""" + _NO_REASONING

# ---------------------------------------------------------------------------
# Stage 2 — Generate perspectives / personas
# ---------------------------------------------------------------------------

GENERATE_PERSONAS = """\
You are assembling a team of {num_perspectives} editors to collaboratively \
research and write a comprehensive article on the following topic:

Topic: {topic}

Related topics for context:
{related_topics}

Each editor should represent a **different perspective, expertise, or angle** \
related to this topic.  For each editor, provide:
1. A short role title (e.g. "Historian", "Industry Analyst")
2. A description of what they will focus on and what questions they will explore

Format your response as a numbered list:
1. <Role Title>: <Description of focus and questions>
2. ...
""" + _NO_REASONING

# ---------------------------------------------------------------------------
# Stage 3a — Writer asks a question (with persona)
# ---------------------------------------------------------------------------

ASK_QUESTION_WITH_PERSONA = """\
You are a writer researching the topic below. You are approaching this topic \
from a specific perspective.

Topic: {topic}
Your perspective: {persona}

Previous conversation with the expert:
{conversation_history}

Based on your perspective, ask your next question to deepen your understanding \
of the topic. Ask specific, focused questions that explore your unique angle. \
Avoid repeating questions that have already been asked.

If you believe you have gathered enough information from this perspective, \
respond with exactly: "Thank you so much for your help!"

Your question:""" + _NO_REASONING

ASK_QUESTION_INITIAL = """\
You are a writer researching the topic below. You are approaching this topic \
from a specific perspective.

Topic: {topic}
Your perspective: {persona}

This is the start of your conversation with an expert. Ask your first question \
to begin exploring the topic from your unique angle.

Your question:""" + _NO_REASONING

# ---------------------------------------------------------------------------
# Stage 3b — Expert answers (grounded in search results)
# ---------------------------------------------------------------------------

EXPERT_ANSWER = """\
You are a knowledgeable expert answering questions about the following topic. \
Ground your answers in the provided search results. Include inline citations \
using [1], [2], etc. to reference specific sources.

Topic: {topic}
Question: {question}

Search results:
{search_results}

Provide a thorough, well-sourced answer. Cite specific sources using [N] notation \
where N corresponds to the source number above. If the search results don't contain \
relevant information, say so honestly.

Your answer:""" + _NO_REASONING

GENERATE_SEARCH_QUERIES = """\
You are a research assistant. Given the topic and question below, generate \
{num_queries} specific search queries that would help find relevant information \
to answer the question.

Topic: {topic}
Question: {question}

Return each query on a separate line, with no numbering or bullet points:""" + _NO_REASONING

# ---------------------------------------------------------------------------
# Stage 4 — Generate outline
# ---------------------------------------------------------------------------

GENERATE_OUTLINE = """\
You are writing a comprehensive, Wikipedia-quality article about the following topic. \
Based on the research conversations and collected information below, generate a \
detailed hierarchical outline for the article.

Topic: {topic}

Research information collected:
{collected_info}

Create a well-organized outline using Markdown heading syntax (##, ###, ####). \
The outline MUST:
1. Start with a "Definition and Overview" or "Introduction" section
2. Cover ALL major aspects discovered during research
3. Be organized logically (chronological, thematic, or by importance)
4. Include DEEP subsections (### and ####) under each main section for thorough coverage
5. End with a "Legacy" or "Impact" or "See Also" section
6. Have at least 5-8 top-level sections (## headings)
7. Have at least 15-25 total headings across all levels (##, ###, ####)
8. Each ## section should have 2-4 subsections (### headings)

Output ONLY the Markdown headings. Do not include any content or explanations, \
just the section headings. Do not start with preamble like "Here is the outline" — \
start directly with the first ## heading.

Example of the expected depth and format:
## Introduction
### Definition and Overview
### Core Concepts
#### Concept A
#### Concept B
## History and Development
### Early Research
### Key Breakthroughs
### Evolution of the Approach
## Technical Architecture
### Component Design
#### Sub-Component 1
#### Sub-Component 2
### Algorithm Details
#### Selection Mechanism
#### Training Loop
## Applications and Impact
### Domain 1
### Domain 2
## Criticism and Limitations
## Legacy and Influence
## See Also

Now generate the outline for {topic}:""" + _NO_REASONING

# ---------------------------------------------------------------------------
# Stage 5 — Generate article section content
# ---------------------------------------------------------------------------

WRITE_SECTION = """\
Write the body content for the section "{section_name}" of a comprehensive article about "{topic}" based on the source material provided below.

Full article outline:
{outline}

Content of previously written sections (build on top of these, do not repeat them):
{previous_sections}

Relevant source material:
{relevant_sources}

Write the content for this section. Requirements:
1. Write in an encyclopedic, neutral, informative tone
2. Include inline citations using [N] notation referencing the sources above (if sources are provided. If no sources are provided, write the section using your own pre-trained knowledge and omit citations).
3. Be thorough but concise — aim for 2-4 paragraphs
4. Use specific facts, dates, and details from the sources (or from your pre-trained knowledge if no sources are provided)
5. Do not include the section heading — only the body text
6. Do not repeat facts, definitions, or explanations that have already been covered in previous sections. Refer back to them if needed, but focus on the specific aspects of the current section.
7. Output ONLY the final polished paragraphs — no drafts, no planning notes
8. CRITICAL: Do NOT write any greetings, conversational intro, preamble, or acknowledgment of the role (such as "I accept this role", "I am ready", "Please provide a topic", or similar). If no relevant sources are listed above, write the section content directly using your pre-trained knowledge. Start writing the first sentence of the section text immediately.

Section content (start writing directly):""" + _NO_REASONING

# ---------------------------------------------------------------------------
# Stage 6 — Polish article (write lead section + cleanup)
# ---------------------------------------------------------------------------

WRITE_LEAD_SECTION = """\
Below is a comprehensive article about "{topic}". Your task is to write a \
lead summary section for this article.

--- BEGIN ARTICLE ---
{article_text}
--- END ARTICLE ---

Write a concise lead section (2-3 paragraphs) that summarizes the article above. \
The lead section should:
1. Introduce "{topic}" clearly in the first sentence
2. Highlight the most important aspects covered in the article
3. Give the reader a high-level overview of the subject
4. Be written in an encyclopedic, neutral tone
5. Not include citations — this is a summary only

Start writing the lead section paragraphs directly:""" + _NO_REASONING

POLISH_ARTICLE = """\
You are a skilled editor polishing a draft article for publication.

Topic: {topic}

Draft article:
{article_text}

Please polish this article by:
1. Removing any duplicate or redundant information across sections
2. Ensuring smooth transitions between sections
3. Fixing any grammatical or stylistic issues
4. Ensuring consistent citation formatting [N]
5. Keeping all factual content and citations intact

Output the polished article (complete text with all sections):""" + _NO_REASONING
