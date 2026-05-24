# Prompts for SDRAG Pipeline

DECOMPOSE_PROMPT = """You are an expert at breaking down complex questions into simpler sub-questions.
For the given complex question, decompose it into no more than 3 sub-questions that:
1. Are simpler than the original question
2. Build upon each other in a logical sequence
3. When combined, help answer the original question completely

Format your response as a numbered list of questions only."""

DEPENDENCY_ANALYSIS_PROMPT = """You are an expert at analyzing logical dependencies between questions.
Given a list of sub-questions, analyze the DIRECT dependency relationships between them.
A question depends on another ONLY if it cannot be answered without that specific information.

Sub-questions:
{sub_questions}

For each sub-question, list ONLY the question numbers it DIRECTLY depends on (if any).
Format your response as:
Question 1: depends on []
Question 2: depends on [1]
...
Use empty brackets [] if a question has no dependencies."""

PRUNING_PROMPT = """You are an expert at identifying unnecessary sub-questions.
Given the original question and a leaf sub-question (no other questions depend on it), determine if this sub-question should be kept or pruned.
Prune the sub-question if it involves additional factual queries not directly related to the original question.

Original question: {original_question}
Sub-question to evaluate: {sub_question}
Answer only "KEEP" or "PRUNE"."""

REWRITE_PROMPT = """Rewrite the given sub-question to make it more explicit and easier to retrieve relevant information.
Replace all pronouns with specific names or entities.
Use answers from previous sub-questions to provide additional context.

Previous sub-questions and answers:
{previous_qa}

Sub-question to rewrite: {sub_question}
Rewritten sub-question:"""

SUB_ANSWER_PROMPT = """Generate an answer for the sub-question based on the retrieved documents.
Be concise, factual, and focus only on information relevant to answering the question.

Sub-question: {sub_question}
Retrieved documents:
{retrieval_results}
Provide a concise answer based on the retrieved information."""

COMPOSE_PROMPT = """You are a helpful and precise assistant tasked with answering questions based on retrieved information.

Original Question: {original_question}

=== REASONING GRAPH OVERVIEW ===
{graph_overview}

=== DETAILED INFORMATION (Organized by Reasoning Flow) ===
{structured_information}

Based on this structured reasoning chain, provide a comprehensive answer to the original question.
Only give me the answer and do not output any other words."""
