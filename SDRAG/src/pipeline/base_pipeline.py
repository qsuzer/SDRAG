import re
import json
import logging
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict, deque

from flashrag.pipeline import BasicPipeline
from flashrag.utils import get_generator, get_retriever
from flashrag.prompt import PromptTemplate
from .prompts import (
    DECOMPOSE_PROMPT,
    DEPENDENCY_ANALYSIS_PROMPT,
    PRUNING_PROMPT,
    REWRITE_PROMPT,
    SUB_ANSWER_PROMPT,
    COMPOSE_PROMPT
)

logger = logging.getLogger(__name__)

def _parse_decomposed_questions(decomposed_text: str) -> List[str]:
    """Parses numbered list of questions from text into a list of strings."""
    lines = decomposed_text.strip().split('\n')
    questions = [re.sub(r'^\d+[\.\)\-]?\s*', '', line.strip()) for line in lines if line.strip()]
    return questions

def _parse_dependencies(dependency_text: str) -> dict:
    """Parses dependency JSON from text into a dictionary."""
    dependencies = {}
    try:
        # Try to find JSON block within markdown code fences
        match = re.search(r"```json\s*(.*?)\s*```", dependency_text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # Try to find the first { and last }
            match = re.search(r"(\{.*\})", dependency_text, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                json_str = dependency_text
        
        data = json.loads(json_str)
        
        # Convert 1-based string keys/values to 0-based integers
        for k, v in data.items():
            if k.isdigit():
                q_idx = int(k) - 1
                if isinstance(v, list):
                    deps = []
                    for d in v:
                        if isinstance(d, int):
                            deps.append(d - 1)
                        elif isinstance(d, str) and d.isdigit():
                            deps.append(int(d) - 1)
                    dependencies[q_idx] = [d for d in deps if d >= 0]
                    
    except Exception as e:
        logger.error(f"Error parsing dependencies: {e}")
        logger.error(f"Raw text: {dependency_text}")
        
    return dependencies

class BaseActivePipeline(BasicPipeline):
    """Base class for active pipelines with shared initialization and a mixed execution model."""

    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        super().__init__(config, prompt_template)
        self.generator = get_generator(config) if generator is None else generator
        self.retriever = get_retriever(config) if retriever is None else retriever
        # Initialize all prompt templates
        self.prompts = {
            "decompose": PromptTemplate(config, system_prompt=DECOMPOSE_PROMPT,
                                        user_prompt="Complex Question: {question}\nSub-questions:"),
            "rewrite": PromptTemplate(config, system_prompt=REWRITE_PROMPT,
                                      user_prompt="Previous sub-questions and answers:\n{previous_qa}\n\nSub-question to rewrite: {sub_question}"),
            "sub_answer": PromptTemplate(config, system_prompt=SUB_ANSWER_PROMPT,
                                         user_prompt="Sub-question: {sub_question}\n\nRetrieved documents:\n{retrieval_results}"),
            "compose": PromptTemplate(config, system_prompt=COMPOSE_PROMPT,
                                      user_prompt="Original Question: {original_question}\nRetrieved Information:\n{all_retrieval_results}"),
            "dependency": PromptTemplate(config, system_prompt=DEPENDENCY_ANALYSIS_PROMPT,
                                         user_prompt="Sub-questions:\n{sub_questions}"),
            "pruning": PromptTemplate(config, system_prompt=PRUNING_PROMPT,
                                      user_prompt="Original question: {original_question}\nSub-question to evaluate: {sub_question}")
        }

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        """
        Mixed execution model: Batch pre-processing + Serial main processing.
        """
        questions = dataset.question

        # --- Batch Pre-processing Step 1: Decompose ---
        logger.info("Step 1: Decomposing all questions in a batch...")
        decompose_inputs = [self.prompts["decompose"].get_string(question=q) for q in questions]
        decomposed_texts = self.generator.generate(decompose_inputs)
        all_sub_questions = [_parse_decomposed_questions(text) for text in decomposed_texts]

        # --- Serial Main Processing Loop ---
        all_preds = []
        all_intermediate_steps = []
        for i, question in enumerate(questions):
            logger.info(f"Processing question {i + 1}/{len(questions)}: {question}")
            sub_questions = all_sub_questions[i]
            try:
                final_answer, intermediate_steps = self._process_single_question(question, sub_questions)
                all_preds.append(final_answer)
                all_intermediate_steps.append(intermediate_steps)
            except Exception as e:
                logger.error(f"Failed to process question '{question}': {e}", exc_info=True)
                all_preds.append("[ERROR: Processing failed]")
                all_intermediate_steps.append({})

        # Update dataset with results
        dataset.update_output("pred", all_preds)
        # Update intermediate steps
        if all_intermediate_steps:
            keys = set(k for d in all_intermediate_steps for k in d)
            for key in keys:
                values = [d.get(key) for d in all_intermediate_steps]
                dataset.update_output(key, values)
        
        return self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

    def _process_single_question(self, question: str, sub_questions: List[str]) -> Tuple[str, Dict]:
        raise NotImplementedError

class BaseGraphPipeline(BaseActivePipeline):
    """
    Base class: Batch Decomposition + Batch Dependency Analysis + Serial Main Loop
    """

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        questions = dataset.question

        # --- Batch Pre-processing Step 1: Decompose ---
        logger.info("Step 1: Decomposing all questions in a batch...")
        decompose_inputs = [self.prompts["decompose"].get_string(question=q) for q in questions]
        decomposed_texts = self.generator.generate(decompose_inputs)
        all_sub_questions = [_parse_decomposed_questions(text) for text in decomposed_texts]

        # --- Batch Pre-processing Step 2: Dependency Analysis ---
        logger.info("Step 2: Analyzing dependencies for all sub-questions in a batch...")
        dep_inputs = [
            self.prompts["dependency"].get_string(sub_questions="\n".join(f"{i + 1}. {q}" for i, q in enumerate(sqs)))
            if sqs else "" for sqs in all_sub_questions
        ]
        valid_dep_inputs = [inp for inp in dep_inputs if inp]
        dep_texts = self.generator.generate(valid_dep_inputs) if valid_dep_inputs else []

        all_dependencies = []
        dep_texts_iter = iter(dep_texts)
        for sqs in all_sub_questions:
            all_dependencies.append(_parse_dependencies(next(dep_texts_iter)) if sqs else {})

        # --- Serial Main Loop ---
        all_preds, all_intermediate_steps = [], []
        for i, question in enumerate(questions):
            logger.info(f"Processing question {i + 1}/{len(questions)}: {question}")
            try:
                final_answer, intermediate_steps = self._process_single_question(
                    question=question,
                    sub_questions=all_sub_questions[i],
                    dependencies_dict=all_dependencies[i]
                )
                all_preds.append(final_answer)
                all_intermediate_steps.append(intermediate_steps)
            except Exception as e:
                logger.error(f"Failed to process question '{question}': {e}", exc_info=True)
                all_preds.append("[ERROR: Processing failed]")
                all_intermediate_steps.append({})

        dataset.update_output("pred", all_preds)
        if all_intermediate_steps:
            keys = set(k for d in all_intermediate_steps for k in d)
            for key in keys:
                dataset.update_output(key, [d.get(key) for d in all_intermediate_steps])

        return self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

    def _process_single_question(self, question: str, sub_questions: List[str], dependencies_dict: Dict) -> Tuple[
        str, Dict]:
        raise NotImplementedError
