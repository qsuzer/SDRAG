import logging
from typing import List, Tuple, Dict, Set, Optional

from flashrag.prompt import PromptTemplate
from flashrag.utils import get_generator, get_retriever
from .base_pipeline import BaseActivePipeline, flatten_retrieval_results

logger = logging.getLogger(__name__)

# --- IRCoT Prompts ---

IRCOT_SYSTEM_PROMPT = """You serve as an intelligent assistant, adept at facilitating users through complex, multi-hop reasoning across multiple documents. This task is illustrated through demonstrations, each consisting of a document set paired with a relevant question and its multi-hop reasoning thoughts. Your task is to generate one thought for current step, DON'T generate the whole thoughts at once! If you reach what you believe to be the final step, start with "So the answer is:"."""

IRCOT_EXAMPLE = """Wikipedia Title: Kurram Garhi
    Kurram Garhi is a small village located near the city of Bannu, which is the part of Khyber Pakhtunkhwa province of Pakistan. Its population is approximately 35000. Barren hills are near this village. This village is on the border of Kurram Agency. Other nearby villages are Peppal, Surwangi and Amandi Kala.

    Wikipedia Title: 2001–02 UEFA Champions League second group stage
    Eight winners and eight runners- up from the first group stage were drawn into four groups of four teams, each containing two group winners and two runners- up. Teams from the same country or from the same first round group could not be drawn together. The top two teams in each group advanced to the quarter- finals.

    Wikipedia Title: Satellite tournament
    A satellite tournament is either a minor tournament or event on a competitive sporting tour or one of a group of such tournaments that form a series played in the same country or region.

    Wikipedia Title: Trojkrsti
    Trojkrsti is a village in Municipality of Prilep, Republic of Macedonia.

    Wikipedia Title: Telephone numbers in Ascension Island
    Country Code:+ 247< br> International Call Prefix: 00 Ascension Island does not share the same country code( +290) with the rest of St Helena.

    Question: Are both Kurram Garhi and Trojkrsti located in the same country?
    Thought: Kurram Garhi is located in the country of Pakistan. Trojkrsti is located in the country of Republic of Macedonia. Thus, they are not in the same country. So the answer is: no.

    """

IRCOT_USER_PROMPT = """{reference}Question: {question}
    {previous_thoughts}Thought:"""


class IRCoTPipeline(BaseActivePipeline):
    """
    Interleaving Retrieval with Chain-of-Thought Reasoning (IRCoT) Pipeline.
    
    IRCoT iteratively:
    1. Generates a reasoning thought based on current retrieved documents
    2. Uses the new thought to retrieve additional relevant documents
    3. Repeats until reaching a conclusion or max iterations
    
    This implementation follows the BaseActivePipeline pattern:
    - No batch pre-decomposition (IRCoT generates thoughts iteratively)
    - Serial processing of questions
    
    Reference: https://arxiv.org/abs/2212.10509
    """
    
    def __init__(self, config, prompt_template=None, retriever=None, generator=None, max_iterations=3):
        super().__init__(config, prompt_template, retriever, generator)
        self.max_iterations = max_iterations
        
        # Initialize IRCoT specific prompt
        full_system_prompt = f"{IRCOT_SYSTEM_PROMPT}\n\n{IRCOT_EXAMPLE}"
        self.prompts["ircot"] = PromptTemplate(
            config,
            system_prompt=full_system_prompt,
            user_prompt=IRCOT_USER_PROMPT
        )
        
        # Reference template for formatting documents
        self.reference_template = "Wikipedia Title: {title}\n{text}\n\n"
    
    def run(self, dataset, do_eval=True, pred_process_fun=None):
        """
        Run IRCoT pipeline on dataset.
        No batch pre-processing - IRCoT is fully iterative.
        """
        questions = dataset.question
        
        all_preds = []
        all_intermediate_steps = []
        
        for i, question in enumerate(questions):
            logger.info(f"IRCoT processing question {i + 1}/{len(questions)}: {question}")
            try:
                final_answer, intermediate_steps = self._process_single_question(question)
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
        
        flatten_retrieval_results(dataset)
        return self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
    
    def _process_single_question(self, question: str) -> Tuple[str, Dict]:
        """
        Process a single question using IRCoT's iterative retrieve-then-reason approach.
        """
        intermediate_steps = {
            "thoughts": [],
            "retrieval_results_per_iter": [],
            "all_doc_ids": set(),
            "doc2score": {},
            "id2doc": {}
        }
        
        # Initial retrieval based on the question
        logger.info("  Initial retrieval...")
        retrieval_results, scores = self.retriever.search(question, return_score=True)
        
        # Initialize document tracking
        for doc_item, score in zip(retrieval_results, scores):
            doc_id = doc_item.get('id', id(doc_item))  # Use hash if no id field
            intermediate_steps["id2doc"][doc_id] = doc_item
            intermediate_steps["doc2score"][doc_id] = score
            intermediate_steps["all_doc_ids"].add(doc_id)
        
        intermediate_steps["retrieval_results_per_iter"].append(retrieval_results)
        
        # Iterative reasoning loop
        for iter_num in range(self.max_iterations):
            logger.info(f"  Iteration {iter_num + 1}/{self.max_iterations}")
            
            # Format retrieved documents for prompt
            reference_text = self._format_retrieval_results(retrieval_results)
            
            # Format previous thoughts
            previous_thoughts = ""
            if intermediate_steps["thoughts"]:
                previous_thoughts = "\n".join(
                    f"Thought {i+1}: {thought}" 
                    for i, thought in enumerate(intermediate_steps["thoughts"])
                ) + "\n"
            
            # Generate prompt
            prompt_input = self.prompts["ircot"].get_string(
                reference=reference_text,
                question=question,
                previous_thoughts=previous_thoughts
            )
            
            # Generate new thought
            new_thought = self.generator.generate([prompt_input])[0].strip()
            intermediate_steps["thoughts"].append(new_thought)
            
            logger.info(f"    Generated thought: {new_thought[:100]}...")
            
            # Check for termination
            if "So the answer is:" in new_thought:
                logger.info("    Reached final answer")
                final_answer = self._extract_final_answer(new_thought)
                return final_answer, intermediate_steps
            
            # If not final iteration, retrieve more documents based on new thought
            if iter_num < self.max_iterations - 1:
                logger.info("    Retrieving based on new thought...")
                new_retrieval_results, new_scores = self.retriever.search(new_thought, return_score=True)
                
                # Update document tracking (merge with existing documents)
                for doc_item, score in zip(new_retrieval_results, new_scores):
                    doc_id = doc_item.get('id', id(doc_item))
                    intermediate_steps["id2doc"][doc_id] = doc_item
                    intermediate_steps["all_doc_ids"].add(doc_id)
                    
                    # Keep the highest score for each document
                    if doc_id in intermediate_steps["doc2score"]:
                        intermediate_steps["doc2score"][doc_id] = max(
                            intermediate_steps["doc2score"][doc_id], score
                        )
                    else:
                        intermediate_steps["doc2score"][doc_id] = score
                
                # Re-rank all documents by score for next iteration
                sorted_doc_ids = sorted(
                    intermediate_steps["doc2score"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                retrieval_results = [
                    intermediate_steps["id2doc"][doc_id] 
                    for doc_id, _ in sorted_doc_ids
                ]
                
                intermediate_steps["retrieval_results_per_iter"].append(retrieval_results)
        
        # If max iterations reached without conclusion
        logger.warning("    Max iterations reached without 'So the answer is:'")
        
        # Force generate an answer
        all_thoughts = "\n".join(intermediate_steps["thoughts"])
        extract_prompt = (
            f"Question: {question}\n"
            f"Reasoning process:\n{all_thoughts}\n\n"
            "Based on the reasoning above, provide the final answer to the question. "
            "Only give me the answer and do not output any other words."
        )
        try:
            final_answer = self.generator.generate([extract_prompt])[0].strip()
        except Exception as e:
            logger.error(f"Failed to force generate answer: {e}")
            final_answer = " ".join(intermediate_steps["thoughts"])

        return final_answer, intermediate_steps
    
    def _format_retrieval_results(self, retrieval_results: List) -> str:
        """Format retrieval results into reference text for the prompt."""
        if not retrieval_results:
            return ""
        
        formatted_text = ""
        for doc in retrieval_results:
            if isinstance(doc, dict):
                title = doc.get('title', 'Unknown')
                text = doc.get('contents', doc.get('content', doc.get('text', str(doc))))
            else:
                title = "Document"
                text = str(doc)
            
            formatted_text += self.reference_template.format(title=title, text=text)
        
        return formatted_text
    
    def _extract_final_answer(self, thought: str) -> str:
        """Extract the final answer from a thought containing 'So the answer is:'."""
        if "So the answer is:" in thought:
            answer = thought.split("So the answer is:")[-1].strip()
            # Remove trailing punctuation
            answer = answer.rstrip('.')
            return answer
        return thought
