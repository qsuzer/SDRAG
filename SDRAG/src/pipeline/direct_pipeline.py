import logging
from typing import List, Optional

from flashrag.prompt import PromptTemplate
from .base_pipeline import BaseActivePipeline

logger = logging.getLogger(__name__)

class DirectLLMPipeline(BaseActivePipeline):
    """
    Direct LLM Pipeline (Zero-shot / Naive Generation).
    
    Flow:
    1. Construct a prompt with the question (no retrieval).
    2. Generate an answer using the LLM.
    """
    
    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        # Note: Retriever is not used here, but kept for compatibility
        super().__init__(config, prompt_template, retriever, generator)
        
        # Default Direct prompt if none provided
        if "direct" not in self.prompts:
            self.prompts["direct"] = PromptTemplate(
                config,
                system_prompt="You are a helpful assistant. Answer the question directly.",
                user_prompt="Question: {question}\nAnswer:"
            )

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        """
        Run Direct LLM pipeline on dataset.
        """
        questions = dataset.question
        
        # Generation
        logger.info("Generating answers directly (no retrieval)...")
        
        # Prepare inputs for generation
        input_prompts = []
        for question in questions:
            prompt = self.prompts["direct"].get_string(
                question=question
            )
            input_prompts.append(prompt)
            
        # Generate
        responses = self.generator.generate(input_prompts)
        dataset.update_output("pred", responses)
        
        # No retrieval results to flatten
        return self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
