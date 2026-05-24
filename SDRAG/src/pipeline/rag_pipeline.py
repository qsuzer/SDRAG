import logging
from typing import List, Optional

from flashrag.prompt import PromptTemplate
from .base_pipeline import BaseActivePipeline, flatten_retrieval_results

logger = logging.getLogger(__name__)

class RAGPipeline(BaseActivePipeline):
    """
    Standard Retrieval-Augmented Generation (RAG) Pipeline.
    
    Flow:
    1. Retrieve documents based on the question.
    2. Construct a prompt with the question and retrieved documents.
    3. Generate an answer using the LLM.
    """
    
    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        super().__init__(config, prompt_template, retriever, generator)
        
        # Default RAG prompt if none provided
        if "rag" not in self.prompts:
            self.prompts["rag"] = PromptTemplate(
                config,
                system_prompt="You are a helpful assistant. Answer the question based on the provided context.",
                user_prompt="Context:\n{reference}\n\nQuestion: {question}\nAnswer:"
            )

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        """
        Run standard RAG pipeline on dataset.
        """
        questions = dataset.question
        
        # 1. Retrieval
        logger.info("Retrieving documents...")
        retrieval_results = self.retriever.batch_search(questions)
        dataset.update_output("retrieval_result", retrieval_results)
        
        # 2. Generation
        logger.info("Generating answers...")
        
        # Prepare inputs for generation
        input_prompts = []
        for question, results in zip(questions, retrieval_results):
            # Format context from retrieval results
            context = "\n\n".join([
                f"Title: {doc.get('title', '')}\nText: {doc.get('contents', '')}" 
                for doc in results
            ])
            
            prompt = self.prompts["rag"].get_string(
                question=question,
                reference=context
            )
            input_prompts.append(prompt)
            
        # Generate
        responses = self.generator.generate(input_prompts)
        dataset.update_output("pred", responses)
        
        flatten_retrieval_results(dataset)
        return self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
