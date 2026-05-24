import re
import json
import logging
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict

from flashrag.pipeline import BasicPipeline
from flashrag.utils import get_generator, get_retriever
from flashrag.prompt import PromptTemplate
from .base_pipeline import flatten_retrieval_results

logger = logging.getLogger(__name__)

# --- Prompts ---
DUALRAG_INFER_PROMPT = """## Task Description
    Your task is to reason based on the retrieved knowledge to answer a question. Adhere strictly to the provided knowledge. Conclude reasoning if: 1. You have the answer. 2. You need more information via retrieval. Reason step-by-step. Do not speculate.

    ## Output Format
    Output in JSON format:
    {{
        "thought": "str, your reasoning thoughts",
        "need_retrieve": true / false
    }}

    ## Example
    **Question**: The Oberoi family is part of a hotel company that has a head office in what city?
    **Known Knowledge**:
    - **Oberoi family**: The Oberoi family is an Indian family that is famous for its involvement in hotels, namely through The Oberoi Group.
    **Reasoning History**:
    1. I need to find out which hotel company the Oberoi family is part of.
    **Output**:
    {{
        "thought": "The Oberoi family is part of The Oberoi Group, which is an Indian hotel company. Next, I need to find out what city The Oberoi Group is headquartered in.",
        "need_retrieve": true
    }}

    ## Current Task
    ### Known Knowledge
    {knowledge}
    ### Question
    {question}
    ### Reasoning History
    {thought}
    ### Output
    """

DUALRAG_NEED_PROMPT = """## Task Description
    Identify additional knowledge needed based on the question, existing knowledge, and reasoning history. Generate retrieval keywords for a dense retriever.
    - **Identify Knowledge**: Focus on key entities (person, location, organization, event, proper noun). Use hint entities if provided.
    - **Generate Keywords**: Keywords should cover needed sub-knowledge. Retain at most two variations for similar meanings. Avoid excessive retrieval. Use "else" entity for ambiguous knowledge.

    ## Output Format
    Output in JSON format:
    {{
        "entities": [
            {{
                "entity": "str, The name of the entity...",
                "keywords": ["str, retrieval query1...", ...]
            }},
            ...
        ]
    }}

    ## Example
    **Question**: After the report by Fortune on October 4, 2023, regarding Sam Bankman-Fried's alleged use of Caroline Ellison as a front at Alameda Research, and the subsequent report by TechCrunch involving Sam Bankman-Fried's alleged motives for committing fraud, is the portrayal of Sam Bankman-Fried's actions by both news sources consistent?
    **Output**:
    {{
        "entities": [
            {{
                "entity": "Sam Bankman-Fried",
                "keywords": [
                    "What's the portrayal of Sam Bankman-Fried's actions in the report by Fortune on October 4, 2023, regarding Sam Bankman-Fried's alleged use of Caroline Ellison as a front at Alameda Research?",
                    "What's the portrayal of Sam Bankman-Fried's actions in the subsequent report by TechCrunch involving Sam Bankman-Fried's alleged motives for committing fraud?"
                ]
            }}
        ]
    }}

    ## Current Task
    ### Known Knowledge
    {knowledge}
    ### Question
    {question}
    ### Reasoning History
    {thought}
    ### Hint Entities (Previously retrieved)
    {known_entity}
    ### Output
    """

DUALRAG_LEARN_PROMPT = """## Task Description
    Read and organize retrieved documents. Summarize information pertinent to the current problem, focusing on multi-hop reasoning connections. Filter irrelevant content.
    ## Note
    - Summarize directly, no interpretation. Preserve original wording and entity names.
    - Extract information relevant to the key entity and query.

    ## Example
    **Question**: Which magazine was started first Arthur's Magazine or First for Women?
    **Retrieval**:
    Key Entity Retrieved: Arthur's Magazine
    Retrieve Queries: When was Arthur's Magazine started?
    Retrieved Documents:
    ##### First Arthur County Courthouse and Jail
    ... (irrelevant content) ...
    ##### Arthur's Magazine
    Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia... merged into "Godey's Lady's Book".
    **Output**:
    Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century.

    ## Current Task
    ### Question
    {question}
    ### Reasoning History
    {thought}
    ### Retrieval Details
    Key Entity Retrieved: {entity}
    Retrieve Queries: {query}
    Retrieved Content:
    {docs}
    ### Output (Summarize relevant info, or "None" if none)
    """

DUALRAG_ANSWER_PROMPT = """## Task Description
    Based on the retrieved knowledge and your previous reasoning, provide a final answer to the question.

    CRITICAL CONSTRAINTS:
    1. Answer as concisely as possible.
    2. Provide the answer only and do not output other information.
    4. Do NOT write "The answer is..." or complete sentences.

    ## Retrieved Knowledge
    {knowledge}
    ## Question
    {question}
    ## Reasoning Steps
    {thought}
    ## Final Answer
    """

# --- Helper Functions ---

def _parse_dualrag_infer(response: str) -> Tuple[str, bool]:
    """Parse DualRAG Infer output: (thought, need_retrieve)"""
    response = response.strip()
    need_retrieve = True
    thought = response
    
    try:
        # Try to parse JSON
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE)
        if match:
            json_str = match.group(1)
        else:
            start = response.find("{")
            end = response.rfind("}")
            if start != -1 and end != -1:
                json_str = response[start : end + 1]
            else:
                json_str = response
        
        data = json.loads(json_str)
        if isinstance(data, dict):
            thought = data.get("thought", response)
            need_retrieve = data.get("need_retrieve", True)
            return thought, need_retrieve
            
    except:
        pass

    # Fallback parsing
    if "Need Retrieve: False" in response or "need retrieve: false" in response.lower():
        need_retrieve = False
        thought = re.sub(r'Need Retrieve:\s*False', '', response, flags=re.IGNORECASE).strip()
    elif "Need Retrieve: True" in response or "need retrieve: true" in response.lower():
        need_retrieve = True
        thought = re.sub(r'Need Retrieve:\s*True', '', response, flags=re.IGNORECASE).strip()
    elif "So the answer is" in response:
        need_retrieve = False
    
    return thought, need_retrieve

def _parse_dualrag_need(response: str) -> Dict[str, List[str]]:
    """Parse DualRAG Need output JSON string"""
    entity2keywords = {}
    try:
        cleaned_response = response.strip()
        match = re.search(r"```(?:json)?\s*(.*?)```", cleaned_response, re.DOTALL | re.IGNORECASE)
        if match:
            cleaned_response = match.group(1).strip()
        
        start_idx = cleaned_response.find("{")
        end_idx = cleaned_response.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            cleaned_response = cleaned_response[start_idx : end_idx + 1]

        data = json.loads(cleaned_response)

        if isinstance(data, dict) and "entities" in data and isinstance(data["entities"], list):
            for item in data["entities"]:
                if isinstance(item, dict) and "entity" in item and "keywords" in item:
                    entity = item["entity"].strip()
                    keywords = [kw.strip() for kw in item["keywords"] if isinstance(kw, str) and kw.strip()]
                    if entity and keywords:
                        if entity in entity2keywords:
                             entity2keywords[entity].extend(k for k in keywords if k not in entity2keywords[entity])
                        else:
                             entity2keywords[entity] = keywords
        else:
            if response and "none" not in response.lower():
                 keywords_fallback = [k.strip() for k in response.split(',') if k.strip()]
                 if keywords_fallback:
                     entity2keywords["General_Fallback"] = keywords_fallback

    except Exception as e:
        logger.error(f"Error parsing Need output: {e}")
        if response and "none" not in response.lower():
             keywords_fallback = [k.strip() for k in response.split(',') if k.strip()]
             if keywords_fallback:
                 entity2keywords["General_Fallback"] = keywords_fallback

    return entity2keywords

def _format_dualrag_knowledge(knowledge_dict: Dict[str, List[str]]) -> str:
    if not knowledge_dict:
        return "No knowledge gathered yet."
    
    output = []
    for entity, summaries in knowledge_dict.items():
        if summaries:
            output.append(f"#### {entity}")
            for summary in summaries:
                output.append(f"- {summary}")
            output.append("")
    
    return "\n".join(output) if output else "No relevant knowledge gathered yet."

def _format_dualrag_thoughts(thoughts: List[str]) -> str:
    if not thoughts:
        return "No thoughts yet."
    return "\n\n".join([f"#### Step {i+1}\n\n{t}" for i, t in enumerate(thoughts)])

def _format_docs_for_learn(docs: List) -> str:
    if not docs:
        return "No documents."
    return "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(docs)])

class DualRAGPipeline(BasicPipeline):
    """
    DualRAG: Reproduction of official implementation
    
    Core Process:
    1. Infer: Generate reasoning steps, decide if retrieval is needed
    2. Need (Entity Identification): Identify entities and retrieval keywords
    3. Retrieve: Retrieve documents for each entity's keywords
    4. Learn (Knowledge Summarization): Summarize documents into knowledge entries
    5. Loop until Infer decides to answer or max iterations reached
    6. Answer: Generate final answer based on knowledge base
    """
    
    def __init__(self, config, prompt_template=None, retriever=None, generator=None,
                 max_iterations=5):
        super().__init__(config, prompt_template)
        self.generator = get_generator(config) if generator is None else generator
        self.retriever = get_retriever(config) if retriever is None else retriever
        self.max_iterations = max_iterations
        
        self.prompts = {
            "infer": PromptTemplate(config, system_prompt=DUALRAG_INFER_PROMPT,
                                   user_prompt="Knowledge: {knowledge}\nQuestion: {question}\nThoughts: {thought}"),
            "need": PromptTemplate(config, system_prompt=DUALRAG_NEED_PROMPT,
                                  user_prompt="Question: {question}\nThoughts: {thought}\nLatest: {latest_thought}\nHint Entities: {known_entity}"),
            "learn": PromptTemplate(config, system_prompt=DUALRAG_LEARN_PROMPT,
                                   user_prompt="Question: {question}\nReasoning History: {thought}\nRetrieval Details:\nKey Entity Retrieved: {entity}\nRetrieve Queries: {query}\nRetrieved Content:\n{docs}"),
            "answer": PromptTemplate(config, system_prompt=DUALRAG_ANSWER_PROMPT,
                                    user_prompt="Knowledge: {knowledge}\nQuestion: {question}\nThoughts: {thought}")
        }
    
    def run(self, dataset, do_eval=True, pred_process_fun=None):
        questions = dataset.question
        all_preds = []
        all_intermediate = []
        
        for i, question in enumerate(questions):
            logger.info(f"DualRAG processing {i+1}/{len(questions)}: {question}")
            
            try:
                final_answer, intermediate = self._solve_question(question)
                all_preds.append(final_answer)
                all_intermediate.append(intermediate)
            except Exception as e:
                logger.error(f"DualRAG error: {e}", exc_info=True)
                all_preds.append("[ERROR: Processing failed]")
                all_intermediate.append({})
        
        dataset.update_output("pred", all_preds)
        
        if all_intermediate:
            keys = set(k for d in all_intermediate for k in d)
            for key in keys:
                dataset.update_output(key, [d.get(key) for d in all_intermediate])
        
        flatten_retrieval_results(dataset)
        return self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
    
    def _solve_question(self, question: str) -> Tuple[str, Dict]:
        knowledge_base = defaultdict(list)
        thoughts = []
        learned_docs = defaultdict(set)
        
        iterations_log = []
        
        for step in range(self.max_iterations):
            logger.info(f"  Iteration {step + 1}/{self.max_iterations}")
            iter_log = {"step": step + 1}
            
            # === 1. Infer ===
            kb_str = _format_dualrag_knowledge(knowledge_base)
            thought_str = _format_dualrag_thoughts(thoughts)
            
            infer_input = self.prompts["infer"].get_string(
                knowledge=kb_str,
                question=question,
                thought=thought_str
            )
            
            infer_output = self.generator.generate([infer_input])[0]
            new_thought, need_retrieve = _parse_dualrag_infer(infer_output)
            
            thoughts.append(new_thought)
            iter_log["thought"] = new_thought
            iter_log["need_retrieve"] = need_retrieve
            
            logger.info(f"    Thought: {new_thought[:100]}...")
            
            if not need_retrieve:
                logger.info("    Reasoning complete")
                iterations_log.append(iter_log)
                break
            
            # === 2. Need ===
            full_thought_str = _format_dualrag_thoughts(thoughts)
            kb_str = _format_dualrag_knowledge(knowledge_base)
            need_input = self.prompts["need"].get_string(
                knowledge=kb_str,
                question=question,
                thought=full_thought_str,
                latest_thought=new_thought,
                known_entity=list(knowledge_base.keys()) if knowledge_base else "None"
            )
            
            need_output = self.generator.generate([need_input])[0]
            entity2keywords = _parse_dualrag_need(need_output)
            
            iter_log["entities"] = entity2keywords
            
            if not entity2keywords:
                logger.warning("    No entities identified")
                iterations_log.append(iter_log)
                continue
            
            # === 3. Retrieve & 4. Learn ===
            iter_log["retrieval"] = {}
            iter_log["learning"] = {}
            
            for entity, keywords in entity2keywords.items():
                all_docs_for_entity = []
                for keyword in keywords:
                    try:
                        results = self.retriever.search(keyword, k=3) # Default k=3
                        all_docs_for_entity.extend(results)
                    except Exception as e:
                        logger.warning(f"Retrieval failed for keyword '{keyword}': {e}")
                
                # Deduplicate docs
                unique_docs = []
                seen_contents = set()
                for doc in all_docs_for_entity:
                    content = doc.get('contents', doc.get('content', str(doc)))
                    if content not in seen_contents and content not in learned_docs[entity]:
                        unique_docs.append(content)
                        seen_contents.add(content)
                        learned_docs[entity].add(content)
                
                iter_log["retrieval"][entity] = unique_docs
                
                if not unique_docs:
                    continue
                    
                # Learn
                docs_str = _format_docs_for_learn(unique_docs)
                learn_input = self.prompts["learn"].get_string(
                    question=question,
                    thought=full_thought_str,
                    entity=entity,
                    query=", ".join(keywords),
                    docs=docs_str
                )
                
                summary = self.generator.generate([learn_input])[0]
                if "None" not in summary and len(summary) > 5:
                    knowledge_base[entity].append(summary)
                    iter_log["learning"][entity] = summary
            
            iterations_log.append(iter_log)
        
        # === Final Answer ===
        kb_str = _format_dualrag_knowledge(knowledge_base)
        thought_str = _format_dualrag_thoughts(thoughts)
        
        answer_input = self.prompts["answer"].get_string(
            knowledge=kb_str,
            question=question,
            thought=thought_str
        )
        
        final_answer = self.generator.generate([answer_input])[0]
        
        intermediate_steps = {
            "iterations": iterations_log,
            "final_knowledge": dict(knowledge_base),
            "final_thoughts": thoughts
        }
        
        return final_answer, intermediate_steps
