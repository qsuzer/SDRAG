import re
import json
import logging
from typing import List, Tuple, Dict, Set, Optional

from flashrag.prompt import PromptTemplate
from flashrag.utils import get_generator, get_retriever
from .base_pipeline import BaseActivePipeline, flatten_retrieval_results

logger = logging.getLogger(__name__)

# --- PER-PSE Prompts ---

PER_TYPE_SYSTEM_PROMPT = """Given a question, determine what type the question belongs to. Types include:
    1. Bridge: A bridge question involves two or more facts that are connected by an intermediate entity (usually an associative link). The bridge question requires finding the intermediary entity, then using it to answer the question.
    2. Comparison: A comparison question involves comparing two or more independent facts. The comparison question requires analyzing and comparing the differences or similarities between different facts to draw a conclusion.

    Examples:
    Question: What language was used by Renana Jhabvala's mother?
    Output:
    ```json
    {{"Response": "Bridge"}}
    ```

    Question: Which film came out first, The Love Route or Engal Aasan?
    Output:
    ```json
    {{"Response": "Comparison"}}
    ```

    NOTE: Always respond with the JSON object. The question should be preferentially judged as "Bridge" or "Comparison".
    Now it's your turn!
    """

PER_BRIDGE_PLAN_SYSTEM_PROMPT = """Given a bridge question, split it into smaller, independent, and individual subqueries. 
    A bridge question involves two or more facts that are connected by an intermediate entity (usually an associative link). The bridge question requires finding the intermediary entity, then using it to answer the question.
    For the subquery generation, input a tag "<A>" where the answer of the parent query should come to make the query complete. Specifically,
        1. Subquery is NOT allowed to ask open-ended question. For example, for question "What language was used by Renana Jhabvala's mother?", it is NOT allowed to decompose and ask "Who is Renana Jhabvala?".
        2. Each subquery is a simple fact question, not a question that requires reasoning. For example, "Who lives longer, <A1> or <A2>?" is NOT allowed.

    Examples:
    Question: What language was used by Renana Jhabvala's mother?
    Result:
    ```json
    {{
        "Response": {{
            "Q1": ["Who was Renana Jhabvala's mother?", "<A1>"],
            "Q2": ["What language was used by <A1>?", "<A2>"]
        }}
    }}
    ```

    Question: Who is Sobe (Sister Of Saint Anne)'s child-in-law?
    Result:
    ```json
    {{
        "Response": {{
            "Q1": ["Who is the child of Sobe (Sister Of Saint Anne)?", "<A1>"],
            "Q2": ["Who is the spouse of <A1>?", "<A2>"]
        }}
    }}
    ```

    Question: What followed the last person to live in Versailles in the country that became allies with America after the battle of Saratoga?
    Result:
    ```json
    {{
        "Response": {{
            "Q1": ["Who became allies with America after the Battle of Saratoga?", "<A1>"],
            "Q2": ["Who was the last person to live in Versailles?", "<A2>"],
            "Q3": ["What followed <A2> in <A1>?", "<A3>"]
        }}
    }}
    ```

    NOTE: Always respond with the JSON object. Do not output any explanation or other text.
    Now it's your turn!
    """

PER_COMPARISON_PLAN_SYSTEM_PROMPT = """Given a comparison question, split it into smaller, independent, and individual subqueries.
    A comparison question involves comparing two or more independent facts. The comparison question requires analyzing and comparing the differences or similarities between different facts to draw a conclusion.
    For the subquery generation, input a tag "<A>" where the answer of the parent query should come to make the query complete. Specifically,
        1. Subquery is NOT allowed to ask open-ended question. For example, for question "What language was used by Renana Jhabvala's mother?", it is NOT allowed to decompose and ask "Who is Renana Jhabvala?".
        2. Each subquery is a simple fact question, not a question that requires reasoning. For example, "Who lives longer, <A1> or <A2>?" is NOT allowed.

    Examples:
    Question: Are Harper High School (Chicago) and Santa Sabina College located in the same country?
    Result:
    ```json
    {{
        "Response": {{
            "Q1": ["What country is Santa Sabina College located in?", "<A1>"],
            "Q2": ["What country is Harper High School (Chicago) located in?", "<A2>"]
        }}
    }}
    ```

    Question: Who lived longer, Constance Keys or Anthony De Jasay?
    Result:
    ```json
    {{
        "Response": {{
            "Q1": ["When was Constance Keys born?", "<A1>"],
            "Q2": ["When did Constance Keys die?", "<A2>"],
            "Q3": ["When was Anthony De Jasay born?", "<A3>"],
            "Q4": ["When did Anthony De Jasay die?", "<A4>"]
        }}
    }}
    ```

    Question: Which film has the director born later, A Flame In My Heart or Butcher, Baker, Nightmare Maker?
    Result:
    ```json
    {{
        "Response": {{
            "Q1": ["Who was the director of the film A Flame in My Heart?", "<A1>"],
            "Q2": ["Who was the director of the film Butcher, Baker, Nightmare Maker?", "<A2>"],
            "Q3": ["When was <A1> born?", "<A3>"],
            "Q4": ["When was <A2> born?", "<A4>"]
        }}
    }}
    ```

    NOTE: Always respond with the JSON object. Do not output any explanation or other text.
    Now it's your turn!
    """

PER_RAG_QA_SYSTEM_PROMPT = """You are a question answering system. Use the retrievals while generating the answers and keep the answers grounded in the retrievals.
    Generate a JSON with a single key "Response" and a value that is a short phrase or a few words. In JSON, put every value as a string always, not float.

    Examples:
    Query: In which state is Hertfordshire located?
    Retrievals:
        1. Hertfordshire: Hertfordshire is the county immediately north of London and is part of the East of England region, a mainly statistical unit. A significant minority of the population across all districts are City of London commuters. To the east is Essex, to the west is Buckinghamshire and to the north are Bedfordshire and Cambridgeshire.
        2. Hertfordshire: Hertfordshire is an administrative and historic county situated in the East of England, which is part of the United Kingdom. Within England, Hertfordshire serves as an administrative division and is bordered by Greater London to the south, with its landscape largely encompassed by the London Basin.
        3. Hertfordshire Chain Walk: The Hertfordshire Chain Walk is located in Hertfordshire, England, and consists of 15 linked circular walks. These walks, each of which is between 4.25 and 9 miles, make up a total distance of 87 miles. The tracks pass through villages in East Hertfordshire close to London, the Icknield Way and the Cambridgeshire border.  
        4. University of Hertfordshire: Campus is the university\'s Learning Resource Centre, a combined library and computer centre. The University of Hertfordshire Students\' Union is headquartered at College Lane campus. The College Lane campus is also the location of Hertfordshire International College, which is part of the Navitas group, providing a direct pathway for international students to the University. 
        5. Moor Park, Hertfordshire: Moor Park, Hertfordshire Moor Park is a private residential estate in the Three Rivers District of Hertfordshire, England. Located approximately northwest of central London and adjacent to the Greater London boundary, it is a suburban residential development. It takes its name from Moor Park, a country house which was originally built in 1678\u20139 for James,
    Answer:
    ```json
    {{"Response": "East of England"}}
    ```

    Query: Who plays michael myers in halloween by Rob Zombie?
    Retrievals:
        1. Halloween (2007 film): Halloween is a 2007 American slasher film written, directed, and produced by Rob Zombie. The film stars Tyler Mane as the adult Michael Myers, Malcolm McDowell as Dr. Sam Loomis. Rob Zombie\'s ""reimagining"" follows the premise of John Carpenter\'s original, with Michael Myers stalking Laurie Strode and her friends on Halloween night.
        2. Halloween (1978 film): A remake was released in 2007, directed by Rob Zombie, which itself was followed by a 2009 sequel. An eleventh installment was released in the United States in 2018. The film, directed by David Gordon Green, is a direct sequel to the original film while disregarding the previous sequels from canon, and retconing the ending of the first film. A sequel is in early development.
        3. Rob Zombie: Rob Zombie Rob Zombie (born Robert Bartleh Cummings; January 12, 1965) is an American musician and filmmaker who rose to fame as a founding member of the heavy metal band White Zombie, releasing four studio albums with the band. He is the older brother of Spider One, lead vocalist for American rock band Powerman 5000. Zombie's first solo effort was a song titled \"\"Hands of Death (Burn Baby Burn)\"\" (1996)
        4. Halloween II (2009 film): by The Weinstein Company, and planned to be released in 2012. That film was ultimately cancelled in 2012. \"\"Halloween 3D\"\" was planned to have Michael Myers stalk Laurie Strode while she was confined in a mental asylum. Halloween II (2009 film) Halloween II is a 2009 American slasher film written, directed, and produced by Rob Zombie. 
        5. Laurie Strode: Laurie Strode Laurie Strode is a fictional character in the \"\"Halloween\"\" franchise, portrayed by actresses Jamie Lee Curtis and Scout Taylor-Compton. One of the two main protagonists of the overall series (the other being Dr. Sam Loomis), she appears in seven of the eleven \"\"Halloween\"\" films, first appearing in John Carpenter's original 1978 film. 
    Answer:
    ```json
    {{"Response": "Tyler Mane"}}
    ```

    Query: Who wrote the theme song to Charlie Brown?
    Retrievals:
        1. Todd Dulaney: Todd Dulaney Todd Anthony Dulaney (born December 20, 1983) is an American gospel musician, and former baseball player. His music career started in 2011, with the release of the CD version, \"\"Pulling Me Through\"\". This would be his breakthrough released upon the \"\"Billboard\"\" Gospel Albums chart. He would release another album, \"\"A Worshipper's Heart\"\", in 2016 with EntertainmentOne Nashville, 
        2. Dulaney High School: Dulaney High School Dulaney High School is a secondary school in Timonium, Baltimore County, Maryland. The school serves a generally upper-middle class suburban community, with students from Timonium and surrounding areas in Baltimore County. Dulaney is a Blue Ribbon School and ranked #259 nationwide in \"\"Newsweek\"\" magazine's 2010 survey of top public high schools in the U.S.
        3. Dulaney High School: Blue Ribbon School of Excellence in 1995. In 2010, Dulaney was named #259 on \"\"Newsweek\"\" magazine's \"\"1,200 Top U.S. high schools\"\" annual national survey. Dulaney High School Dulaney High School is a secondary school in Timonium, Baltimore County, Maryland. The school serves a generally upper-middle class suburban community,
        4. Todd Dulaney: No. 13 on the Independent Albums chart. Dulaney's wife is Kenyetta Stone-Dulaney, and together they have four children, Tenley, Taylor, Tyler, and Todd Jr., who attend church at Living Word Christian located in Forest Park, Illinois. Todd Dulaney Todd Anthony Dulaney (born December 20, 1983) is an American gospel musician, and former baseball player. His music career started in 2011, with
        5. Clermont (Alexandria, Virginia): Clermont Plantation was built by Benjamin Dulaney in the late 18th century. Dulaney, a friend of George Washington, used the estate as his summer residence. Clermont was large in size with two parlors, eleven bedrooms, and multiple outbuildings. Dulaney's family members were loyalists during the American Revolutionary War and many of them lost their possessions and property.
    Answer:
    ```json
    {{"Response": "Vince Guaraldi"}}
    ```

    NOTE: Always respond with the JSON object. Do not output any explanation or other text.
    Now it's your turn!
    """

PER_BRIDGE_AGGREGATE_SYSTEM_PROMPT = """You are a question answering system. Use the evidence while generating the answer and keep the answer grounded in the evidence. Each piece of evidence is represented as "Question >> Answer", where ">>" means "the Answer to the Question is...".
    Generate a JSON with a single key "Response" and a value that is a short phrase or a few words. In JSON, put every value as a string always, not float.

    Examples:
    Question: When was the baseball team winning the world series in 2015 baseball created?
    Evidence:
        1. Who won the world series in 2015 baseball? >> Kansas City Royals
        2. When was Kansas City Royals created? >> 1969
    Answer:
    ```json
    {{"Response": "1969"}}
    ```

    Question: When did the French come to the region where Philipsburg is located?
    Evidence:
        1. Where is Philipsburg located? >> Sint Maarten
        2. What terrain feature is located in the Sint Maarten region? >> Great Bay and Great Salt Pond
        3. When did the French come to Great Bay and Great Salt Pond? >> 1625
    Answer:
    ```json
    {{"Response": "1625"}}
    ```

    Question: How many people who started the great migration of the Slavs live in the country the football tournament is held?
    Evidence:
        1. Who started the Great Migration of the Slavs? >> Germans
        2. Where was the football tournament held? >> Brazil
        3. How many of Germans live in Brazil? >> 5 million
    Answer:
    ```json
    {{"Response": "5 million"}}
    ```

    NOTE: Always respond with the JSON object. Do not output any explanation or other text.
    Now it's your turn!
    """

PER_COMPARISON_AGGREGATE_SYSTEM_PROMPT = """You are a question answering system. Use the evidence while generating the answer and keep the answer grounded in the evidence. Each piece of evidence is represented as "Question >> Answer", where ">>" means "the Answer to the Question is...".
    Generate a JSON with a single key "Response" and a value that is a short phrase or a few words. In JSON, put every value as a string always, not float.

    Examples:
    Question: Which film came out first, The Love Route or Engal Aasan?
    Evidence:
        1. When was The Love Route released? >> February 25, 1915
        2. When was Engal Aasan released? >> July 2009
    Answer:
    ```json
    {{"Response": "The Love Route"}}
    ```

    Question: Who lived longer, Dina Vierny or Muhammed Bin Saud Al Saud?
    Evidence:
        1. When was Dina Vierny born? >> 25 January 1919
        2. When did Dina Vierny die? >> 20 January 2009
        3. When was Muhammed Bin Saud Al Saud born? >> 21 March 1934
        4. When did Muhammed Bin Saud Al Saud die? >> 8 July 2012
    Answer:
    ```json
    {{"Response": "Dina Vierny"}}
    ```

    Question: Do both films The Falcon (Film) and Valentin The Good have the directors from the same country?
    Evidence:
        1. Who is the director of The Falcon (Film)? >> Vatroslav Mimica
        2. Who is the director of Valentin The Good? >> Martin Frič
        3. Which country is Vatroslav Mimica from? >> Croatia
        4. Which country is Martin Frič from? >> Czech Republic
    Answer:
    ```json
    {{"Response": "No"}}
    ```

    NOTE: Always respond with the JSON object. Do not output any explanation or other text.
    Now it's your turn!
    """

PER_HYDE_SYSTEM_PROMPT = """Please write a passage to answer the question."""


class PERQARAGPipeline(BaseActivePipeline):
    """
    PER-PSE RAG Pipeline (Planner-Executor-Reasoner).
    
    Workflow:
    1. Planner: Determine Question Type (Bridge/Comparison) & Generate Decomposition Plan.
    2. Executor: Execute the plan graph topologically (filling <A> tags).
       - Supports HyDE (Hypothetical Document Embeddings).
       - Step-by-step retrieval and QA.
    3. Reasoner: Synthesize final answer from the execution trace.
    
    Reference: "Beyond the Answer: Advancing Multi-Hop QA with Fine-Grained Graph Reasoning and Evaluation"
    """

    def __init__(self, config, prompt_template=None, retriever=None, generator=None, use_hyde=True):
        # Base init handles generator/retriever setup
        super().__init__(config, prompt_template, retriever, generator)
        self.use_hyde = use_hyde
        
        # Register PER-specific prompts
        self.prompts.update({
            "per_type": PromptTemplate(config, system_prompt=PER_TYPE_SYSTEM_PROMPT, user_prompt="Question: {question}\nOutput: "),
            "per_bridge_plan": PromptTemplate(config, system_prompt=PER_BRIDGE_PLAN_SYSTEM_PROMPT, user_prompt="Question: {question}\nResult: "),
            "per_comparison_plan": PromptTemplate(config, system_prompt=PER_COMPARISON_PLAN_SYSTEM_PROMPT, user_prompt="Question: {question}\nResult: "),
            "per_rag_qa": PromptTemplate(config, system_prompt=PER_RAG_QA_SYSTEM_PROMPT, user_prompt="Query: {question}\nRetrievals:\n{paragraphs}\nAnswer: "),
            "per_bridge_agg": PromptTemplate(config, system_prompt=PER_BRIDGE_AGGREGATE_SYSTEM_PROMPT, user_prompt="Question: {question}\nEvidence:\n{paragraphs}\nAnswer: "),
            "per_comparison_agg": PromptTemplate(config, system_prompt=PER_COMPARISON_AGGREGATE_SYSTEM_PROMPT, user_prompt="Question: {question}\nEvidence:\n{paragraphs}\nAnswer: "),
            "per_hyde": PromptTemplate(config, system_prompt=PER_HYDE_SYSTEM_PROMPT, user_prompt="Question: {question}\nPassage: "),
        })

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        """
        Override run to skip the generic batch decomposition from BaseActivePipeline.
        PER uses its own specific planner logic per question.
        """
        questions = dataset.question
        all_preds = []
        all_intermediate_steps = []
        
        for i, question in enumerate(questions):
            logger.info(f"PER-PSE processing question {i + 1}/{len(questions)}: {question}")
            try:
                # No pre-decomposed sub_questions passed here, PER generates them internally
                final_answer, intermediate_steps = self._process_single_question(question)
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
                values = [d.get(key) for d in all_intermediate_steps]
                
                # Fix: Ensure 'prompt' is a string for metrics calculation
                if key == 'prompt':
                    sanitized_values = []
                    for v in values:
                        if isinstance(v, list):
                            try:
                                v_str = "\n".join([m.get('content', str(m)) if isinstance(m, dict) else str(m) for m in v])
                            except:
                                v_str = str(v)
                            sanitized_values.append(v_str)
                        else:
                            sanitized_values.append(v)
                    values = sanitized_values

                dataset.update_output(key, values)
        
        flatten_retrieval_results(dataset)
        return self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

    def _parse_json_response(self, response: str, key: str = "Response"):
        """Helper to robustly parse JSON from LLM response."""
        try:
            cleaned = response.strip()
            
            # Strategy 1: Markdown code block
            # Case insensitive for 'json', optional
            match = re.search(r"```(?:json|JSON)?\s*(.*?)```", cleaned, re.DOTALL)
            if match:
                candidate = match.group(1).strip()
                if candidate:
                    cleaned = candidate
            
            # Strategy 2: Find outermost {} if Strategy 1 failed or returned non-JSON
            if not cleaned.startswith("{"):
                # Find the first {
                start_idx = cleaned.find("{")
                # Find the last }
                end_idx = cleaned.rfind("}")
                
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    cleaned = cleaned[start_idx : end_idx + 1]
            
            data = json.loads(cleaned)
            if key and key in data:
                return data[key]
            return data
        except Exception as e:
            logger.warning(f"PER Parser: Failed to parse JSON: {e}. Raw response: {response[:100]}...")
            # Fallback: return raw string if parsing fails
            return response

    def _process_single_question(self, question: str) -> Tuple[str, Dict]:
        """
        PER-PSE Execution Logic: Planning -> Execution -> Reasoning
        """
        intermediate_steps = {}
        
        # --- Step 1: Planning ---
        logger.info(f"PER: Step 1 - Planning")
        
        # 1.1 Classify Question Type
        type_prompt = self.prompts["per_type"].get_string(question=question)
        type_resp = self.generator.generate([type_prompt])[0]
        q_type = self._parse_json_response(type_resp, "Response")
        
        if not isinstance(q_type, str):
            q_type = str(q_type)
            
        intermediate_steps["question_type"] = q_type
        
        # 1.2 Generate Plan
        if "comparison" in q_type.lower():
            plan_prompt = self.prompts["per_comparison_plan"].get_string(question=question)
        else:
            # Default to Bridge for 'bridge' type or unknowns
            plan_prompt = self.prompts["per_bridge_plan"].get_string(question=question)
            
        plan_resp = self.generator.generate([plan_prompt])[0]
        plan = self._parse_json_response(plan_resp, "Response")
        intermediate_steps["plan"] = plan
        
        if not isinstance(plan, dict):
            return "Plan generation failed.", intermediate_steps

        # --- Step 2: Executor (Graph Execution) ---
        # Sort keys (Q1, Q2...) to ensure dependency order
        sorted_keys = sorted(plan.keys(), key=lambda k: int(re.search(r'\d+', k).group()) if re.search(r'\d+', k) else 0)
        
        tag_memory = {} # Store answers for placeholders like <A1>
        execution_trace = []
        all_retrieved_docs = []

        for q_key in sorted_keys:
            q_info = plan[q_key]
            # Expected structure: ["Question Template", "<Tag>"]
            if not isinstance(q_info, list) or len(q_info) < 2:
                continue
                
            template, tag = q_info[0], q_info[1]
            query = template
            
            # 2.1 Dependency Resolution (Fill placeholders)
            placeholders = re.findall(r"<A(\d+)>", template)
            can_execute = True
            for ph_num in placeholders:
                ph_tag = f"<A{ph_num}>"
                if ph_tag in tag_memory:
                    query = query.replace(ph_tag, tag_memory[ph_tag])
                else:
                    logger.warning(f"  PER: Missing dependency {ph_tag} for {q_key}. Skipping.")
                    can_execute = False
            
            if not can_execute:
                continue

            logger.info(f"  Executing {q_key}: {query}")

            # 2.2 HyDE (Optional)
            search_query = query
            if self.use_hyde:
                hyde_input = self.prompts["per_hyde"].get_string(question=query)
                hyde_doc = self.generator.generate([hyde_input])[0]
                search_query = f"{query}\n{hyde_doc}"
            
            # 2.3 Retrieval
            retrievals, _ = self.retriever.search(search_query, return_score=True)
            
            formatted_docs = []
            if retrievals:
                for idx, doc in enumerate(retrievals):
                    if isinstance(doc, dict):
                        content = doc.get('contents', doc.get('text', str(doc)))
                        title = doc.get('title', '')
                        doc_str = f"{idx+1}. {title}: {content}"
                    else:
                        doc_str = f"{idx+1}. {str(doc)}"
                    formatted_docs.append(doc_str)
                
            docs_str = "\n".join(formatted_docs)
            all_retrieved_docs.extend(formatted_docs)

            # 2.4 Answer Generation
            qa_prompt = self.prompts["per_rag_qa"].get_string(
                question=query, 
                paragraphs=docs_str
            )
            ans_resp = self.generator.generate([qa_prompt])[0]
            answer = self._parse_json_response(ans_resp, "Response")
            
            if isinstance(answer, list) and len(answer) > 0:
                answer = answer[0]
            elif not isinstance(answer, str):
                answer = str(answer)

            # 2.5 Update Memory
            target_tag = tag 
            tag_memory[target_tag] = answer
            
            execution_trace.append({
                "id": q_key,
                "query": query,
                "answer": answer,
                "tag": target_tag
            })

        intermediate_steps["execution_trace"] = execution_trace
        intermediate_steps["sub_retrieval_results"] = all_retrieved_docs

        # --- Step 3: Reasoner (Aggregation) ---
        logger.info("PER: Step 3 - Aggregating")
        
        evidence_list = [f"{step['query']} >> {step['answer']}" for step in execution_trace]
        evidence_str = "\n".join([f"{i+1}. {e}" for i, e in enumerate(evidence_list)])
        
        if "comparison" in q_type.lower():
            agg_prompt = self.prompts["per_comparison_agg"].get_string(
                question=question,
                paragraphs=evidence_str
            )
        else:
            agg_prompt = self.prompts["per_bridge_agg"].get_string(
                question=question,
                paragraphs=evidence_str
            )
        
        intermediate_steps["prompt"] = agg_prompt
            
        final_resp = self.generator.generate([agg_prompt])[0]
        final_answer = self._parse_json_response(final_resp, "Response")
        
        if isinstance(final_answer, list) and len(final_answer) > 0:
            final_answer = final_answer[0]
            
        return str(final_answer), intermediate_steps
