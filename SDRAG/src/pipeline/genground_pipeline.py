import re
import json
import logging
from typing import List, Tuple, Dict, Set, Optional
from collections import Counter
from copy import deepcopy

from flashrag.pipeline import BasicPipeline
from flashrag.utils import get_generator, get_retriever
from flashrag.prompt import PromptTemplate
from .base_pipeline import flatten_retrieval_results

logger = logging.getLogger(__name__)

# --- GenGround Prompts & Examples ---

GENGROUND_ICL_EXAMPLES = {
        "hotpot": """Question: Which magazine was started first, Arthur's Magazine or First for Women?
    Thought 1: First, I should ask "When Arthur's Magazine started?"
    Answer 1: Arthur's Magazine was an American literary periodical published in Philadelphia, which started in **19th (1844)**.
    Thought 2: Then, I should ask "When First for Women started?"
    Answer 2: "First for Women" is a well-known women's magazine that covers topics such as health, beauty, fitness, food, and lifestyle. It started in **1989**.
    Thought 3: Which one started first?
    Answer 3: FINISH[Arthur's Magazine]

    Question: Who was the captain of the only battleship to provide gunfire support during the Vietnam War?
    Thought 1: Who was the captain of the battleship that provided gunfire support during the Vietnam War?
    Answer 1: Rear Adm. J. Edward Snyder, Jr. (October 23, 1924 – November 4, 2007) was notable as the captain of the battleship USS "New Jersey" during that ship's deployment to the Vietnam War in 1968.
    Thought 2: I have got all the information. And **Rear Adm. J. Edward Snyder** is the captain providing gunfire support during the Vietnam War.
    Answer 2: FINISH[Rear Adm. J. Edward Snyder]

    Question: How old is the female main protagonist of Catching Fire?
    Thought 1: What is the Catching Fire?
    Answer 1: Catching Fire is the second book in "The Hunger Games trilogy" written by Suzanne Collins. It is written in the voice of **Katniss Everdeen**.
    Thought 2: Katniss Everdeen is the protagonist of Catching Fire. How old is Katniss Everdeen in Catching Fire book?
    Answer 2: Katniss Everdeen in Catching Fire book is **16 years old**
    Thought 3: Katniss Everdeen, the female main protagonist of Catching Fire, is **16** years old.
    Answer 3: FINISH[16]

    Question: What is one of the stars of The Newcomers known for?
    Thought 1: Who are the stars in Newcomers?
    Answer 1: **Chris Evans** is one of the stars in the Newcomers.
    Thought 2: Chris Evans is one of the star in The Newcomers. What is the Chris Evans known for?
    Answer 2: Chris Evans is known for **superhero roles as the Marvel Comics**
    Thought 3: I have got all the information. Chris Evans is a star in Newcomers, who is known for **superhero roles as the Marvel Comics**
    Answer 3: FINISH[Superhero roles as the Marvel Comics]""",
        
        "musique": """Question: Where do Greyhound buses leave from in the city where Arna Selznick's employer is headquartered?
    Thought 1: Who is the employer of Arna Selznick?
    Answer 1: The employer of Arna Selznick is **Nelvana** since he directed Nelvana's 1985 animated film The Care Bears Movie.
    Thought 2: Nelvana is an animation studio and entertainment company. Where is the headquarters of Nelvana?
    Answer 2: The headquarters of Nelvana is in **Toronto**.
    Thought 3: Where do Greyhound buses leave from Toronto?
    Answer 3: Greyhound buses leave from **Toronto Coach Terminal**
    Thought 4: The employer of Arna Selznick is Nelvana. Nelvana's headquartered is in Toronto. Greyhound leave from **Toronto Coach Terminal**
    Answer 4: FINISH[Toronto Coach Terminal]

    Question: Which county does Lloyd Dane's birthplace belong to?
    Thought 1: What is the Lloyd Dane's birthplace?
    Answer 1: Lloyd Dane's birthplace is **Eldon**.
    Thought 2: Eldon is a city. Which country does Eldon belong to?
    Answer 2: Eldon belongs to **Miller County**.
    Thought 3: Lloyd Dane's birthplace is Eldon. Eldon belongs to Miller County.
    Answer 3: FINISH[Miller County]

    Question: Who wrote "Turn Me On" by the performer of "Happy Pills"?
    Thought 1: Happy Pills is a song from American. Who is the performer of "Happy Pills"?
    Answer 1: The performer of Happy Pills is **Norah Jones**.
    Thought 2: Turn Me On is a song. Who wrote "Turn Me On" performed by Norah Jones?
    Answer 2: **John D. Loudermilk** wrote "Turn Me On".
    Thought 3: The performer of "Happy Pills" is **Norah Jones**. **John D. Loudermilk** wrote the "Turn Me On" by Norah Jones.
    Answer 3: FINISH[John D. Loudermilk]"""
}

GENGROUND_GENERATION_SYSTEM = """Let's think step by step to answer the question. You need to decompose a complex question into several sub-questions and answer them step by step until get the final answer."""

GENGROUND_GENERATION_USER = """Please answer the question step by step by interleaving Thought and Answer.
    - Thought: reason about the current situation and formulate a sub-question. Your Thought process should aim to formulate as simple and specific a question as possible, which should include a clear entity or event.
    - Answer: answer the sub-question proposed in the Thought step.

    Starting below, you must follow the following format:
    Question: a complex question
    Thought 1: The first sub-question
    Answer 1: answer the first sub-question
    ... (the Thought and Answer steps can repeat N times)
    Thought n: the final thought
    Answer n: FINISH[your final answer]

    Note:
    1. It is better NOT TO use pronouns in Answer and Thought step, but to use the corresponding results obtained previously. For example, instead of "What is the most popular movie directed by this person", you should use "What is the most popular movie directed by Martin Scorsese".
    2. Your final answer should be an entity, e.g., a date, place name, and person name. You should always bold the key information with **.
    3. You should always give the answer your trust most despite not knowing it exactly. Try to avoid giving "I do not know".

    Here are some examples:

    {icl_example}

    Question: {query}
    """

GENGROUND_BATCH_GROUNDING_SYSTEM = """You will be provided with {n} documents delimited by triple quotes and a question.
    Your task is to answer the question using only the provided document and to cite the passage(s) of the document used to answer the question.
    Please be careful:
    1. If the document does not contain the information needed to answer this question then simply write `Insufficient information`.
    2. If an answer to the question is provided, it must be annotated with a citation. Use the following format to cite relevant passages ({{"citation": …}})."""

GENGROUND_BATCH_GROUNDING_USER = """\"\"\"{doc}\"\"\"

    Question: {question}"""

GENGROUND_REVISE_SYSTEM = """You will be provided with {n} documents delimited by triple quotes and a question.
    Your task is to edit the candidate answer using only the provided document and to cite the passage(s) of used to edit the candidate answer.
    Your answers need to be short and precise (less than 20 words). Do not introduce information that is not relevant to the question."""

GENGROUND_REVISE_USER = """\"\"\"{knowledge}\"\"\"
    Question: {q}
    Candidate Answer: {ans}
    Your output should be JSON format: {{"answer": "<the edited answer>","citation":"<cite the passage used to edit the candidate answer>"}}"""


class GenGroundPipeline(BasicPipeline):
    """
    Generate-then-Ground Pipeline for Multi-hop Question Answering.
    
    Official implementation reference: https://github.com/mangopy/Generate-then-Ground
    """
    
    def __init__(self, config, prompt_template=None, retriever=None, generator=None,
                 max_iterations=10, batch_doc_size=3, dataset_name="hotpot"):
        super().__init__(config, prompt_template)
        self.generator = get_generator(config) if generator is None else generator
        self.retriever = get_retriever(config) if retriever is None else retriever
        self.max_iterations = max_iterations
        self.batch_doc_size = batch_doc_size
        self.dataset_name = dataset_name
        
        self.icl_example = GENGROUND_ICL_EXAMPLES.get(dataset_name, GENGROUND_ICL_EXAMPLES["hotpot"])
        
        logger.info(f"GenGround initialized: max_iterations={max_iterations}, batch_doc_size={batch_doc_size}")
        
        self.prompts = {
            "generation": PromptTemplate(
                config,
                system_prompt=GENGROUND_GENERATION_SYSTEM,
                user_prompt=GENGROUND_GENERATION_USER
            ),
            "batch_grounding": PromptTemplate(
                config,
                system_prompt=GENGROUND_BATCH_GROUNDING_SYSTEM,
                user_prompt=GENGROUND_BATCH_GROUNDING_USER
            ),
            "revise": PromptTemplate(
                config,
                system_prompt=GENGROUND_REVISE_SYSTEM,
                user_prompt=GENGROUND_REVISE_USER
            )
        }
        self.re_art = re.compile(r'\b(a|an|the)\b')
        self.re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')
    
    def run(self, dataset, do_eval=True, pred_process_fun=None):
        questions = dataset.question
        
        all_preds = []
        all_intermediate_steps = []
        
        for i, question in enumerate(questions):
            logger.info(f"GenGround processing question {i + 1}/{len(questions)}: {question}")
            try:
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
                dataset.update_output(key, values)
        
        flatten_retrieval_results(dataset)
        return self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

    def _process_single_question(self, question: str) -> Tuple[str, Dict]:
        intermediate_steps = {
            "thoughts": [],
            "answers": [],
            "candidate_answers": [],
            "grounded_answers": [],
            "retrieval_results": [],
            "reasoning_trajectory": ""
        }
        
        reasoning_trajectory = ""
        
        generation_prompt = GENGROUND_GENERATION_USER.format(
            icl_example=self.icl_example,
            query=question
        )

        for i in range(1, self.max_iterations + 1):
            logger.info(f"  Iteration {i}/{self.max_iterations}")
            
            generation_prompt_final = None

            try:
                if hasattr(self.prompts["generation"], 'system_prompt') and self.prompts["generation"].system_prompt:
                    prompt_with_system = self.prompts["generation"].get_string(
                        icl_example=self.icl_example,
                        query=question
                    )
                    generation_prompt_final = prompt_with_system
                else:
                    generation_prompt_final = generation_prompt

                if reasoning_trajectory:
                    if isinstance(generation_prompt_final, list):
                        found = False
                        for msg in reversed(generation_prompt_final):
                            if msg.get('role') == 'user':
                                msg['content'] = msg['content'].replace(
                                    f"Question: {question}",
                                    f"Question: {question}\n{reasoning_trajectory}"
                                )
                                found = True
                                break
                        if not found:
                             logger.warning("Could not find user message to append trajectory")
                    else:
                        generation_prompt_final = generation_prompt_final.replace(
                            f"Question: {question}",
                            f"Question: {question}\n{reasoning_trajectory}"
                        )
                
                if isinstance(generation_prompt_final, list):
                    found = False
                    for msg in reversed(generation_prompt_final):
                        if msg.get('role') == 'user':
                            msg['content'] = msg['content'] + f"Thought {i}: "
                            found = True
                            break
                    if not found:
                             logger.warning("Could not find user message to append thought")
                else:
                    generation_prompt_final = generation_prompt_final + f"Thought {i}: "
            
            except Exception as e:
                logger.error(f"Failed during prompt preparation: {e}", exc_info=True)
                break
                
            try:
                generate_input = [generation_prompt_final]
                generated = self.generator.generate(generate_input)
                response = generated[0] if isinstance(generated, list) else generated
                response = response.strip()
                
                if 'Answer' in response:
                    if f'Answer {i}:' in response:
                        parts = response.split(f'Answer {i}:')
                    elif 'Answer:' in response:
                        parts = response.split('Answer:')
                    else:
                        parts = response.split('Answer')
                    
                    sub_question = parts[0].strip()
                    candidate_answer = parts[1].strip() if len(parts) > 1 else ""
                else:
                    lines = response.strip().split('\n')
                    sub_question = lines[0].strip()
                    
                    answer_prompt = None
                    if isinstance(generation_prompt_final, list):
                        answer_prompt = deepcopy(generation_prompt_final)
                        found = False
                        for msg in reversed(answer_prompt):
                            if msg.get('role') == 'user':
                                msg['content'] = msg['content'].rstrip(f"Thought {i}: ") + f"Thought {i}: {sub_question}\nAnswer {i}: "
                                found = True
                                break
                        if not found:
                             logger.warning("Could not find user message for answer-gen")
                    else:
                        answer_prompt = generation_prompt_final.rstrip(f"Thought {i}: ") + f"Thought {i}: {sub_question}\nAnswer {i}: "
                    
                    answer_generated = self.generator.generate([answer_prompt])
                    candidate_answer = answer_generated[0] if isinstance(answer_generated, list) else answer_generated
                    candidate_answer = candidate_answer.strip()
                
            except Exception as e:
                logger.error(f"Generation failed at step {i}: {e}", exc_info=True)
                break
            
            if 'FINISH' in candidate_answer:
                final_answer = self._normalize_answer(candidate_answer)
                reasoning_step = f"Thought {i}: {sub_question}\nAnswer {i}: {final_answer}\n"
                reasoning_trajectory += reasoning_step
                intermediate_steps["reasoning_trajectory"] = reasoning_trajectory
                logger.info(f"  Reached FINISH signal.  {final_answer}")
                return final_answer, intermediate_steps
            
            sub_question = self._normalize_answer(sub_question)
            candidate_answer = self._normalize_answer(candidate_answer)
            
            intermediate_steps["thoughts"].append(sub_question)
            intermediate_steps["candidate_answers"].append(candidate_answer)
            
            logger.info(f"    Sub-question: {sub_question[:100]}...")
            logger.info(f"    Candidate answer: {candidate_answer[:100]}...")
            
            logger.info(f"    Retrieving documents...")
            raw_docs, scores = self.retriever.search(sub_question, return_score=True)
            
            documents = self._extract_doc_texts(raw_docs)
            intermediate_steps["retrieval_results"].append(documents)
            
            logger.info(f"    Grounding answer with {len(documents)} retrieved documents...")
            grounded_answer = self._grounding(
                sub_question=sub_question,
                candidate_answer=candidate_answer,
                documents=documents,
                batch_size=self.batch_doc_size
            )
            
            intermediate_steps["grounded_answers"].append(grounded_answer)
            intermediate_steps["answers"].append(grounded_answer)
            
            logger.info(f"    Grounded answer: {grounded_answer[:100]}...")
            
            reasoning_step = f"Thought {i}: {sub_question}\nAnswer {i}: {grounded_answer}\n"
            reasoning_trajectory += reasoning_step
        
        logger.warning(f"  Reached max iterations without FINISH signal")
        final_answer = intermediate_steps["grounded_answers"][-1] if intermediate_steps["grounded_answers"] else "NO ANSWER"
        intermediate_steps["reasoning_trajectory"] = reasoning_trajectory
        
        return final_answer, intermediate_steps

    def _grounding(self, sub_question: str, candidate_answer: str,
                   documents: List[str], batch_size: int = 2) -> str:
        if not documents:
            return candidate_answer
        
        logger.info(f"     Grounding: {len(documents)} docs, batch_size={batch_size}")
        
        if len(documents) <= batch_size:
            useful_docs = self._batch_grounding(sub_question, documents)
            docs_for_revise = useful_docs if useful_docs else documents
            logger.info(f"     Direct revision with {len(docs_for_revise)}/{len(documents)} docs")
        else:
            doc_queue = documents.copy()
            iteration = 0
            max_iterations = (len(documents) // batch_size) + 1
            
            while len(doc_queue) >= batch_size and iteration < max_iterations:
                batch_docs = doc_queue[:batch_size]
                useful_docs = self._batch_grounding(sub_question, batch_docs)
                
                doc_queue = doc_queue[batch_size:] + useful_docs
                iteration += 1
                
                if len(doc_queue) == 0 or (len(useful_docs) > 0 and len(doc_queue) <= batch_size):
                    break
            
            docs_for_revise = doc_queue if doc_queue else documents
            logger.info(f"     After filtering: {len(docs_for_revise)}/{len(documents)} docs remain")
        
        if not docs_for_revise:
            return candidate_answer
        
        revised_answer = self._revise_answer(
            sub_question=sub_question,
            candidate_answer=candidate_answer,
            documents=docs_for_revise
        )
        
        return revised_answer
    
    def _batch_grounding(self, sub_question: str, batch_docs: List[str]) -> List[str]:
        if not batch_docs:
            return []
        
        formatted_docs = '\n'.join([str(doc) for doc in batch_docs])
        
        try:
            grounding_input_messages = self.prompts["batch_grounding"].get_string(
                n=len(batch_docs),
                doc=formatted_docs,
                question=sub_question
            )
        except Exception as e:
            logger.error(f"Failed to format grounding prompt: {e}")
            return []

        try:
            generated = self.generator.generate([grounding_input_messages])
            response = generated[0] if isinstance(generated, list) else generated
            
            if "citation" in response.lower():
                evidence = ""
                try:
                    idx = response.lower().index("citation") + len("citation")
                    evidence = response[idx:].replace('"', '').replace("({", '').replace("})", '').replace(":", '').strip()
                    
                    if evidence.endswith(")") or evidence.endswith("}"):
                         evidence = evidence[:-1].strip()

                    if not evidence:
                        return []
                except Exception as e:
                    return []

                f1_scores = self._single_f1_score(evidence, batch_docs)
                
                if not f1_scores or max(f1_scores) == 0:
                    return []

                best_doc_index = f1_scores.index(max(f1_scores))
                best_doc = batch_docs[best_doc_index]
                
                return [best_doc]

            insufficient_signals = [
                "insufficient information", "not mentioned", "not contain",
                "cannot answer", "does not contain"
            ]
            if any(signal in response.lower() for signal in insufficient_signals):
                return []
                
        except Exception as e:
            logger.warning(f" F1 Batch grounding failed: {e}", exc_info=True)
        
        return []

    def _revise_answer(self, sub_question: str, candidate_answer: str,
                       documents: List[str]) -> str:
        if not documents:
            return candidate_answer
        
        formatted_docs = '\n'.join([str(doc) for doc in documents])
        
        try:
            revise_input_messages = self.prompts["revise"].get_string(
                n=len(documents),
                knowledge=formatted_docs,
                q=sub_question,
                ans=candidate_answer
            )
        except Exception as e:
            logger.error(f"Failed to format revise prompt: {e}")
            return candidate_answer

        try:
            generated = self.generator.generate([revise_input_messages])
            response = generated[0] if isinstance(generated, list) else generated
            
            try:
                parsed = json.loads(response)
                if "answer" in parsed:
                    return parsed["answer"]
            except:
                pass
            
            if response and len(response.strip()) > 0:
                if '{"answer"' in response or "{'answer'" in response:
                    match = re.search(r'"answer"\s*:\s*"([^"]+)"', response)
                    if match:
                        return match.group(1)
                
                return response.strip()
                
        except Exception as e:
            logger.warning(f"     Answer revision failed: {e}", exc_info=True)
        
        return candidate_answer

    def _normalize_answer_for_f1(self, s: str) -> str:
        def remove_articles(text):
            return self.re_art.sub(' ', text)
        def white_space_fix(text):
            return ' '.join(text.split())
        def remove_punc(text):
            return self.re_punc.sub(' ', text)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def _prec_recall_f1_score(self, pred_items: list, gold_items: list) -> Tuple[float, float, float]:
            common = Counter(gold_items) & Counter(pred_items)
            num_same = sum(common.values())
            if num_same == 0:
                return 0.0, 0.0, 0.0
            
            if not pred_items:
                return 0.0, 0.0, 0.0
            if not gold_items:
                return 0.0, 0.0, 0.0

            precision = 1.0 * num_same / len(pred_items)
            recall = 1.0 * num_same / len(gold_items)
            f1 = (2 * precision * recall) / (precision + recall)
            return precision, recall, f1

    def _single_f1_score(self, guess: str, answers: List[str]) -> List[float]:
        if guess is None or answers is None:
            return [0.0] * (len(answers) if answers else 0)
        
        g_tokens = self._normalize_answer_for_f1(guess).split()
        
        scores = []
        for a in answers:
            p, r, f1 = self._prec_recall_f1_score(g_tokens, self._normalize_answer_for_f1(a).split())
            scores.append(f1)
        
        return scores

    def _normalize_answer(self, text: str) -> str:
        text = text.strip()
        
        finish_match = re.search(r'FINISH\[(.*)\]', text, re.IGNORECASE)
        if finish_match:
            text = finish_match.group(1).strip()
            
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = text.rstrip('.')
        
        return text

    def _extract_doc_texts(self, raw_docs: List) -> List[str]:
        texts = []
        if not raw_docs:
            return texts
        
        for doc in raw_docs:
            if isinstance(doc, str):
                texts.append(doc)
            elif isinstance(doc, dict):
                if 'contents' in doc:
                    texts.append(doc['contents'])
                elif 'content' in doc:
                    texts.append(doc['content'])
                elif 'text' in doc:
                    texts.append(doc['text'])
                else:
                    title = doc.get('title', '')
                    content_str = str(doc)
                    
                    if title:
                         content = doc.get('contents', doc.get('text', ''))
                         if content:
                             texts.append(f"{title}\n{content}")
                         else:
                             texts.append(content_str)
                    else:
                         texts.append(content_str)
            else:
                texts.append(str(doc))
        
        return [t for t in texts if t and t.strip()]
