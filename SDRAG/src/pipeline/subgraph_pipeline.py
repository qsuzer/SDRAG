import logging
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict, deque

from flashrag.prompt import PromptTemplate
from .base_pipeline import (
    BaseGraphPipeline, 
    _parse_decomposed_questions, 
    _parse_dependencies,
    flatten_retrieval_results
)
from .prompts import FINAL_SYNTHESIS_PROMPT

logger = logging.getLogger(__name__)

class QueryGraph:
    """
    A data structure to represent, manipulate, and update the sub-question dependency graph.
    Corresponds to "Graph Construction", "Graph Pruning", and "Graph Updating".
    """

    def __init__(self, sub_questions: List[str]):
        """
        Initialize the graph, where each sub-question is a node.
        """
        self.nodes = {}
        # adj_list: key depends on nodes in value list
        self.adj_list = defaultdict(list)
        # rev_adj_list: key is depended on by nodes in value list
        self.rev_adj_list = defaultdict(list)

        for i, q_text in enumerate(sub_questions):
            self.nodes[i] = {
                "id": i,
                "question": q_text,
                "rewritten_q": None,
                "answer": None,
                "docs": None,
                "pruned": False,
                "runtime_deps": []
            }

    def __len__(self):
        return len([n for n in self.nodes if not self.nodes[n]['pruned']])

    def add_dependencies(self, dependencies_dict: Dict[int, List[int]]):
        """
        Build graph edges based on dependency dictionary.
        """
        for q_idx, deps in dependencies_dict.items():
            if q_idx not in self.nodes: continue
            valid_deps = [d for d in deps if d in self.nodes]
            self.adj_list[q_idx] = valid_deps
            for d in valid_deps:
                self.rev_adj_list[d].append(q_idx)

    def get_leaf_nodes(self) -> List[int]:
        """
        Find all leaf nodes (nodes that no other node depends on).
        Candidates for pruning.
        """
        return [i for i in self.nodes if i not in self.rev_adj_list and not self.nodes[i]['pruned']]

    def prune_nodes(self, nodes_to_prune: Set[int]):
        """
        Mark specified nodes as pruned.
        """
        for i in nodes_to_prune:
            if i in self.nodes:
                self.nodes[i]['pruned'] = True
                logger.info(f"  Pruning node {i}: {self.nodes[i]['question']}")

    def get_topological_sort(self) -> List[int]:
        """
        Perform topological sort (Kahn's algorithm) to get execution path.
        Ignores pruned nodes.
        """
        active_nodes = [i for i in self.nodes if not self.nodes[i]['pruned']]
        num_active_nodes = len(active_nodes)

        in_degree = {i: 0 for i in active_nodes}
        active_graph = defaultdict(list)

        for q_idx in active_nodes:
            valid_deps = [d for d in self.adj_list.get(q_idx, []) if not self.nodes[d]['pruned']]
            self.nodes[q_idx]['runtime_deps'] = valid_deps

            for dep in valid_deps:
                active_graph[dep].append(q_idx)
            in_degree[q_idx] = len(valid_deps)

        queue = deque([i for i in in_degree if in_degree[i] == 0])
        order = []
        while queue:
            current = queue.popleft()
            order.append(current)
            for neighbor in active_graph.get(current, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != num_active_nodes:
            logger.warning("Circular dependency detected. Falling back to sequential order.")
            return [i for i in active_nodes]
        return order

    def get_node(self, node_id: int) -> Dict:
        return self.nodes[node_id]

    def get_dependencies(self, node_id: int) -> List[int]:
        return self.nodes[node_id].get('runtime_deps', [])

    def update_node_data(self, node_id: int, rewritten_q: str, answer: str, docs: List):
        if node_id in self.nodes:
            self.nodes[node_id]['rewritten_q'] = rewritten_q
            self.nodes[node_id]['answer'] = answer
            self.nodes[node_id]['docs'] = docs

    def format_results_by_order(self, order: List[int]) -> str:
        formatted_text = ""
        for i, q_idx in enumerate(order):
            if q_idx in self.nodes:
                node = self.nodes[q_idx]
                formatted_text += f"Sub-question {i + 1} (Original ID: {q_idx + 1}): {node['question']}\n"
                formatted_text += f"Retrieved Information: {node['docs']}\n\n"
        return formatted_text.strip()

def _format_graph_qa_context(graph: QueryGraph, indices: List[int]) -> str:
    context = ""
    for i in indices:
        node = graph.get_node(i)
        if node['answer']:
            context += f"Question {i + 1}: {node['question']}\n"
            context += f"Answer {i + 1}: {node['answer']}\n\n"
    return context.strip()

class SubGraphPipeline(BaseGraphPipeline):
    """
    Process: 1. Graph Construction -> 2. Graph Pruning -> 3. Graph Update (Topo Rewrite+Retrieve) -> 4. Answer Aggregation
    """

    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        super().__init__(config, prompt_template, retriever, generator)
        
        self.prompts["final_synthesis"] = PromptTemplate(
            config, 
            system_prompt=FINAL_SYNTHESIS_PROMPT,
            user_prompt="Original Question: {original_question}\nIntermediate Reasoning Steps:\n{all_steps}"
        )

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        questions = dataset.question

        logger.info("Step 1: Decomposing all questions in a batch...")
        decompose_inputs = [self.prompts["decompose"].get_string(question=q) for q in questions]
        decomposed_texts = self.generator.generate(decompose_inputs)
        all_sub_questions = [_parse_decomposed_questions(text) for text in decomposed_texts]

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
                values = [d.get(key) for d in all_intermediate_steps]
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

    def _process_single_question(self, question: str, sub_questions: List[str], dependencies_dict: Dict) -> Tuple[str, Dict]:
        intermediate_steps = {"decomposed_questions": sub_questions, "dependencies": dependencies_dict}
        if not sub_questions: 
            return "Could not decompose the question.", intermediate_steps

        # --- 1. Graph Construction ---
        graph = QueryGraph(sub_questions)
        graph.add_dependencies(dependencies_dict)
        intermediate_steps["initial_graph_nodes"] = len(graph.nodes)

        # --- 2. Graph Pruning ---
        leaf_nodes = graph.get_leaf_nodes()
        nodes_to_prune = set()
        if leaf_nodes:
            pruning_inputs = [
                self.prompts["pruning"].get_string(
                    original_question=question,
                    sub_question=graph.get_node(i)['question']
                ) for i in leaf_nodes
            ]
            pruning_decisions = self.generator.generate(pruning_inputs)
            
            for leaf_node_idx, decision in zip(leaf_nodes, pruning_decisions):
                if "PRUNE" in decision.upper():
                    nodes_to_prune.add(leaf_node_idx)

        graph.prune_nodes(nodes_to_prune)
        intermediate_steps["pruned_graph_nodes"] = len(graph)
        intermediate_steps["pruned_node_ids"] = list(nodes_to_prune)

        if len(graph) == 0:
            return "All sub-questions were pruned.", intermediate_steps

        # --- 3. Graph Updating ---
        processing_order = graph.get_topological_sort()
        intermediate_steps["processing_order"] = processing_order

        for i in processing_order:
            node = graph.get_node(i)
            sub_q = node['question']
            rewritten_q = sub_q

            # a. Rewriting based on dependencies
            dep_indices = graph.get_dependencies(i)
            if dep_indices:
                context = _format_graph_qa_context(graph, dep_indices)
                if context:
                    rewritten_q = self.generator.generate([
                        self.prompts["rewrite"].get_string(previous_qa=context, sub_question=sub_q)
                    ])[0]

            # b. Retrieval
            results, _ = self.retriever.search(rewritten_q, return_score=True)

            # c. Sub-answering
            sub_ans = self.generator.generate([
                self.prompts["sub_answer"].get_string(sub_question=rewritten_q, retrieval_results=str(results))
            ])[0]

            # d. Update node state
            graph.update_node_data(i, rewritten_q=rewritten_q, answer=sub_ans, docs=results)

        intermediate_steps["graph_nodes_data"] = graph.nodes

        all_retrieval_results = []
        for node_id in graph.nodes:
            node = graph.nodes[node_id]
            if node.get('docs'):
                all_retrieval_results.append(node['docs'])
        intermediate_steps["sub_retrieval_results"] = all_retrieval_results

        # --- 4. Answer Composition ---
        all_steps_text = self._format_graph_steps_for_synthesis(graph, processing_order)
        
        compose_input = self.prompts["final_synthesis"].get_string(
            original_question=question,
            all_steps=all_steps_text
        )
        intermediate_steps["prompt"] = compose_input

        final_answer = self.generator.generate([compose_input])[0]
        
        return final_answer, intermediate_steps

    def _format_graph_steps_for_synthesis(self, graph: QueryGraph, order: List[int]) -> str:
        output = ""
        for step_i, node_id in enumerate(order):
            node = graph.get_node(node_id)
            output += f"--- Step {step_i + 1} ---\n"
            output += f"Sub-question: {node['rewritten_q']} (Original: {node['question']})\n"
            output += f"Answer: {node['answer']}\n"
            output += f"Retrieved Info: {str(node['docs'])[:500]}...\n\n"
        return output.strip()
