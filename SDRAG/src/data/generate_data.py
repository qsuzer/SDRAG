import json
import re
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI

# ================= Configuration =================
# Use environment variables for sensitive info
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4")

MAX_WORKERS = 5
# =============================================

if not API_KEY:
    print("Warning: OPENAI_API_KEY environment variable not set.")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# === Prompts ===
DECOMPOSE_SYSTEM_PROMPT = "You are a helpful and logical assistant expert in query decomposition."
DECOMPOSE_USER_TEMPLATE = """You are an expert at breaking down complex questions into simpler sub-questions.

Your task is to decompose the given complex question into a logical sequence of sub-questions.

### Guidelines:
1. **Analyze First**: Before generating questions, explain the logic required to answer the original question.
2. **Simplicity**: Sub-questions must be simpler than the original.
3. **Dependency**: Ensure they build upon each other logically.
4. **Completeness**: Combined, they must fully answer the original question.
5. **Constraint**: Keep the number of sub-questions typically between 2 and 4.

### Output Format:
Thought: <Reasoning regarding dependencies and steps>
Sub-questions:
1. <sub-question 1>
2. <sub-question 2>
...

### Input:
Complex Question: {question}
"""

DEPENDENCY_SYS_PROMPT_COT = """You are an expert at analyzing logical dependencies between questions.

Your task is to identify DIRECT dependencies based on:
1. Reference (pronouns referring to previous answers)
2. Entailment (explicitly requiring information from previous questions)

Format your response strictly as follows:
Thought: <Analyze the logical links step-by-step>
JSON:
```json
{
    "1": [],
    "2": [1],
    ...
}
```"""

PRUNING_SYS_PROMPT_COT = """You are an expert at identifying unnecessary sub-questions (Graph Pruning).

Determine if a leaf sub-question should be kept or pruned.
- PRUNE if: Irrelevant details, duplicate info, or not needed for the original question.
- KEEP if: Directly needed answer or identifies a key entity.

Format your response strictly as follows:
Thought: <Briefly analyze why this question is needed or not>
Decision: <KEEP or PRUNE>"""


def generate_decomposition_data(item):
    """Generates decomposition training data."""
    question = item.get('question', '')
    if not question: return None
    
    prompt = DECOMPOSE_USER_TEMPLATE.format(question=question)
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0, 
            max_tokens=512
        )
        content = response.choices[0].message.content.strip()
        
        if "Thought:" not in content or "Sub-questions:" not in content:
            return None
            
        return {
            "messages": [
                {"role": "system", "content": "You are an expert planner."},
                {"role": "user", "content": f"Task: Decompose the complex question.\nComplex Question: {question}\n\nProvide a 'Thought' explaining the plan, followed by 'Sub-questions' as a numbered list.\nResponse:"},
                {"role": "assistant", "content": content}
            ],
            "metadata": item
        }
    except Exception as e:
        print(f"Error in decomposition: {e}")
        return None

def generate_dependency_data(item):
    """Generates dependency analysis training data."""
    # Assumes item already has sub-questions (e.g. from decomposition step)
    # This is a simplified placeholder. In a real pipeline, you'd chain these.
    # For this reproduction script, we assume the input item has 'sub_questions' list.
    sub_qs = item.get('sub_questions', [])
    if not sub_qs: return None

    formatted_subs = "\n".join([f"{i+1}. {q}" for i, q in enumerate(sub_qs)])
    real_user_input = f"Sub-questions:\n{formatted_subs}"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": DEPENDENCY_SYS_PROMPT_COT},
                {"role": "user", "content": real_user_input}
            ],
            temperature=0,
            max_tokens=512
        )
        content = response.choices[0].message.content.strip()
        
        if "```json" not in content and "{" not in content:
            return None

        return {
            "messages": [
                {"role": "system", "content": "You are an expert at analyzing logical dependencies between questions."},
                {"role": "user", "content": f"Sub-questions:\n{formatted_subs}"},
                {"role": "assistant", "content": content}
            ]
        }
    except Exception as e:
        print(f"Error in dependency: {e}")
        return None

def process_file(input_file, output_file, task_type):
    print(f"Processing {input_file} for task: {task_type}...")
    results = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        # Handle both jsonl and json list
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            data = json.load(f)
        else:
            data = [json.loads(line) for line in f]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for item in data:
            if task_type == 'decompose':
                futures.append(executor.submit(generate_decomposition_data, item))
            elif task_type == 'dependency':
                futures.append(executor.submit(generate_dependency_data, item))
            # Add pruning logic if needed
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            res = future.result()
            if res:
                results.append(res)

    print(f"Generated {len(results)} samples. Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Input JSON/JSONL file")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--task", type=str, choices=['decompose', 'dependency'], required=True, help="Task type")
    args = parser.parse_args()

    process_file(args.input_file, args.output_file, args.task)
