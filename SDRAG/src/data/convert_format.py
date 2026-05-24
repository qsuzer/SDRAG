import json
import re
import os
import argparse

def extract_content(text, start_marker, end_marker=None):
    """Helper to extract content between markers."""
    try:
        start_idx = text.find(start_marker)
        if start_idx == -1:
            return None
        start_idx += len(start_marker)
        
        if end_marker:
            end_idx = text.find(end_marker, start_idx)
            if end_idx == -1:
                return text[start_idx:].strip()
            return text[start_idx:end_idx].strip()
        else:
            return text[start_idx:].strip()
    except Exception:
        return None

def convert_to_sharegpt(input_file, output_file):
    """Converts OpenAI format to ShareGPT/ms-swift format if needed."""
    # Note: ms-swift supports standard OpenAI format (messages list) directly in many cases.
    # This script is a placeholder if specific formatting is needed.
    # For now, we assume the output of generate_data.py (ChatML style "messages") is sufficient.
    
    print(f"Converting {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
        
    # If any specific conversion is needed, do it here.
    # For example, filtering invalid samples.
    
    valid_data = [d for d in data if d.get('messages')]
    
    print(f"Saved {len(valid_data)} samples to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in valid_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    
    convert_to_sharegpt(args.input_file, args.output_file)
