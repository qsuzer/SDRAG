import argparse
import os
import sys
from flashrag.config import Config
from flashrag.utils import get_dataset

# Add src to path to import custom pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from pipeline.subgraph_pipeline import SubGraphPipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="2wikimultihopqa")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--config_path", type=str, default="config/default_config.yaml")
    parser.add_argument("--context_strategy", type=str, default="topological", 
                        choices=['hierarchical', 'topological', 'role_based'])
    args = parser.parse_args()

    # Load Config
    config_dict = {
        "gpu_id": args.gpu_id,
        "dataset_name": args.dataset_name,
        "save_note": f"sdrag_{args.context_strategy}"
    }
    
    if os.path.exists(args.config_path):
        config = Config(args.config_path, config_dict)
    else:
        print(f"Config file {args.config_path} not found. Using default settings.")
        # You might want to provide a minimal default config here or rely on flashrag defaults
        config = Config("flashrag/config/basic_config.yaml", config_dict)

    # Load Dataset
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    # Initialize Pipeline
    pipeline = SubGraphPipeline(config)

    # Run
    print(f"Running SDRAG on {args.dataset_name} ({args.split})...")
    result = pipeline.run(test_data)
    
    print("Done!")

if __name__ == "__main__":
    main()
