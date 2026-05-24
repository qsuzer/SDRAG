import argparse
import sys
import os
import logging

# Add FlashRAG to path if not installed
sys.path.append(os.path.abspath("../FlashRAG"))

from flashrag.config import Config
from flashrag.utils import get_dataset, get_retriever, get_generator
from src.pipeline import (
    SubGraphPipeline,
    DualRAGPipeline,
    PERQARAGPipeline,
    GenGroundPipeline,
    IRCoTPipeline,
    RAGPipeline,
    DirectLLMPipeline
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PIPELINE_MAP = {
    "subgraph": SubGraphPipeline,
    "dualrag": DualRAGPipeline,
    "perqa": PERQARAGPipeline,
    "genground": GenGroundPipeline,
    "ircot": IRCoTPipeline,
    "rag": RAGPipeline,
    "direct": DirectLLMPipeline
}

def main():
    parser = argparse.ArgumentParser(description="Run SDRAG pipelines")
    parser.add_argument("--pipeline", type=str, required=True, choices=PIPELINE_MAP.keys(), help="Pipeline to run")
    parser.add_argument("--config", type=str, default="config/basic_config.yaml", help="Path to config file")
    parser.add_argument("--dataset", type=str, help="Dataset name (overrides config)")
    parser.add_argument("--split", type=str, help="Dataset split (overrides config)")
    parser.add_argument("--gpu_id", type=str, help="GPU IDs (overrides config)")
    parser.add_argument("--test_sample_num", type=int, help="Number of samples to test")
    
    args = parser.parse_args()
    
    # Load config
    config_dict = {}
    if args.dataset:
        config_dict["dataset_name"] = args.dataset
    if args.split:
        config_dict["split"] = [args.split]
    if args.gpu_id:
        config_dict["gpu_id"] = args.gpu_id
    if args.test_sample_num is not None:
        config_dict["test_sample_num"] = args.test_sample_num
        
    config_dict["save_note"] = args.pipeline
        
    logger.info(f"Loading config from {args.config}")
    config = Config(args.config, config_dict)
    
    # Load dataset
    logger.info(f"Loading dataset {config['dataset_name']}...")
    all_split = get_dataset(config)
    # Use the first split specified
    split_name = config['split'][0]
    dataset = all_split[split_name]
    
    logger.info(f"Loaded {len(dataset)} samples from {split_name} split")
    
    # Initialize Pipeline
    pipeline_cls = PIPELINE_MAP[args.pipeline]
    logger.info(f"Initializing {pipeline_cls.__name__}...")
    
    # Most pipelines need retriever and generator, but DirectLLM might not need retriever
    # BaseActivePipeline handles initialization of retriever/generator if passed as None
    # but we can also initialize them explicitly here if needed.
    # For now, we let the pipeline class handle it via super().__init__ which calls get_retriever/get_generator
    
    pipeline = pipeline_cls(config)
    
    # Run Pipeline
    logger.info("Running pipeline...")
    result = pipeline.run(dataset)
    
    logger.info("Pipeline execution completed.")
    print(f"Results saved to: {config['save_dir']}")

if __name__ == "__main__":
    main()
