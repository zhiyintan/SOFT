import os
import json
from pathlib import Path
import subprocess
import sys
import logging
from datetime import datetime
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'experiment_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

def create_metrics_file(metrics_file_template, metrics_file_path, reports_file_path):
    os.makedirs(metrics_file_path, exist_ok=True)
    os.makedirs(reports_file_path, exist_ok=True)
    metrics_file_path = os.path.join(metrics_file_path, metrics_file_template)
    with open(metrics_file_path, mode='w+', encoding='utf-8') as fl:
        fl.write("Model,Precision,Recall,F1-Score,Accuracy\n")
    return metrics_file_path

def run_inference(run_args):
    try:
        env = os.environ.copy()
        env.update({
            "SSL_CERT_FILE": "~/tls-ca-bundle.pem"
        })

        # logging.info(f"Starting inference process with args: {json.dumps(run_args, indent=2)}")
        
        result = subprocess.run(
            [sys.executable, "evaluation_runner.py", json.dumps(run_args)],
            capture_output=True,
            text=True,
            env=env
        )
        
        # Log the full output for debugging
        if result.stdout:
            logging.info(f"Process stdout:\n{result.stdout}")
        if result.stderr:
            logging.error(f"Process stderr:\n{result.stderr}")
            
        if result.returncode != 0:
            logging.error(f"Error running inference for {run_args['model_name']}")
            logging.error(f"Return code: {result.returncode}")
            return False
            
        return True
        
    except subprocess.CalledProcessError as e:
        logging.error(f"Process error: {str(e)}")
        logging.error(f"Command output: {e.output}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        logging.exception("Full traceback:")
        return False

if __name__ == "__main__":
    experiment_cfg_path = os.path.join('experimental-configs', 'experiments_cfg.json')
    experiment_cfg = json.loads(Path(experiment_cfg_path).read_text())

    methods = experiment_cfg['methods']
    num_examples = experiment_cfg['number_of_examples'] 
    examples_seed = experiment_cfg['examples_seed']
    datasets = experiment_cfg['datasets']
    system_prompts = experiment_cfg['system_prompts']
    class_labels = experiment_cfg['class_labels']
    examples_methods = experiment_cfg['examples_methods']
    query_templates = experiment_cfg['query_templates']
    temperature = experiment_cfg['temperature']
    bit_precision = experiment_cfg['bit_precision']
    run_id = experiment_cfg['run_id']
    finetuned = experiment_cfg['finetuned']

    if finetuned:
        models_cfg_path = os.path.join('experimental-configs', f"models.{bit_precision}.ft.json")
    else:
        models_cfg_path = os.path.join('experimental-configs', f'models.{bit_precision}.json')
    models_cfg = json.loads(Path(models_cfg_path).read_text())
    
    system_prompts_cfg_path = os.path.join('experimental-configs', 'system_prompts.json')
    system_prompts_cfg = json.loads(Path(system_prompts_cfg_path).read_text())

    datasets_path = os.path.join('datasets', 'formatted')
    
    max_retries = 3
    failed_runs = []
    retry_runs = []
    for model_name, model_metadata in models_cfg.items():
        logging.info(f"Processing model: {model_name}")
        run_args_list = {
            "model_name": model_name,
            "model_path": model_metadata['path'],
            "model_context_length": model_metadata['context_length'],
            "label_schema": model_metadata['label_schema'],
            "args": [] 
        }
        for dataset in datasets:
            logging.info(f"Processing dataset: {dataset}")
            for method in methods:
                for sp in system_prompts:
                    if method != 'zero-shot':
                        for em in examples_methods:
                            for qt in query_templates:
                                for temp in temperature:
                                    run_args = {
                                        "dataset": dataset,
                                        "method": method,
                                        "system_prompt": sp,
                                        "example_method": em,
                                        "query_template": qt,
                                        "temperature": temp,
                                        "model_name": model_name,
                                        #"model_path": model_metadata['path'],
                                        #"model_context_length": model_metadata['context_length'],
                                        "bit_precision": bit_precision,
                                        "run_id": run_id,
                                        "num_examples": num_examples[method],
                                        "examples_seed": examples_seed
                                    }
                                run_args_list['args'].append(run_args)
                    else:
                        for qt in query_templates:
                            for temp in temperature:
                                run_args = {
                                    "dataset": dataset,
                                    "method": method, 
                                    "system_prompt": sp,
                                    "query_template": qt,
                                    "temperature": temp,
                                    "model_name": model_name,
                                    #"model_path": model_metadata['path'],
                                    #"model_context_length": model_metadata['context_length'],
                                    "bit_precision": bit_precision,
                                    "run_id": run_id
                                }
                                run_args_list['args'].append(run_args)
        json.dump(run_args_list, open(f'run_args_{model_name}.json', 'w'), indent=2)
        continue

        retry_count = 0
        success = False
        
        while retry_count < max_retries and not success:
            if retry_count > 0:
                logging.info(f"Retry {retry_count} for {model_name}")
            logging.info(f"Running inference for {model_name}")
            success = run_inference(run_args_list)
            if not success:
                retry_count += 1
                # Add small delay between retries
                time.sleep(5)
        
        if not success:
            failed_runs.append(run_args_list)
        elif retry_count > 0:
            retry_runs.append({
                "run_args": run_args_list,
                "retries": retry_count
            })

    # Log summary

    if retry_runs:
        logging.info("Successful retries:")
        for run in retry_runs:
            logging.info(f"Model: {run['run_args']['model_name']}, Retries: {run['retries']}")
    
    if failed_runs:
        logging.error("Failed runs after all retries:")
        for run in failed_runs:
            logging.error(f"Model: {run['model_name']}")
        
        # Save failed runs for potential retry
        with open(f'failed_runs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
            json.dump(failed_runs, f, indent=2)