import sys
import json
import pandas as pd
import os
from vllm import LLM, SamplingParams
import string
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
import random
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import re

def write_classification_logs(model_name, reports_file, precision, recall, f1, accuracy, report):
    log_data = (
        f"-------------------------------------------------------------------------\n"
        f"\t\t\t\t\t\t{model_name}\n"
        f"-------------------------------------------------------------------------\n"
        f"Precision: {precision}\tRecall: {recall}\tF1: {f1}\tAccuracy: {accuracy}\n"
        f"-------------------------------------------------------------------------\n"
        f"{report}\n"
        f"-------------------------------------------------------------------------\n"
    )
    with open(reports_file, mode='a', encoding='utf-8') as log_file:
        log_file.write(log_data)

def save_metrics_to_csv(metrics_file_path, model_name, p, r, f1, acc):
    """Save metrics to a CSV file"""
    metrics_data = {
        'Model': model_name,
        'Precision': p,
        'Recall': r, 
        'F1-Score': f1,
        'Accuracy': acc
    }
    
    # Create or append to CSV
    mode = 'a' if os.path.exists(metrics_file_path) else 'w'
    header = mode == 'w'
    
    pd.DataFrame([metrics_data]).to_csv(
        metrics_file_path,
        mode=mode,
        header=header,
        index=False
    )

def form_multiple_choice_prompt(class_labels):
    return '\n'.join([f"{letter}) {label}" for letter, label in zip(string.ascii_lowercase, class_labels)])

def add_examples(num_examples, query_template, examples_method, examples_seed, df_train, system_prompt, labels):
    all_example_pairs = []
    random.seed(examples_seed)
    template_qa = "{sentence}\n### Question: Which is the most likely intent for this citation?\n{options}\n### Answer: "
    template_simple = "{sentence} \nClass: "
    multiple_choice = form_multiple_choice_prompt(labels) if query_template[0] == '2' else None
    prompt = template_qa if query_template[0] == '2' else template_simple

    if examples_method[0] == '1':
        all_example_pairs = [f"\n{prompt.format(sentence=ex, options=multiple_choice if multiple_choice else '')} {label}\n" for label in labels
            for ex in df_train[df_train['citation_class_label'] == label].sample(num_examples, random_state=examples_seed)['citation_context']]

        random.shuffle(all_example_pairs)
        all_example_pairs = ''.join(all_example_pairs)
        system_prompt[0]['content'] += "\n\n########\n\n# EXAMPLES #\n" + all_example_pairs

    elif examples_method[0] == '2':
        all_example_pairs = [
            [{"role": "user", "content": f"{prompt.format(sentence=ex, options=multiple_choice if multiple_choice else '')}"}, {"role": "assistant", "content": label}]
            for label in labels
            for ex in df_train[df_train['citation_class_label'] == label].sample(num_examples, random_state=examples_seed)['citation_context']
        ]
        random.shuffle(all_example_pairs)
        system_prompt.extend([pair for sublist in all_example_pairs for pair in sublist])

    return system_prompt

def clean_prediction(text, dataset):
    """Clean and validate predictions using regex patterns"""
    # Define valid labels and their regex patterns
    if dataset == 'scicite':
        valid_labels = {
            'background information': r'background\b|background information\b',
            'results comparison': r'results?[_\s-]?comparison\b|compar[es]?[_\s-]?results\b',
            'method': r'method\b|methodolog[ies]*\b',
        }
    if dataset == 'acl_arc_scicite_schema':
        valid_labels = {
            'background information': r'background\b|background information\b',
            'result comparison': r'results?[_\s-]?comparison\b|compar[es]?[_\s-]?results\b',
            'method': r'method\b|methodolog[ies]*\b',
        }
    if dataset == 'act2_scicite_schema':
        valid_labels = {
            'background information': r'background\b|background information\b',
            'result comparison': r'results?[_\s-]?comparison\b|compar[es]?[_\s-]?results\b',
            'method': r'method\b|methodolog[ies]*\b',
        }
    else:
        valid_labels = {
            'background': r'background\b',
            'compares_contrasts': r'compare[s]?[_\s-]?contrast[s]?\b',
            'extension': r'extend[s]?\b|extension\b',
            'future': r'future\b',
            'motivation': r'motivat[ion|e]*\b',
            'uses': r'uses\b|use\b',
        }
    
    # Convert to lowercase and strip whitespace
    text = text.lower().strip()
    
    # Try to match each label pattern
    for label, pattern in valid_labels.items():
        if re.search(pattern, text, re.IGNORECASE):
            return label
            
    return "unknown"  # Default fallback

def get_predictions(llm, system_prompt, sentences, query_template, temperature, multiple_choice=None, args=None):
    predicted_classes = []
    template_qa = "{sentence}\n### Question: Which is the most likely intent for this citation? \n{options}\n### Answer:"
    template_simple = "{sentence} \nClass:"
        
    sampling_params = SamplingParams(temperature=temperature, max_tokens=15)
    
    # Create prompts
    prompts = []
    for sentence in tqdm(sentences, desc="Creating prompts"):
        prompt = template_qa if query_template[0] == '2' else template_simple
        message = system_prompt + [{
            "role": "user",
            "content": prompt.format(sentence=sentence, options=multiple_choice if multiple_choice else "")
        }]
        prompt_text = "\n".join([m["content"] for m in message])
        prompts.append(prompt_text)
    
    # Generate and process predictions
    outputs = llm.generate(prompts, sampling_params)
    for output in tqdm(outputs, desc="Processing predictions"):
        try:
            raw_prediction = output.outputs[0].text
            predicted_class = clean_prediction(raw_prediction, args['dataset'])
            predicted_classes.append(predicted_class)
        except Exception as e:
            print(f"Error processing prediction: {e}")
            predicted_classes.append("unknown")
    return predicted_classes

def evaluate(true_labels, predicted_labels):
    try:        
        p, r, f1, sup = precision_recall_fscore_support(true_labels, predicted_labels, average='macro')
        acc = round(accuracy_score(true_labels, predicted_labels), 4)
        report = classification_report(true_labels, predicted_labels, output_dict=False)
        return round(p, 4), round(r, 4), round(f1, 4), round(acc, 4), report
    except:
        return None, None, None, None, None

def run_analysis(args, llm):
    # Load configurations
    with open('experimental-configs/experiments_cfg.json') as f:
        experiment_cfg = json.load(f)
        class_labels = experiment_cfg['class_labels']

    # Load data
    datasets_path = os.path.join('datasets', 'formatted')
    df_test = pd.read_csv(os.path.join(datasets_path, args['dataset'], "test.csv"))
    
    # Create output directories
    os.makedirs('results', exist_ok=True)
    
    system_prompts_cfg = json.loads(Path('experimental-configs/system_prompts.json').read_text())
    prompt_id = f"{args['dataset']}{args['system_prompt']}"
    
    system_prompt = [{
        "role": "system",
        "content": system_prompts_cfg[prompt_id]
    }]
    
    if args['method'] != 'zero-shot':
        df_train = pd.read_csv(os.path.join(datasets_path, args['dataset'], "train.csv"))
        
        system_prompt = add_examples(
            num_examples=args['num_examples'],
            query_template=args['query_template'],
            examples_method=args['example_method'],
            examples_seed=args['examples_seed'],
            df_train=df_train,
            system_prompt=system_prompt,
            labels=class_labels[args['dataset']]
        )
    
    # Run inference and evaluate
    test_sentences = df_test["cite_context_paragraph"].to_list()
    multiple_choice = form_multiple_choice_prompt(class_labels[args['dataset']]) if args['query_template'][0] == '2' else None
    df_test['predicted_classes'] = get_predictions(
        llm=llm,
        system_prompt=system_prompt,
        sentences=test_sentences,
        query_template=args['query_template'],
        temperature=args['temperature'],
        multiple_choice=multiple_choice,
        args=args
    )
    
    # Save results and metrics
    metrics_file_path = os.path.join('results', args['dataset'], args['run_id'], args['method'], "metrics")
    reports_file_path = os.path.join(metrics_file_path, "classification_reports")
    os.makedirs(metrics_file_path, exist_ok=True)
    os.makedirs(reports_file_path, exist_ok=True)
    
    if args['method'] != 'zero-shot':
        metrics_file_template = f"_metrics_{args['dataset']}_{args['method']}_SP{args['system_prompt']}_QT{args['query_template'][0]}_EM{args['example_method'][0]}_T{args['temperature']}_{args['bit_precision']}.csv"
        model_output_file_path = os.path.join('results', args['dataset'], args['run_id'], args['method'], f"SP{args['system_prompt']}", f"QT{args['query_template'][0]}", f"EM{args['example_method'][0]}", f"T{args['temperature']}")
        model_output_file_template = f"{args['model_name']}_{args['dataset']}_{args['method']}_SP{args['system_prompt']}_QT{args['query_template'][0]}_EM{args['example_method'][0]}_T{args['temperature']}_{args['bit_precision']}.csv"
    else:
        metrics_file_template = f"_metrics_{args['dataset']}_{args['method']}_SP{args['system_prompt']}_QT{args['query_template'][0]}_T{args['temperature']}_{args['bit_precision']}.csv"
        model_output_file_path = os.path.join('results', args['dataset'], args['run_id'], args['method'], f"SP{args['system_prompt']}", f"QT{args['query_template'][0]}", f"T{args['temperature']}")
        model_output_file_template = f"{args['model_name']}_{args['dataset']}_{args['method']}_SP{args['system_prompt']}_QT{args['query_template'][0]}_T{args['temperature']}_{args['bit_precision']}.csv"
    
    os.makedirs(model_output_file_path, exist_ok=True)
    df_test.to_csv(os.path.join(model_output_file_path, model_output_file_template), index=False)
    
    # Calculate and save metrics
    p, r, f1, acc, report = evaluate(df_test["citation_class_label"].str.replace('EXTENDS', 'EXTENSION').str.lower().tolist(), df_test["predicted_classes"].tolist())
    write_classification_logs(args['model_name'], os.path.join(reports_file_path, metrics_file_template.replace('csv', 'log')), p, r, f1, acc, report)
    save_metrics_to_csv(os.path.join(metrics_file_path, metrics_file_template), args['model_name'], p, r, f1, acc)

    # Save detailed results CSV
    results_data = {
        'sentence': test_sentences,
        'true_label': df_test["citation_class_label"].str.replace('EXTENDS', 'EXTENSION').str.lower(),
        'predicted_label': df_test["predicted_classes"],
        'correct': df_test["citation_class_label"].str.replace('EXTENDS', 'EXTENSION').str.lower() == df_test["predicted_classes"]
    }
    
    results_csv_path = os.path.join(model_output_file_path, 
                                   model_output_file_template.replace('.csv', '_detailed.csv'))
    pd.DataFrame(results_data).to_csv(results_csv_path, index=False)


if __name__ == "__main__":
    os.environ["SSL_CERT_FILE"] = "~/tls-ca-bundle.pem"
    from tqdm import tqdm
    if len(sys.argv) < 2:
        with open('/nfs/home/tanz/cc/classifiers/CitationIntentOpenLLM/run_args_QWEN2.5_14B_AA_FT.json') as f:
            args = json.load(f)
    elif sys.argv[1].endswith('.json'):
        with open(sys.argv[1]) as f:
            args = json.load(f)
    else:
        args = json.loads(sys.argv[1])
    # Initialize VLLM engine
    llm = LLM(
        args['model_path'],
        max_model_len=args['model_context_length'],
    )
    for run_arg in tqdm(args['args']):
        print(f"Running analysis for: {run_arg}")
        run_arg["label_schema"] = args['label_schema']
        run_analysis(run_arg, llm)
    
    del llm