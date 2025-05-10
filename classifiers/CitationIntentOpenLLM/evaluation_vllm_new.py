import sys
import json
import pandas as pd
import os
from vllm import LLM, SamplingParams
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import re
from transformers import AutoTokenizer

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

def clean_prediction(text, dataset):
    """
    Extracts citation_object and citation_function from LLM output, tolerating missing quotes and extra whitespace.
    Maps to canonical label forms using object_map and function_map.
    Returns a combined label string for evaluation.
    """

    object_map = {
        "performed work": "performed work",
        "discovery": "discovery",
        "produced resource": "produced resource"
    }
    function_map = {
        "contextualize": "contextualize",
        "signal gap": "signal gap",
        "highlight limitation": "highlight limitation",
        "justify design choice": "justify design choice",
        "use": "use",
        "modify": "modify",
        "evaluate against": "evaluate against"
    }

    def extract_citation_object(text):
        match = re.search(r'"?citation_object"?\s*:\s*"?([A-Za-z\s\-]+)"?', text, re.IGNORECASE)
        return match.group(1).strip().lower() if match else None

    def extract_citation_function(text):
        match = re.search(r'"?citation_function"?\s*:\s*"?([A-Za-z\s\-]+)"?', text, re.IGNORECASE)
        return match.group(1).strip().lower() if match else None

    obj = extract_citation_object(text)
    func = extract_citation_function(text)

    obj = object_map.get(obj, "unknown")
    func = function_map.get(func, "unknown")

    return obj, func


def get_predictions(llm, system_prompt, sentences, temperature, tokenizer, args=None):
    predicted_classes = []
       
    sampling_params = SamplingParams(temperature=temperature, max_tokens=256)
    
    # Create prompts
    prompts = []
    for sentence in tqdm(sentences, desc="Creating prompts"):
        message = system_prompt + [{
            "role": "user",
            "content": f"{sentence}\n### Instruction: Classify the citation based on the two dimensions: Citation Object and Citation Function. Provide the answer as a JSON object with keys 'citation_object' and 'citation_function'.",
        }]
        prompt_text = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True) + '\n{'
        prompts.append(prompt_text)
    
    # Generate and process predictions
    outputs = llm.generate(prompts, sampling_params)
    for output in tqdm(outputs, desc="Processing predictions"):
        try:
            raw_prediction = '{' + output.outputs[0].text
            obj, func = clean_prediction(raw_prediction, args['dataset'])
            predicted_classes.append((obj, func))
        except Exception as e:
            print(f"Error processing prediction: {e}")
            predicted_classes.append(("unknown", "unknown"))
    return predicted_classes


def evaluate(true_labels, predicted_labels):
    try:        
        p, r, f1, sup = precision_recall_fscore_support(true_labels, predicted_labels, average='macro')
        acc = round(accuracy_score(true_labels, predicted_labels), 4)
        report = classification_report(true_labels, predicted_labels, output_dict=False)
        return round(p, 4), round(r, 4), round(f1, 4), round(acc, 4), report
    except:
        return None, None, None, None, None


def run_analysis(args, llm, tokenizer):
    # Load data
    datasets_path = os.path.join('datasets', 'formatted')
    df_test = pd.read_csv(os.path.join(datasets_path, args['dataset'], "test.csv"))
    
    # Create output directories
    os.makedirs('results', exist_ok=True)
    
    system_prompt = [{
        "role": "system",
        "content": "# CONTEXT #\nYou are an expert researcher tasked with classifying the object and function of a citation in a scientific publication based on a two-dimensional schema.\n\n########\n\n# OBJECTIVE #\nYou will be given a sentence containing a citation marked with [CITED_AUTHOR]. Your task is to classify this citation along two dimensions:\n1.  **Citation Object:** What contribution from the cited work is being referenced?\n2.  **Citation Function:** Why is the cited work being referenced by the citing work?\nYou must assign exactly one class from each dimension.\n\n########\n\n# DIMENSION DEFINITIONS #\n\n## Dimension 1: Citation Object \u2014 What is being cited? ##\nDetermine what contribution from the cited work is being referenced.\n\nPossible Classes for Dimension 1:\n1 - Performed Work: The citing work references the actions, processes, or events the cited work carried out, without focusing on specific reusable outputs.\n    - Ask: Is the citing author referencing \u201cwhat the cited work has done\u201d?\n    - Example: \"[CITED_AUTHOR] developed a new method for clustering.\"\n2 - Discovery: The citing work references observations, findings, conclusions, or insights drawn by the cited work.\n    - Ask: Is the citing author referencing something the cited work discovered or proved?\n    - Examples: \u201c[CITED_AUTHOR] found that MI outperforms frequency.\u201d \u201c[CITED_AUTHOR] observed cross-domain variation.\u201d\n3 - Produced Resource: The citing work references a specific, reusable output from the cited work, including but not limited to: Data (Datasets, Lexicons), Method (Models, Algorithms, Formula, Feature Sets, Curated Input Examples, Experimental Setup, Evaluation Metrics, Benchmarks, Frameworks, Architectures), Tool (Software, Source Code, Libraries, Platforms, Systems).\n    - Ask: Is the citing author referring to a concrete artifact, method, or standard created by the cited work?\n    - Examples: \u201cWe use [CITED_AUTHOR]\u2019s parser.\u201d\n\n## Dimension 2: Citation Function \u2014 Why is it cited? ##\nDetermine why the cited work is being referenced by the citing work.\n\nPossible Classes for Dimension 2:\n1 - Contextualize: The citing work uses the cited work to provide background information, describe prior research, or acknowledge related work. The citation serves to frame the current study within the existing body of knowledge.\n    - Ask: Is the citation being used to describe prior work without affecting the citing author\u2019s approach?\n    - Example: \"Previous work by [CITED_AUTHOR] describes the process of feature extraction.\"\n2 - Signal Gap: The citing work references the cited work to highlight an unresolved issue or area where further research is needed. The citing work or the cited work acknowledges that something (not yet known the outcome) can be done, usually forward-looking.\n    - Ask: Is the citation used to show what hasn\u2019t been done yet? Or something the citing paper will do in the future?\n    - Example: \"[CITED_AUTHOR] noted that tuning is needed.\", \"Despite advancements in [CITED_AUTHOR], tuning is needed.\"\n3 - Highlight Limitation: The citing work or the cited work point out flaws, weaknesses, or constraints that have been recognized in the cited work, usually backward-looking.\n    - Ask: Is the citing paper critiquing the cited work?\n    - Example: \"[CITED_AUTHOR]\u2019s method fails to account for dynamic shifts in the data.\"\n4 - Justify Design Choice: The citing work references the cited work to support or substantiate its own methodological or conceptual decisions, demonstrating that the approach or theory adopted by the citing work is grounded in established research. This can involve using the cited work to justify a design decision or to support a theoretical perspective in the citing work, without explicitly applying the resource from cited work directly.\n    - Ask: Is the cited work being used to defend or justify a choice the citing authors made, but without directly applying the cited work\u2019s resource (e.g., method, setting, model)?\n    - Example: \"[CITED_AUTHOR] shows this setting helps improve the accuracy on noisy data, thus we adopt this setting.\", \u201cFollowing [CITED_AUTHOR] , we select these features.\u201d\n5 - Use: The citing work directly applies a resource from the cited work\u2019s own research. The citing work leverages or incorporates the cited work\u2019s resource into its own work. These resource including but not limited to: Data (Datasets, Lexicons), Method (Models, Algorithms, Formula, Feature Sets, Curated Input Examples, Experimental Setup, Evaluation Metrics, Benchmarks, Frameworks, Architectures, annotation schema, rules, restrict a scale), Tool (Software, Source Code, Libraries, Platforms, Systems).\n    - Ask: Is the cited contribution being used or applied in the current work?\n    - Example: \"We use [CITED_AUTHOR]\u2019s sentiment analysis tool to classify the dataset.\", \u201cBase on [CITED_AUTHOR], we\u2026\u201d, \u201cWe follow [CITED_AUTHOR]\u2019\u2019s setting\u2026\u201d\n6 - Modify: The citing work builds upon or extends the cited work, adjusting or enhancing the original method, model, or idea to better fit its own context or to improve upon it.\n    - Ask: Is the cited contribution being used or applied in the current work after modifying or rebuilding?\n    - Example: \"We use a variation algorithm of what has been used in [CITED_AUTHOR].\"\n7 - Evaluate Against: The citing work references the cited work as a benchmark or point of result comparison, typically to test or evaluate the performance of its own approach or results.\n    - Ask: Is the citing author comparing their results against those of the cited work?\n    - Example: \"We evaluate our model's performance against [CITED_AUTHOR]\u2019s approach to demonstrate its superiority.\", \u201cWe achieve better accuracy than [CITED_AUTHOR]\u201d\n\n########\n\n# RESPONSE RULES #\n- Analyze only the citation marked with the [CITED_AUTHOR] tag within the provided sentence.\n- Assign exactly one class for Dimension 1 (Citation Object) from [\"Performed Work\", \"Discovery\", \"Produced Resource\"].\n- Assign exactly one class for Dimension 2 (Citation Function) from [\"Contextualize\", \"Signal Gap\", \"Highlight Limitation\", \"Justify Design Choice\", \"Use\", \"Modify\", \"Evaluate Against\"].\n- Respond with a JSON object containing two keys: \"citation_object\" and \"citation_function\". The values should be the exact names of the chosen classes.\n- Example JSON Response:\n  {\n    \"citation_object\": \"Produced Resource\",\n    \"citation_function\": \"Use\"\n  }\n- Do not provide any explanation, elaboration, or any text other than the JSON response.\n\n########",
    }]
    
    # Run inference and evaluate
    test_sentences = df_test["cite_context_paragraph"].to_list()
    predicted_classes = get_predictions(
        llm=llm,
        system_prompt=system_prompt,
        sentences=test_sentences,
        temperature=args['temperature'],
        tokenizer=tokenizer,
        args=args
    )
    df_test['predicted_classes'] = predicted_classes
    
    # Save results and metrics
    metrics_file_path = os.path.join('results', args['dataset'], args['run_id'], "metrics")
    reports_file_path = os.path.join(metrics_file_path, "classification_reports")
    os.makedirs(metrics_file_path, exist_ok=True)
    os.makedirs(reports_file_path, exist_ok=True)
    
    model_output_file_path = os.path.join('results', args['dataset'], args['run_id'], f"T{args['temperature']}")
    os.makedirs(model_output_file_path, exist_ok=True)
    model_output_file_template = f"{args['model_name']}_{args['dataset']}_T{args['temperature']}_{args['bit_precision']}.csv"
    df_test.to_csv(os.path.join(model_output_file_path, model_output_file_template), index=False)

    predictions = {
        "citation_object": [p[0] for p in predicted_classes],
        "citation_function": [p[1] for p in predicted_classes],
    }
    for predicted_class in ["citation_object", "citation_function"]:
        metrics_file_template = f"_metrics_{args['dataset']}_T{args['temperature']}_{args['bit_precision']}_{predicted_class}.csv"    

        # Calculate and save metrics
        p, r, f1, acc, report = evaluate(df_test[predicted_class].str.lower().tolist(), predictions[predicted_class])
        write_classification_logs(args['model_name'], os.path.join(reports_file_path, metrics_file_template.replace('csv', 'log')), p, r, f1, acc, report)
        save_metrics_to_csv(os.path.join(metrics_file_path, metrics_file_template), args['model_name'], p, r, f1, acc)

    # Save detailed results CSV
    results_data = {
        'sentence': test_sentences,
        'citation_object': df_test["citation_object"].str.lower().to_list(),
        'predicted_citation_object': predictions["citation_object"],
        'citation_function': df_test["citation_function"].str.lower().to_list(),
        'predicted_citation_function': predictions["citation_function"],
    }
    
    results_csv_path = os.path.join(model_output_file_path, 
                                   model_output_file_template.replace('.csv', '_detailed.csv'))
    pd.DataFrame(results_data).to_csv(results_csv_path, index=False)


if __name__ == "__main__":
    os.environ["SSL_CERT_FILE"] = "~/tls-ca-bundle.pem"
    from tqdm import tqdm
    if sys.argv[1].endswith('.json'):
        with open(sys.argv[1]) as f:
            args = json.load(f)
    else:
        args = json.loads(sys.argv[1])
    # Initialize VLLM engine
    llm = LLM(
        args['model_path'],
        max_model_len=args['model_context_length'],
    )
    tokenizer = AutoTokenizer.from_pretrained(args['model_path'])
    for run_arg in tqdm(args['args']):
        print(f"Running analysis for: {run_arg}")
        run_arg["label_schema"] = args['label_schema']
        run_analysis(run_arg, llm, tokenizer)
    
    del llm