import string, random
import os
import pandas as pd
import json
from pathlib import Path

os.chdir(os.path.dirname(os.path.abspath(__file__)))
args_list = json.load(open('run_args_QWEN2.5_14B_AA_FT.json'))

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
        print(labels)
        all_example_pairs = [
            f"\n{prompt.format(sentence=ex, options=multiple_choice if multiple_choice else '')} {label}\n" 
            for label in labels
            for ex in df_train[df_train['citation_class_label'] == label].sample(num_examples, random_state=examples_seed)['citation_context']
        ]

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

for args in args_list['args']:
    args['label_schema'] = args_list['label_schema']
    datasets_path = os.path.join('datasets', 'formatted')
    system_prompts_cfg = json.loads(Path('experimental-configs/system_prompts.json').read_text())
    prompt_id = f"{args['dataset']}{args['system_prompt']}"
    system_prompt = [{
            "role": "system",
            "content": system_prompts_cfg[prompt_id]
        }]
    df_train = pd.read_csv(os.path.join(datasets_path, args['dataset'], "train.csv"))
    with open('experimental-configs/experiments_cfg.json') as f:
            experiment_cfg = json.load(f)
            class_labels = experiment_cfg['class_labels']
    if args['method'] != 'zero-shot':
        add_examples(
            num_examples=args['num_examples'],
            query_template=args['query_template'],
            examples_method=args['example_method'],
            examples_seed=args['examples_seed'],
            df_train=df_train,
            system_prompt=system_prompt,
            labels=class_labels[args['dataset']]
        )