from vllm import LLM, SamplingParams
import pandas as pd
import json
from pathlib import Path
import argparse

# Define the new prompt generation function
# Define the new prompt generation function
def scicite_prompt(citing_context, citing_title, cited_title, cited_abstract, citing_section, model_type: str) -> str:
    """
    Generates the prompt for citation intent classification, handling potentially missing abstracts.

    Args:
        citing_context: The sentence/paragraph containing the citation marker.
        citing_title: The title of the paper making the citation.
        cited_title: The title of the paper being cited.
        cited_abstract: The abstract of the paper being cited (can be empty string or None).
        model_type: The type of LLM ('llama', 'gemma', 'mistral', 'qwen') for formatting.

    Returns:
        The formatted prompt string specific to the model type.
    """
    # Handle potentially missing abstract for prompt display
    # Use "N/A" if abstract is None, NaN, or an empty/whitespace string

    # Shared prompt core - uses f-string interpolation later
    prompt_core = f'''
You are an expert in academic writing tasked with determining why each reference is cited in a paper. Use the following six predefined intents:
(1) Motivation: Use this label when the citation is invoked to highlight a gap, limitation, or open problem in the research area that the current paper aims to address.
(2) Compare: Use this label when the citation serves purely for comparison—e.g. to contrast approaches, results, findings, benchmarks, or theoretical positions. often appear with a couple of other citation, use expression like "most", "most of...while some of".
(3) Uses: Use this label when the current paper directly employs elements from the cited work (such as data sets, algorithms, tools, evaluation metrics, code, or experimental setups). often use expression like '(also) make use of', 'use' in the context.
(4) Extends: Use this label when the current paper builds upon or modifies the cited work's data, methods, theories, or frameworks, rather than simply making a comparison. often use expression like 'following...we', 'we exploit'.
(5) Future: Use this label when the citation is mentioned as a potential avenue for future work—i.e. suggesting that methods, data, or ideas from the cited resource could be adopted later.
(6) Background: Use this label if none of the above apply. In this case, the citation simply provides general background (definitions, standard techniques, prior findings) without directly shaping the paper's approach. ofter use expression without verbs.

For each citation you receive: 
(1) 'citing_context': The sentence(s) surrounding the citation marker (e.g. "...[CITED_AUTHOR]…”).
(2) 'citing_section': Section name where the citation appears (e.g. “Introduction,” “Related Work,” “Discussion”).
(3) 'citing_title': Title of the paper in which the citation occurs.
(4) 'cited_title': Title of the work being cited (or “N/A” if unavailable).

Section-Intent Heuristics
Even if the section name isn't given, you can often infer it from the content. Generally, citations in an Introduction flag Motivation (gaps/problems), Extends(one research is extened by another research), or Background (definitions); in Related Work they Compare prior methods (sometimes noting Uses or Extends); Methods/Experimental Setup passages usually signal Uses (adopted data, tools, metrics), Extends (modified techniques), or occasionally a Comparable existing methods, occasionally a Motivation to update a method, occasionally a backgroud of method; Results/Discussion paragraphs tend to Compare results or highlight how the new work Extends earlier findings; and in a Conclusion/Future Work you'll see Future citations (or occasionally a final Motivation or Compare) pointing toward follow-on studies.

Annotation Procedure:
1. Read the title of the paper citing_title [{citing_title}] in which the citation occurs, get a sense of the theme of this paper. 
2. Read the citing_context [{citing_context}], carefully examine the sentence(s) containing the citation marker to understand the citaiton intent.
2. Consider the section name citing_section [{citing_section}] as a hint, like what has been mentioned in "Section-Intent Heuristics".
3. Check the cited work (if provided) [{cited_title}]. If cited_title [] and/or cited_abstract [] ≠ “N/A”, skim them and get a sense of what it has done, provide or prove.
4. Select the single best intent. Choose the one category from Motivation, Compare, Uses, Extends, Future, or Background that most accurately reflects why this resource is cited.
5. Format your response. Reasoning (**2-3 sentences, in 200 words or less**): Focus on how the context (and section, abstract/title if relevant) supports your choice.

First, write your reasoning (**2-3 sentences focusing solely on the `citing_context`'s primary function and signals, noting how `citing_section` (if available and not 'N/A') supports or conflicts with this interpretation**), then state the exact ONE answer as: ##Answer## {{BACKGROUND/MOTIVATION/USES/EXTENDS/COMPARE/FUTURE}}'''
# {cited_title}
# 'cited_abstract': Abstract of the cited work (or “N/A” if unavailable): [{abstract_display}].

    # Model-specific templating
    if model_type == "llama":
        # Llama-3 Instruct format
        return f'''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an academic citation analyst. As an assistant, you must classify the citation intent within the `citing_context` using the provided information.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{prompt_core}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
''' # Assistant tag prompts the model to start generation
    elif model_type == "gemma":
         # Gemma Instruct format (check documentation for specific version, e.g., Gemma 2 vs 3)
        return f'''<start_of_turn>user
You are an academic citation analyst. As an assistant, you must classify the citation intent within the `citing_context` using the provided information.
{prompt_core}<end_of_turn>
<start_of_turn>model
''' # Model tag prompts generation
    elif model_type == "mistral":
         # Mistral Instruct format
         return f'''<s>[INST] You are an academic citation analyst. As an assistant, you must classify the citation intent within the `citing_context` using the provided information.

{prompt_core} [/INST]''' # Wrap user instructions in [INST]
    elif model_type == "qwen":
        # Qwen Chat format
        return f'''<|im_start|>system
You are an academic citation analyst. As an assistant, you must classify the citation intent within the `citing_context` using the provided information.<|im_end|>
<|im_start|>user
{prompt_core}<|im_end|>
<|im_start|>assistant
''' # Assistant tag prompts generation
    else:
        raise ValueError("Unsupported model type. Please choose from 'llama', 'gemma', 'mistral', or 'qwen'.")

# Model Configuration (Replace paths with your actual model locations)
model_config = {
    "mistral": {
        "path": "unsloth/Mistral-Small-24B-Instruct-2501-unsloth-bnb-4bit",
        "args":{
            "max_model_len": 4096,
            "load_format": "bitsandbytes", # Ensure bitsandbytes is installed if using
        }
    },
    "llama": {
        "path": "unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
        "args":{
            "max_model_len": 4096,
            "load_format": "bitsandbytes",
        }
    },
    "qwen": {
        "path": "unsloth/Qwen2.5-72B-Instruct-bnb-4bit",
        "args":{
            "max_model_len": 4096,
            "load_format": "bitsandbytes",
            # "tensor_parallel_size": 2 # Uncomment if using multiple GPUs
        }
    },
    "gemma":{
        "path": "google/gemma-3-27b-it", 
        "args":{
            "max_model_len": 4096
        }
    }
}

def main(model_type: str):
    """
    Main function to load data, run inference, parse results, and save annotations.
    """
    # --- Configuration ---
    # !!! ADJUST THESE PATHS AND FILENAMES AS NEEDED !!!
    data_dir = Path("datasets/acl_arc_full_section")  # Directory containing input .txt files
    input_file_glob = "*.txt" # Pattern to find input files
    input_file_separator = "\t" # Separator used in the input files (e.g., '\t' for TSV)
    citation_marker_in_data = "#AUTHOR_TAG" # The placeholder for citation in context
    citation_marker_for_llm = "[CITED_AUTHOR]" # The placeholder to use in the prompt
    # --- End Configuration ---

    all_contexts_processed = []
    all_citing_titles = []
    all_cited_titles = []
    all_cited_abstracts_raw = [] # Keep raw abstracts to pass to prompt function
    all_unique_ids = []
    all_splits = []
    all_original_annotations = []
    all_section_titles = []

    print("Loading and preparing data from text files...")
    if not data_dir.is_dir():
        print(f"Error: Data directory not found: {data_dir}")
        return

    for file in data_dir.glob(input_file_glob):
        split = file.stem
        print(f" Processing {file}...")
        try:
            # Read data, handle potential bad lines, specify separator
            dataset = pd.read_csv(file, sep=input_file_separator, engine='python', on_bad_lines='warn', quoting=3) # quoting=3 helps with potential quote issues in TSV

            # Define required columns expected in the input file
            required_cols = ["unique_id", "citation_class_label", "citing_title", "cited_title", "cited_abstract", "cite_context_paragraph"]
            if not all(col in dataset.columns for col in required_cols):
                missing_cols = set(required_cols) - set(dataset.columns)
                print(f"Warning: Skipping {file}. Missing columns: {', '.join(missing_cols)}")
                continue

            # --- Data Cleaning and Preparation ---
            # Fill NaN values specifically for text columns to prevent errors
            dataset['citing_title'] = dataset['citing_title'].fillna('')
            dataset['cited_title'] = dataset['cited_title'].fillna('')
            # Keep cited_abstract as is (fillna('') happens later if needed, or handled by prompt function)
            dataset['cited_abstract'] = dataset['cited_abstract'].fillna('') # Fill NaN abstracts with empty string
            dataset['cite_context_paragraph'] = dataset['cite_context_paragraph'].fillna('')
            dataset['section_title'] = dataset['section_title'].fillna('') # Fill NaN section titles with empty string

            # Convert columns to string type to avoid issues with mixed types
            for col in required_cols:
                 if col != 'unique_id': # Assuming unique_id might be numeric, keep as is if needed
                      dataset[col] = dataset[col].astype(str)


            # Process citation contexts: Replace marker
            temp_contexts = []
            for item in dataset["cite_context_paragraph"].tolist():
                 # Basic replacement, assumes item is now a string due to astype(str)
                 context_str = item.replace(citation_marker_in_data, citation_marker_for_llm)
                 temp_contexts.append(context_str)

            # Append data to lists
            all_contexts_processed.extend(temp_contexts)
            all_citing_titles.extend(dataset["citing_title"].tolist())
            all_cited_titles.extend(dataset["cited_title"].tolist())
            all_cited_abstracts_raw.extend(dataset["cited_abstract"].tolist()) # Store raw abstracts (now '' if NaN)
            all_unique_ids.extend(dataset["unique_id"].tolist()) # Ensure unique_id is consistent type
            all_splits.extend([split] * len(dataset))
            all_original_annotations.extend(dataset["citation_class_label"].tolist())
            all_section_titles.extend(dataset["section_title"].tolist()) # Store section titles

        except Exception as e:
            print(f"Error processing {file}: {e}")
            import traceback
            traceback.print_exc() # Print detailed traceback for debugging

    print(f"Prepared {len(all_contexts_processed)} samples for processing.")
    if not all_contexts_processed:
        print("No valid samples found. Exiting.")
        return

    # --- LLM Initialization and Generation ---
    sampling_params = SamplingParams(
        temperature=0.0, # Set to 0 for deterministic classification
        max_tokens=2048   # Allow enough tokens for reasoning + label
    )

    # Initialize the vLLM engine
    print(f"Initializing model: {model_type}")
    if model_type not in model_config:
        print(f"Error: Model type '{model_type}' not found in model_config.")
        return

    model_path = model_config[model_type]["path"]
    model_args = model_config[model_type].get("args", {})
    try:
        # Add trust_remote_code=True - often required for custom code in HF models
        engine = LLM(model=model_path, trust_remote_code=True, **model_args)
    except Exception as e:
        print(f"Error initializing LLM engine: {e}")
        print("Check model path, arguments (like load_format), and available GPU resources/drivers.")
        return

    print("Generating prompts...")
    # Create prompts using the prepared data
    prompts = [
        scicite_prompt(
            citing_context=ctx,
            citing_title=ctt,
            cited_title=cdt,
            cited_abstract=cda, # Pass the raw abstract (potentially '')
            citing_section=cst,
            model_type=model_type
        )
        for ctx, ctt, cdt, cda, cst in zip(
                all_contexts_processed,
                all_citing_titles,
                all_cited_titles,
                all_cited_abstracts_raw,
                all_section_titles # Pass section titles for context
            )
    ]
    for epoch in range(1, 4):
        # Optional: Clean up large lists to free memory before inference if needed
        # Be cautious if you need these lists later for output generation
        # del all_contexts_processed, all_citing_titles, all_cited_titles, all_cited_abstracts_raw

        print(f"Running inference on {len(prompts)} samples...")
        # Generate responses from the LLM
        outputs = engine.generate(prompts, sampling_params=sampling_params)
        print("Inference complete.")

        print("Processing outputs...")
        output_texts = [o.outputs[0].text for o in outputs] # Extract the generated text
        annotations = []
        # Define the valid output labels
        label_options = ["BACKGROUND", "COMPARE", "EXTENDS", "FUTURE", "MOTIVATION", "USES"]

        # --- Output Parsing and Saving ---
        # Ensure you have the necessary lists available here. If deleted, reload or restructure.
        # Assuming lists were NOT deleted:
        for i, output_text in enumerate(output_texts):
            # Get corresponding input data using index 'i'
            unique_id = all_unique_ids[i]
            context = all_contexts_processed[i] # The context passed to the model
            split = all_splits[i]
            original_annotation = all_original_annotations[i]
            citing_title = all_citing_titles[i]
            cited_title = all_cited_titles[i]
            cited_abstract = all_cited_abstracts_raw[i] # The abstract passed to the model

            # --- Robust Parsing Logic ---
            answer_marker = "##Answer##"
            output_text_stripped = output_text.strip()
            answer_start = output_text_stripped.rfind(answer_marker) # Find last marker

            reasoning = output_text_stripped # Default if marker not found
            answer = "Extraction Failed" # Default label

            if answer_start != -1:
                # Extract reasoning (text before the marker)
                reasoning = output_text_stripped[:answer_start].strip()
                # Extract potential answer part (text after the marker)
                potential_answer_part = output_text_stripped[answer_start + len(answer_marker):].strip()

                if potential_answer_part:
                    # Take the first word, clean it (remove punctuation, make uppercase)
                    first_word = potential_answer_part.split()[0]
                    cleaned_word = ''.join(filter(str.isalpha, first_word)).upper()

                    if cleaned_word in label_options:
                        answer = cleaned_word # Assign valid label
                    else:
                        # Log the unexpected label for debugging
                        print(f"Warning: ID {unique_id} - Invalid label found after marker: '{potential_answer_part[:50]}...' -> Cleaned: '{cleaned_word}'")
                        answer = f"{cleaned_word}"
                else:
                    answer = "Empty Label" # Marker found but nothing after it
                    print(f"Warning: ID {unique_id} - Empty label after marker.")
            else:
                # Marker not found - check if the entire output is a valid label (common with some models)
                if output_text_stripped:
                    first_word_output = output_text_stripped.split()[0]
                    cleaned_output_word = ''.join(filter(str.isalpha, first_word_output)).upper()
                    if cleaned_output_word in label_options:
                        answer = cleaned_output_word
                        reasoning = "" # No reasoning provided if only label is output
                        print(f"Info: ID {unique_id} - Output interpreted as label only: '{answer}'")
                    else:
                        answer = "No Marker Found"
                        print(f"Warning: ID {unique_id} - Answer marker '##Answer##' not found in output.")
                else:
                    answer = "Empty Output"
                    reasoning = ""
                    print(f"Warning: ID {unique_id} - Empty output received from model.")
            # --- End Parsing Logic ---

            # Append results to annotations list
            annotations.append({
                "unique_id": unique_id,
                "citing_context": context, # Context with [CITED_AUTHOR]
                "annotation": answer,       # The extracted label
                "reasoning": reasoning,     # Text before ##Answer##
                "citing_title": citing_title,
                "cited_title": cited_title,
                # Store abstract as it was passed (empty string if missing)
                "cited_abstract": cited_abstract if cited_abstract else "N/A",
                "original_annotation": label_options[int(original_annotation)],
                "section_title": all_section_titles[i], # The section title passed to the model
                "split": split,
                "raw_output": output_text # Store the full raw text from LLM for inspection
            })
        # --- End Output Loop ---
        output_file = f"annotate_outputs/acl_arc_data_acl_schema_full_section_outputs_{model_type}_{str(epoch)}.json" # Output JSON filename
        print(f"Saving {len(annotations)} annotations to {output_file}...")
        try:
            with open(output_file, "w") as f:
                json.dump(annotations, f, indent=2)
            print("Annotations saved successfully.")
        except Exception as e:
            print(f"Error saving annotations to {output_file}: {e}")

    print("Script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM for citation intent classification using specified model.")
    # Make model_type a required argument
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=model_config.keys(),
        help="Type of the model to use (e.g., 'llama', 'mistral'). Must be defined in model_config."
    )
    args = parser.parse_args()

    # Run the main function with the selected model type
    main(args.model_type)