from vllm import LLM, SamplingParams
import pandas as pd
import json
from pathlib import Path
import argparse

# Define the new prompt generation function
def acl_prompt(citing_context, citing_title, model_type: str) -> str:
    # Shared prompt core - uses f-string interpolation later
    prompt_core = f'''
** Intent Categories:**

1.  **BACKGROUND:** Use this label when the citation states, mentions, or points to background information, giving more context about a problem, concept, approach, topic, or the importance of the problem in the field. This category encompasses citations previously labeled as Motivation, Extends (when not directly using/modifying the method), Future Work, and general Background.
2.  **METHOD:** Use this label when the current paper is making direct use of a method, tool, algorithm, dataset, evaluation metric, code, or specific experimental setup from the cited work. Look for verbs indicating usage (e.g., "use," "employ," "adopt," "apply") or descriptions suggesting the cited work provides a component *for* the current study.
3.  **RESULT_COMPARISON:** Use this label when the citation is used specifically to compare the results, findings, performance, or benchmarks of the current paper with those of the cited work. Look for comparative language (e.g., "outperforms," "similar to," "in contrast to," "compared with") applied directly to outcomes.

**Rationale for Simplified Schema:**
This concise annotation scheme focuses on the most distinct and actionable intents: direct use of methods/resources (METHOD) and comparison of outcomes (RESULT_COMPARISON). Other contextual citations, including motivation, extensions of ideas (but not direct method use), future possibilities, and general context, are grouped under BACKGROUND.

**Input Information (For each citation):**

*   `citing_context`: The sentence(s) surrounding the citation marker (e.g., "...[CITED_AUTHOR]…”).
*   `citing_title`: Title of the paper in which the citation occurs.

**Annotation Procedure:**

1.  **Read the Citing Title:** Understand the main topic of the paper {{citing_title}}.
2.  **Analyze the Citing Context:** Carefully examine the sentence(s) {{citing_context}} containing the citation marker [CITED_AUTHOR] to determine *how* the cited work is being referenced. Focus on the verbs and surrounding phrases.
3.  **Select the Single Best Intent:** Choose **one** category from **BACKGROUND, METHOD, or RESULT_COMPARISON** that most accurately reflects the *primary reason* for this specific citation in this context.
4.  **Provide Reasoning:** Explain your choice in **2-3 sentences** (max 500 words). Focus on how the `citing_context`, keywords, verbs, section, and potentially the cited title support your selected intent label.

citing_context: {citing_context}
citing_title: {citing_title}

**Final Answer Format:**

First, write your reasoning (**2-3 sentences focusing solely on the `citing_context`'s primary function and signals, noting how `citing_section` (if available and not 'N/A') supports or conflicts with this interpretation**), then provide your answer in exactly this format without any extra spaces: ##Answer##BACKGROUND or ##Answer##METHOD or ##Answer##RESULT_COMPARISON'''


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
            "max_model_len": 8192,
            "load_format": "bitsandbytes", # Ensure bitsandbytes is installed if using
        }
    },
    "llama": {
        "path": "unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
        "args":{
            "max_model_len": 8192,
            "load_format": "bitsandbytes",
        }
    },
    "qwen": {
        "path": "unsloth/Qwen2.5-72B-Instruct-bnb-4bit",
        "args":{
            "max_model_len": 8192,
            "load_format": "bitsandbytes",
            # "tensor_parallel_size": 2 # Uncomment if using multiple GPUs
        }
    },
    "gemma":{
        "path": "google/gemma-3-27b-it", # Example path, adjust as needed
        "args":{
            "max_model_len": 8192
            # Add quantization args here if using a quantized Gemma
        }
    }
}

def main(model_type: str):
    """
    Main function to load data, run inference, parse results, and save annotations.
    """
    # --- Configuration ---
    # !!! ADJUST THESE PATHS AND FILENAMES AS NEEDED !!!
    data_dir = Path("classifiers/CitationIntentOpenLLM/datasets/formatted/act2")  # Directory containing input .txt files
    input_file_glob = "test.csv" # Pattern to find input files
    input_file_separator = "," # Separator used in the input files (e.g., '\t' for TSV)
    citation_marker_in_data = "#AUTHOR_TAG" # The placeholder for citation in context
    citation_marker_for_llm = "[CITED_AUTHOR]" # The placeholder to use in the prompt
    # --- End Configuration ---

    all_contexts_processed = []
    all_citing_titles = []
    all_unique_ids = []
    all_original_annotations = []

    print("Loading and preparing data from text files...")
    if not data_dir.is_dir():
        print(f"Error: Data directory not found: {data_dir}")
        return

    for file in data_dir.glob(input_file_glob):
        split = file.stem
        print(f" Processing {file}...")
        try:
            # Read data, handle potential bad lines, specify separator
            dataset = pd.read_csv(file, sep=input_file_separator)

            # Define required columns expected in the input file
            required_cols = ["unique_id", "citation_class_label", "citing_title", "cite_context_paragraph"]
            if not all(col in dataset.columns for col in required_cols):
                missing_cols = set(required_cols) - set(dataset.columns)
                print(f"Warning: Skipping {file}. Missing columns: {', '.join(missing_cols)}")
                continue

            # --- Data Cleaning and Preparation ---
            # Fill NaN values specifically for text columns to prevent errors
            dataset['citing_title'] = dataset['citing_title'].fillna('')
            dataset['cite_context_paragraph'] = dataset['cite_context_paragraph'].fillna('')

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
            all_unique_ids.extend(dataset["unique_id"].tolist()) # Ensure unique_id is consistent type
            all_original_annotations.extend(dataset["citation_class_label"].tolist())

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
        top_p=1.0,       # Disable top_p when temp is 0
        max_tokens=512   # Allow enough tokens for reasoning + label
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
        acl_prompt(
            citing_context=ctx,
            citing_title=ctt,
            model_type=model_type
        )
        for ctx, ctt in zip(
                all_contexts_processed,
                all_citing_titles
            )
    ]

    # Optional: Clean up large lists to free memory before inference if needed
    # Be cautious if you need these lists later for output generation
    # del all_contexts_processed, all_citing_titles, all_cited_titles, all_cited_abstracts_raw
    for epoch in range(1, 4):
        print(f"Running inference on {len(prompts)} samples...")
        # Generate responses from the LLM
        outputs = engine.generate(prompts, sampling_params=sampling_params)
        print("Inference complete.")

        print("Processing outputs...")
        output_texts = [o.outputs[0].text for o in outputs] # Extract the generated text
        annotations = []
        # Define the valid output labels
        
        label_options = {"BACKGROUND": "BACKGROUND", "COMPARES_CONTRASTS":"RESULT_COMPARISON", "EXTENSION":"BACKGROUND", "FUTURE":"BACKGROUND", "MOTIVATION":"BACKGROUND", "USES": "METHOD"}
        new_label_options = ["BACKGROUND", "METHOD", "RESULT_COMPARISON"]

        # --- Output Parsing and Saving ---
        # Ensure you have the necessary lists available here. If deleted, reload or restructure.
        # Assuming lists were NOT deleted:
        for i, output_text in enumerate(output_texts):
            # Get corresponding input data using index 'i'
            unique_id = all_unique_ids[i]
            context = all_contexts_processed[i] # The context passed to the model
            original_annotation = all_original_annotations[i]
            citing_title = all_citing_titles[i]

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
                    # Take the first word (up to any whitespace)
                    first_word = potential_answer_part.split()[0]
                    # Clean but preserve underscores
                    cleaned_word = ''.join(c for c in first_word if c.isalpha() or c == '_').upper()

                    if cleaned_word in new_label_options:
                        answer = cleaned_word # Assign valid label
                    else:
                        # Log the unexpected label for debugging
                        print(f"Warning: ID {unique_id} - Invalid label found after marker: '{potential_answer_part[:50]}...' -> Cleaned: '{cleaned_word}'")
                        answer = f"{cleaned_word}"
            else:
                # Marker not found - check if the entire output is a valid label (common with some models)
                if output_text_stripped:
                    first_word_output = output_text_stripped.split()[0]
                    cleaned_output_word = ''.join(filter(str.isalpha, first_word_output)).upper()
                    if cleaned_output_word in new_label_options:
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
                "citing_context": context, # Context with @@CITATION@@
                "annotation": answer,       # The extracted label
                "reasoning": reasoning,     # Text before ##Answer##
                "citing_title": citing_title,
                "original_annotation": label_options[original_annotation],
                "split": split,
                "raw_output": output_text # Store the full raw text from LLM for inspection
            })
        # --- End Output Loop ---
        output_file = f"annotate_outputs/act2_scicite_schema_{model_type}_{str(epoch)}.json" # Output JSON filename

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