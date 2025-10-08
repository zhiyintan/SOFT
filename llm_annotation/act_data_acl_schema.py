from vllm import LLM, SamplingParams
import pandas as pd
import json
from pathlib import Path
import argparse

# Define the new prompt generation function
def acl_prompt(citing_context, citing_title, model_type: str) -> str:
    # Shared prompt core - uses f-string interpolation later
    prompt_core = f'''You are an academic citation analyst. As an assistant, you must classify the citation intent within the `citing_context` using the provided information.

**Input Information:**

*   **`citing_title`**: The title of the paper *making* the citation. (Provides overall topic context).
*   **`citing_context`**: The sentence(s) from the citing paper where the citation @@CITATION@@ occurs. (This is the primary text for analysis).

**Task:**

1.  **Analyze Context:**
    *   Read the `citing_context` carefully to understand how the citation is used in the sentence.
    *   Use the `cited_title` to understand the general topic/type of the cited work.
    *   **If `cited_abstract` is available (not "N/A")**, read it to get details on the cited work's contributions, methods, and findings to help interpret the citation intent.
    *   **If `cited_abstract` is "N/A"**, rely primarily on the `cited_title` and the `citing_context` itself.
    *   Consider the `citing_title` for general topic context of the citing paper.
2.  **Identify Key Signals in `citing_context`:**
    *   Look for specific verbs: "uses," "applies," "adapts," "extends," "compares," "outperforms," "confirms," "introduces," "discusses."
    *   Look for phrases indicating motivation: "limitation of," "gap in," "challenge is," "fails to address," "in response to."
    *   Look for phrases indicating comparison: "similar to," "different from," "better than," "consistent with," "in contrast to."
    *   Look for general statements of fact or background information referencing the cited work.
    *   Look for mentions of future directions referencing the cited work.
3.  **Match to Category:** Choose the *best fitting* category based on the primary function of the citation in the `citing_context`, informed by the available information (`cited_title`, `citing_context`, and `cited_abstract` if available):
    *   **Background:** The citation provides relevant information (e.g., definitions, standard approaches, previous findings) for the domain that the present paper discusses. The `citing_context` makes a general statement referencing the cited work.
    *   **Motivation:** The citation illustrates the need for the work done in the citing paper, often by pointing out a gap, limitation, or problem in the cited work or the area it represents.
    *   **Uses:** The citing paper directly employs data, methods, tools, evaluation metrics, code, experimental setup, etc., described *or strongly implied by title/context* in the cited paper. The `citing_context` explicitly states this usage.
    *   **Extends:** The citing paper builds upon or modifies data, methods, theories, etc., from the cited paper. The `citing_context` indicates adaptation, improvement, or further development based on the cited work.
    *   **Compare:** The citing paper explicitly compares its approach, results, findings, or position to those of the cited paper.
    *   **Future:** The citation is mentioned in the `citing_context` as relevant to potential future work extending the citing paper's contributions.
4.  **Resolve Ambiguity:**
    *   Focus on the *main reason* the citation appears in the `citing_context`.
    *   **If `cited_abstract` is available**, use its details to clarify the nature of the cited work (e.g., is it a method, dataset, or general concept?).
    *   **If `cited_abstract` is "N/A"**, rely more heavily on the `cited_title` (e.g., does it sound like a method paper?) and the specific verbs/phrasing in the `citing_context` to make the best judgment.

**Examples (Illustrative):**

Input:
`citing_title`: "Efficient Graph Neural Networks for Node Classification"
`citing_context`: "Several foundational approaches exist for learning on graphs @@CITATION@@."
`cited_title`: "Graph Convolutional Networks (GCN)"
`cited_abstract`: "Introduces Graph Convolutional Networks (GCN), a scalable approach for semi-supervised learning on graph-structured data. We demonstrate competitive results on citation networks and knowledge graphs."
Reasoning: The context makes a general statement about foundational work. The cited title and abstract describe a foundational method (GCN), supporting the Background classification. -> ##Answer## BACKGROUND

Input:
`citing_title`: "Addressing Long-Range Dependencies in Sequence Modeling"
`citing_context`: "However, standard recurrent models @@CITATION@@ struggle with capturing long-range dependencies, a limitation our proposed architecture aims to overcome."
`cited_title`: "Long Short-Term Memory (LSTM)"
`cited_abstract`: "Presents LSTM, a recurrent neural network architecture designed to handle long-range dependencies better than simple RNNs." # Example abstract for a relevant cited work
Reasoning: The context explicitly points out a limitation ('struggle with long-range dependencies') of the cited work's type ('standard recurrent models', exemplified by LSTM here) to justify the citing paper's goals. -> ##Answer## MOTIVATION

Input:
`citing_title`: "Protein Structure Prediction using Deep Learning"
`citing_context`: "We implemented the residue-residue distance prediction module based on the architecture described in @@CITATION@@."
`cited_title`: "AlphaFold: Improved protein structure prediction using potentials from deep learning"
`cited_abstract`: "Describes AlphaFold, a deep learning system predicting protein 3D structure from sequence. Details the neural network architecture and training methodology."
Reasoning: The context clearly states implementation ('implemented...module based on') of a specific part ('architecture') from the cited work. The cited title and abstract confirm it describes a relevant architecture. -> ##Answer## USES

Input:
`citing_title`: "Enhanced Object Detection in Adverse Weather"
`citing_context`: "Our detection network adapts the anchor box mechanism from @@CITATION@@ by adding anchors optimized for low-visibility conditions."
`cited_title`: "YOLOv3: An Incremental Improvement"
`cited_abstract`: "Presents YOLOv3, improving YOLO with multi-scale predictions, a better backbone network, and a new objectness score. Fast and accurate object detection."
Reasoning: The context states adaptation ('adapts...by adding') of a specific mechanism ('anchor box mechanism') from the cited work. The cited title and abstract confirm YOLOv3 involves object detection mechanisms. -> ##Answer## EXTENDS

Input:
`citing_title`: "A New Optimizer for Faster Deep Learning Training"
`citing_context`: "On the ImageNet dataset, our proposed optimizer achieved convergence 15% faster than Adam @@CITATION@@."
`cited_title`: "Adam: A Method for Stochastic Optimization"
`cited_abstract`: "Presents the Adam optimization algorithm, computationally efficient and well-suited for problems with large datasets/parameters."
Reasoning: The context makes a direct performance comparison ('faster than') between the citing paper's contribution ('our proposed optimizer') and the cited work ('Adam'), confirmed as an optimizer by its title/abstract. ##Answer## COMPARE

Input:
`citing_title`: "Quantum Algorithms for Chemistry Simulations"
`citing_context`: "Investigating the application of adaptive measurement strategies, such as those used in @@CITATION@@, could further improve the efficiency of our simulation framework in future iterations."
`cited_title`: "Adaptive Measurement Techniques for VQE"
`cited_abstract`: "N/A" # Example where abstract is missing
Reasoning: The context explicitly mentions the cited work's potential topic ('adaptive measurement strategies', suggested by the title) as relevant for future improvement ('could further improve...in future iterations'), even without abstract details. ##Answer## FUTURE

**Task:**
For the given input:
`citing_title`: {citing_title}
`citing_context`: {citing_context}

First, write your reasoning (3-4 sentences focusing on the `citing_context` and how the available information (`cited_title`, `cited_abstract` if present) informs the interpretation), then state the final answer as: ##Answer## BACKGROUND/MOTIVATION/USES/EXTENDS/COMPARE/FUTURE
'''

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
    citation_marker_for_llm = "@@CITATION@@" # The placeholder to use in the prompt
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
        label_options = {"BACKGROUND", "MOTIVATION", "USES", "EXTENDS", "COMPARE", "FUTURE"}

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
                    # Take the first word, clean it (remove punctuation, make uppercase)
                    first_word = potential_answer_part.split()[0]
                    cleaned_word = ''.join(filter(str.isalpha, first_word)).upper()

                    if cleaned_word in label_options:
                        answer = cleaned_word # Assign valid label
                    else:
                        # Log the unexpected label for debugging
                        print(f"Warning: ID {unique_id} - Invalid label found after marker: '{potential_answer_part[:50]}...' -> Cleaned: '{cleaned_word}'")
                        answer = f"Invalid Label: {cleaned_word}"
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
                "citing_context": context, # Context with @@CITATION@@
                "annotation": answer,       # The extracted label
                "reasoning": reasoning,     # Text before ##Answer##
                "citing_title": citing_title,
                "original_annotation": original_annotation, # Original label for comparison
                "split": split,
                "raw_output": output_text # Store the full raw text from LLM for inspection
            })
        # --- End Output Loop ---
        output_file = f"annotate_outputs/act2_acl_schema_{model_type}_{str(epoch)}.json" # Output JSON filename

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