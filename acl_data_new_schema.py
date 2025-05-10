from vllm import LLM, SamplingParams
import pandas as pd
import json
from pathlib import Path
import argparse
from transformers import AutoTokenizer

# Define the new prompt generation function
def new_prompt(citing_context, citing_title, citing_section, model_type: str) -> str:
    conversation = [
        {
            "role": "system",
            "content": "You are an expert annotator specializing in academic citation analysis. Your primary task is to analyze a given text snippet containing a specific citation and determine the appropriate labels for three distinct dimensions: Citation Object, Citation Function, and Stance, based *strictly* on the detailed guidelines provided below. Your analysis must focus *only* on the relationship between the citing text and the specific work indicated by the citation.\n\n**Crucially, before stating the final labels, you MUST output your step-by-step reasoning process.** This reasoning should be clearly structured and justify each label choice by referencing specific evidence from the `citing_context` and the relevant points from the guidelines below.\n\n# Citation Annotation Instructions & Guidelines\n---\n\n## 1: Citation Object — What is being cited?\nYour task in this dimension is to determine what contribution from the cited work is being referenced in the given context.\n- **Performed Work**: The citing work references the actions, processes, or events the cited work carried out, without focusing on specific reusable outputs.\n  * Ask: Is the citing author referencing “what the cited work has done”?\n  * Example: \"They developed a new method for clustering.\"\n- **Discovery**: The citing work references observations, findings, conclusions, or insights drawn by the cited work.\n  * Ask: Is the citing author referencing something the cited work discovered or proved?\n  * Examples: “They found that MI outperforms frequency.” “They observed cross-domain variation.”\n- **Produced Resource**: The citing work references a specific, reusable output from the cited work, including but not limited to: Data (Datasets, Lexicons), Method (Models, Algorithms, Formula, Feature Sets, Curated Input Examples, Experimental Setup, Evaluation Metrics, Benchmarks, Frameworks, Architectures), Tool (Software, Source Code, Libraries, Platforms, Systems). This includes reusable conceptual structures like scoring formulas or evaluation protocols.\n  * Ask: Is the citing author referring to a concrete artifact, method, or standard created by the cited work?\n  * Examples: “We use their parser.” “The COMLEX lexicon.” “Their noun phrase constraint.”\n\n**Guideline Notes (Object):**\n*   Be careful distinguishing Discovery (findings) from Performed Work (actions/methods described but not necessarily reused).\n*   Produced Resource includes conceptual outputs (formulas, protocols) not just physical tools.\n\n---\n\n## 2. Citation Function — Why is it cited?\nYour task in this dimension is to determine why the cited work is being referenced by the citing work.\n- **Contextualize**: The citing work uses the cited work to provide background information, describe prior research, or acknowledge related work. The citation serves to frame the current study within the existing body of knowledge.\n  * Ask: Is the citation being used to describe prior work without affecting the citing author’s approach?\n  * Example: \"Previous work by [cited work] describes the process of feature extraction.\"\n- **Signal Gap**: The citing work references the cited work to highlight an unresolved issue or area where further research is needed. This can be raised by the citing or cited work, includes planned future work by the citing author, and is usually forward-looking.\n  * Ask: Is the citation used to show what hasn’t been done yet? Or something the citing paper will do in the future (even if suggested by the cited work)?\n  * Example: \"[cited work] noted that tuning is needed.\", \"Despite advancements in [cited work], tuning is needed.\"\n- **Highlight Limitation**: The citing work or the cited work point out flaws, weaknesses, or constraints that have been recognized in the cited work. Usually backward-looking, motivating the citing work's improvement.\n  * Ask: Is the citing paper critiquing the cited work?\n  * Example: \"[cited work]’s method fails to account for dynamic shifts in the data.\"\n- **Justify Design Choice**: The citing work references the cited work (often its findings or established nature) to support its own methodological or conceptual decisions, showing that the approach chosen is grounded in established research, *without* directly applying or modifying the cited work’s resource.\n  * Ask: Is the cited work being used to defend or justify a choice the citing authors made, leveraging its findings or status, but without directly applying the cited work’s resource (e.g., method, setting, model)?\n  * Example: \"[cited work] shows this setting helps improve the accuracy on noisy data, thus we adopt this setting.\" (Citing the *finding* to justify, not necessarily using the exact method).\n- **Use**: The citing work directly applies a resource from the cited work’s own research. The citing work leverages or incorporates the cited work’s resource into its own work. These resources include Data, Method, or Tool as defined under Produced Resource. Direct application is key.\n  * Ask: Is the cited contribution being actively used or applied *by the citing paper* in the current work? (e.g., running their code, using their dataset, applying their formula).\n  * Example: \"We use [cited work]’s sentiment analysis tool to classify the dataset.\"\n- **Modify**: The citing work builds upon or extends the cited work, adjusting, enhancing, reconfiguring, or integrating a cited contribution (e.g., method, model, tool).\n  * Ask: Is the cited contribution being used or applied in the current work *after* modification, extension, or adaptation by the citing paper?\n  * Example: \"We use a variation of the algorithm described in [cited work], adapting it for...\"\n- **Evaluate Against**: The citing work references the cited work as a benchmark or point of comparison, typically to test or evaluate the performance of its own approach or results against the cited work's results or method. This includes comparing scores reported in the cited paper.\n  * Ask: Is the citing author comparing their results or method performance against those of the cited work?\n  * Example: \"We evaluate our model's performance against [cited work]’s approach.\", “We achieve better accuracy than [cited work]”\n\n**Guideline Notes (Function):**\n*   **Use vs. Contextualize:** Only label 'Use' if the *citing paper* actively applies the resource. Describing *how the cited paper* used something is Contextualize (Note 2). Evaluating on a dataset implies Use (Note 4).\n*   **Use vs. Modify:** If the citing work adapts, enhances, or reconfigures the cited contribution, it's Modify (Note 5, 11). If used 'as is' or 'based on' without described changes, it's Use (Note 11).\n*   **Signal Gap vs. Highlight Limitation:** Signal Gap points to unresolved/future issues. Highlight Limitation points to existing flaws in the cited work (Note 9).\n*   **Evaluate Against vs. Highlight Limitation:** Evaluate Against is for performance comparison/benchmarking. Highlight Limitation is for pointing out flaws to motivate improvement (Note 6). Simply reporting a cited work's better performance without comparison is Contextualize, not Evaluate Against (Note 8).\n*   **Justify Design Choice vs. Use:** Justify uses cited findings/status to support a decision. Use involves direct application of a cited resource (Guideline definition, Note 12).\n*   **Future Use:** Plans to use a resource are often Signal Gap, not Use (Note 3).\n\n---\n\n## 3. Stance — How is the citation framed?\nThis dimension captures the tone or evaluative position the citing author takes toward the cited work.\n- **Neutral**: The citing work presents the cited work factually, without any praise or critique. It’s purely descriptive or procedural, and there’s no evaluative language used.\n  * Ask: Is the citing work simply describing or reporting what the cited work did?\n  * Example: \"X proposed a method for feature extraction.\"\n- **Endorsing**: The citing work presents the cited work positively, either through explicit praise (\"standard method,\" \"widely used,\" \"significant improvement\") or implicitly by adopting/using the cited work’s method, model, or results without critique. Using a finding to justify a design choice is also Endorsing.\n  * Ask: Is the citing work positively aligning with the cited work or using it without critique?\n  * Example: \"We adopt X’s method, as it has been shown to improve performance.\"\n- **Critical**: The citing work critiques, disagrees, or points out flaws in the cited work. This includes explicit critique, refutation, negative evaluations, or highlighting limitations.\n  * Ask: Is the citing work pointing out a flaw or offering a negative critique of the cited work?\n  * Example: \"X’s method is inadequate for handling large datasets due to its computational complexity.\"\n\n**Guideline Notes (Stance):**\n*   Use/Modify/Justify Design Choice often implies Endorsing unless critique is present (Note 2, 7, 12).\n*   Highlight Limitation usually implies Critical.\n*   Contextualize is often Neutral unless explicit evaluative language is used.\n\n---\n\n### Input Fields\n\n- **citing_context**: The sentence(s) around the citation marker (e.g. “As shown in …[CITED_AUTHOR]…”).\n- **citing_section**: Section name (e.g. “Abstract”, “Introduction”, “Method”, “Experiment”, “Conclusion”, “Discussion”, “Future Work”).\n- **citing_title**: Title of the paper containing the citation.\n\n---\n\n### Annotation Procedure\n\n1.  **Scan the title** (`{citing_title}`) to understand the paper's main theme or focus.\n2.  **Read the `citing_context` carefully**, paying attention to the words immediately around the citation and the sentence's overall purpose.\n3.  **Consider the `citing_section`** for additional clues about the citation's role (e.g., Methods sections often involve Use/Modify/Justify, Introduction often involves Contextualize/Signal Gap/Highlight Limitation).\n4.  **Perform Step-by-Step Reasoning (Output this first):**\n    *   **Reasoning Step 1: Analyze Citation Object:**\n        *   Identify the *exact* phrase/concept being cited from the context.\n        *   Determine the **Object Type** (Performed Work, Discovery, Produced Resource).\n        *   **Justify** your choice by explaining *why* it fits the definition, referencing specific words from the `citing_context` and relevant guideline points.\n    *   **Reasoning Step 2: Analyze Citation Function:**\n        *   Determine the **Function Type** (Contextualize, Signal Gap, Highlight Limitation, Justify Design Choice, Use, Modify, Evaluate Against).\n        *   **Justify** your choice by explaining *why* the citing author mentions the object, referencing specific words/phrases from the `citing_context` and relevant guideline points (especially the Notes distinguishing similar functions).\n    *   **Reasoning Step 3: Analyze Stance:**\n        *   Determine the **Stance Type** (Neutral, Endorsing, Critical).\n        *   **Justify** your choice based on evaluative language (or lack thereof) in the `citing_context`, the chosen Citation Function, referencing specific words/phrases and relevant guideline points.\n5.  **Format the Final Output (After the reasoning):** Provide *only* the final labels in the specified format below, based on your reasoning.\n\ntext\nTarget: <Performed Work|Discovery|Produced Resource>\nEngagement: <Contextualize|Signal Gap|Highlight Limitation|Justify Design Choice|Use|Modify|Evaluate Against>\nStance: <Neutral|Endorsing|Critical>"
        },
        {
            "role": "user",
            "content": "citing_context: ['The framework represents a generalization of several predecessor NLG systems based on Meaning-Text Theory: FoG (Kittredge and Polgu~re, 1991), LFS (Iordanskaja et al., 1992), and JOYCE (Rambow and Korelsky, 1992).', 'The framework was originally developed for the realization of deep-syntactic structures in NLG ( [CITED_AUTHOR] ) .']\nciting_section: related work\nciting_title: A framework for MT and multilingual NLG systems based on uniform lexico-structural processing",
        },
        {
            "role": "assistant",
            "content": "**Reasoning Step 1: Analyze Citation Object**\n\nFirst, I analyze the citation object by pinpointing what exactly is being cited within the context. The most relevant sentence for the citation is:\n\n> The framework was originally developed for the realization of deep-syntactic structures in NLG ([CITED_AUTHOR]).\n\nThe cited work is credited with the original development of a framework intended for a particular purpose (\"the realization of deep-syntactic structures in NLG\"). The language focuses on the action performed by the cited work—developing a framework, especially for a particular application in NLG (Natural Language Generation).\n\nBased on the guideline definitions:\n- **Performed Work**: This is about referencing what the cited work has done.\n- **Produced Resource**: This would involve referencing a specific, reusable artifact (like a tool, method, dataset) for direct use.\n\nHere, the context is not focused on using the framework or its technical details as a resource but is stating the cited work’s contribution (\"was originally developed\")—namely, developing the framework for a particular aspect of NLG, which matches the concept of **Performed Work**.\n\n**Reasoning Step 2: Analyze Citation Function**\n\nThe function requires examining why this citation is provided in context. The broader passage is in a \"related work\" section, and the sentence’s intent is to describe the contribution of the cited authors (\"The framework was originally developed…\").\n\nLooking at the guideline definitions:\n- **Contextualize**: Used when providing background information or framing prior work.\n- **Use/Modify/Evaluate Against/Justify Design Choice**: Would require evidence of direct application, adaptation, or justification of method—none of which is present here.\n- **Signal Gap/Highlight Limitation**: Would require mention of gaps, critiques, or flaws—none are evident.\n\nThe text’s primary purpose is to summarize and situate the cited work historically and thematically, serving to contextualize the development of the framework. No evaluative or comparative function is present.\n\n**Reasoning Step 3: Analyze Stance**\n\nNow, to determine the stance, I consider the tone:\n- The text is straightforward and factual (\"was originally developed for...\").\n- There is no evaluative (praise or critique) language.\n- Describes what the cited work did, matching the **Neutral** category.\n\nThe guidelines note that Contextualize is often Neutral unless explicit evaluative language is present, which fits this case.\n\n---\n\n**Final Labels**\n\nTarget: Performed Work  \nEngagement: Contextualize  \nStance: Neutral"
        },
        {
            "role": "user",
            "content": f"citing_context: {citing_context}\nciting_section: {citing_section}\nciting_title: {citing_title}"
        }
    ]
    return conversation

# Model Configuration (Replace paths with your actual model locations)
model_config = {
    "mistral": {
        "path": "unsloth/Mistral-Small-24B-Instruct-2501-unsloth-bnb-4bit",
        "args":{
            "max_model_len": 6144,
            "load_format": "bitsandbytes", # Ensure bitsandbytes is installed if using
        }
    },
    "llama": {
        "path": "unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
        "args":{
            "max_model_len": 6144,
            "load_format": "bitsandbytes",
        }
    },
    "qwen": {
        "path": "unsloth/Qwen2.5-72B-Instruct-bnb-4bit",
        "args":{
            "max_model_len": 6144,
            "load_format": "bitsandbytes",
            # "tensor_parallel_size": 2 # Uncomment if using multiple GPUs
        }
    },
    "gemma":{
        "path": "google/gemma-3-27b-it", 
        "args":{
            "max_model_len": 6144
        }
    }
}

def main(model_type: str):
    """
    Main function to load data, run inference, parse results, and save annotations.
    """
    # --- Configuration ---
    # !!! ADJUST THESE PATHS AND FILENAMES AS NEEDED !!!
    data_dir = Path("datasets/acl_arc_new_framework")  # Directory containing input .txt files
    input_file_glob = "*.csv" # Pattern to find input files
    input_file_separator = "\t" # Separator used in the input files (e.g., '\t' for TSV)
    citation_marker_in_data = "#AUTHOR_TAG" # The placeholder for citation in context
    citation_marker_for_llm = "[CITED_AUTHOR]" # The placeholder to use in the prompt
    # --- End Configuration ---

    all_contexts_processed = []
    all_citing_titles = []
    all_unique_ids = []
    all_original_annotations = []
    all_section_titles = []

    print("Loading and preparing data from text files...")
    if not data_dir.is_dir():
        print(f"Error: Data directory not found: {data_dir}")
        return

    for file in data_dir.glob(input_file_glob):
        print(f" Processing {file}...")
        try:
            # Read data, handle potential bad lines, specify separator
            dataset = pd.read_csv(file, sep=input_file_separator, engine='python', on_bad_lines='warn', quoting=3) # quoting=3 helps with potential quote issues in TSV

            # Define required columns expected in the input file
            required_cols = ["unique_id", "citing_title", "section_title", "cite_context_paragraph", "citation_object", "citation_function", "stance"]
            if not all(col in dataset.columns for col in required_cols):
                missing_cols = set(required_cols) - set(dataset.columns)
                print(f"Warning: Skipping {file}. Missing columns: {', '.join(missing_cols)}")
                continue

            # --- Data Cleaning and Preparation ---
            # Fill NaN values specifically for text columns to prevent errors
            dataset['citing_title'] = dataset['citing_title'].fillna('')
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
            all_unique_ids.extend(dataset["unique_id"].tolist()) # Ensure unique_id is consistent type
            all_original_annotations.extend(dataset[["citation_object", "citation_function", "stance"]].to_dict('records'))
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
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"Error initializing LLM engine: {e}")
        print("Check model path, arguments (like load_format), and available GPU resources/drivers.")
        return

    print("Generating prompts...")
    # Create prompts using the prepared data
    prompts = [
        new_prompt(
            citing_context=ctx,
            citing_title=ctt,
            citing_section=cst,
            model_type=model_type
        )
        for ctx, ctt, cst in zip(
                all_contexts_processed,
                all_citing_titles,
                all_section_titles
            )
    ]
    prompts = tokenizer.apply_chat_template(prompts, tokenize=False)
    prompts = [p+'**\nReasoning Step 1: Analyze Citation Object**\n' for p in prompts]
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

        # --- Output Parsing and Saving ---
        for i, output_text in enumerate(output_texts):
            unique_id = all_unique_ids[i]
            context = all_contexts_processed[i]
            original_annotation = all_original_annotations[i]
            citing_title = all_citing_titles[i]
            section_title = all_section_titles[i]

            # --- Robust Extraction Logic ---
            import re

            # Extract reasoning (everything before the first "---" or "Final Labels")
            reasoning = ""
            target, engagement, stance = "", "", ""

            # Try to split on "---" or "Final Labels"
            reasoning_match = re.split(r"\n\s*-{3,}\s*\n|\n\s*\*\*Final Labels\*\*\s*\n", output_text, maxsplit=1)
            if len(reasoning_match) > 1:
                reasoning = reasoning_match[0].strip()
                label_section = reasoning_match[1]
            else:
                reasoning = output_text.strip()
                label_section = output_text

            # Try to extract labels with various patterns and tolerate missing colons/extra whitespace
            def extract_label(label, text):
                # Try several patterns for robustness
                patterns = [
                    rf"{label}\s*:\s*([^\n\r]+)",         # Standard: Target: value
                    rf"{label}\s*-\s*([^\n\r]+)",         # With dash: Target- value
                    rf"{label}\s+([^\n\r]+)",             # Just space: Target value
                ]
                for pat in patterns:
                    m = re.search(pat, text, re.IGNORECASE)
                    if m:
                        return m.group(1).strip()
                return ""

            target = extract_label("Target", label_section)
            engagement = extract_label("Engagement", label_section)
            stance = extract_label("Stance", label_section)

            # Fallback: Try to extract all labels from a single line if present
            if not (target and engagement and stance):
                multi_match = re.search(
                    r"Target\s*[:\-]?\s*([^\n\r]+)[\n\r]+Engagement\s*[:\-]?\s*([^\n\r]+)[\n\r]+Stance\s*[:\-]?\s*([^\n\r]+)",
                    label_section, re.IGNORECASE)
                if multi_match:
                    target, engagement, stance = [multi_match.group(i).strip() for i in range(1, 4)]

            # Fallback: Try to find all labels in a single line (e.g., "Target: X Engagement: Y Stance: Z")
            if not (target and engagement and stance):
                inline_match = re.search(
                    r"Target\s*[:\-]?\s*([^\n\r]+)\s+Engagement\s*[:\-]?\s*([^\n\r]+)\s+Stance\s*[:\-]?\s*([^\n\r]+)",
                    label_section, re.IGNORECASE)
                if inline_match:
                    target, engagement, stance = [inline_match.group(i).strip() for i in range(1, 4)]

            # Fallback: If any label is missing, mark as Extraction Failed
            if not (target and engagement and stance):
                print(f"Warning: ID {unique_id} - Failed to extract all labels. Output: {output_text[:200]}...")
                annotation = "Extraction Failed"
            else:
                annotation = {
                    "Target": target,
                    "Engagement": engagement,
                    "Stance": stance
                }

            # Append results to annotations list
            annotations.append({
                "unique_id": unique_id,
                "citing_context": context,
                "annotation": annotation,
                "reasoning": reasoning,
                "citing_title": citing_title,
                "original_annotation": original_annotation,
                "section_title": section_title,
                "raw_output": output_text
            })
        # --- End Output Loop ---
        output_file = f"annotate_outputs/acl_new_schema_{model_type}_{str(epoch)}.json" # Output JSON filename
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