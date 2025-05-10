# **SOFT**, a **S**emantically **O**rthogonal **F**ramework with **T**wo dimensions

## Repository Structure
```
root
├── classifiers                >>  open-sourced CIC classifiers that used for experiments
│   ├── CitationIntentOpenLLM
│   └── CitePrompt 
├── visualization              >>  paper figures code
└── *.py                       >>  get annotation from open LLMs
```

## Requirements

```vllm==0.8.4```

## How TO Use

`python acl_data_acl_schema.py --model_type=[gemma|llama|mistral|qwen]`

## Acknowledgment

[CitationIntentOpenLLM](https://github.com/athenarc/CitationIntentOpenLLM)

[CitePrompt](https://github.com/AvishekLahiri/CitePrompt)