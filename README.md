# **SOFT**, a **S**emantically **O**rthogonal **F**ramework with **T**wo dimensions

This repository contains the dataset and accompanying code for the paper “Semantically Orthogonal Framework for Citation Classification: Disentangling Intent and Content” (DOI: [https://doi.org/10.1007/978-3-032-05409-8_12](https://doi.org/10.1007/978-3-032-05409-8_12)).

## Repository Structure
```
root
├── classifiers                >>  open-sourced CIC classifiers that used for experiments
│   ├── CitationIntentOpenLLM
│   └── CitePrompt 
├── visualization              >>  paper figures code
└── llm_annotation             >>  get annotations from open LLMs for Human-LLM IAA evaluations
```

## Requirements

```vllm==0.8.4```

## How TO Use

`python llm_annotation/acl_data_acl_schema.py --model_type=[gemma|llama|mistral|qwen]`

## Citation

If you use SOFT or our dataset/code in your research, please cite our paper:
```bibtex
@InProceedings{10.1007/978-3-032-05409-8_12,
    author="Duan, Changxu
    and Tan, Zhiyin",
    editor="Balke, Wolf-Tilo
    and Golub, Koraljka
    and Manolopoulos, Yannis
    and Stefanidis, Kostas
    and Zhang, Zheying",
    title="Semantically Orthogonal Framework for Citation Classification: Disentangling Intent and Content",
    booktitle="Linking Theory and Practice of Digital Libraries",
    year="2026",
    publisher="Springer Nature Switzerland",
    address="Cham",
    pages="183--206",
    abstract="Understanding the role of citations is essential for research assessment and citation-aware digital libraries. However, existing citation classification frameworks often conflate citation intent (why a work is cited) with cited content type (what part is cited), limiting their effectiveness in auto classification due to a dilemma between fine-grained type distinctions and practical classification reliability. We introduce SOFT, a Semantically Orthogonal Framework with Two dimensions that explicitly separates citation intent from cited content type, drawing inspiration from semantic role theory. We systematically re-annotate the ACL-ARC dataset using SOFT and release a cross-disciplinary test set sampled from ACT2. Evaluation with both zero-shot and fine-tuned Large Language Models (LLMs) demonstrates that SOFT enables higher agreement between human annotators and LLMs, and supports stronger classification performance and robust cross-domain generalization compared to ACL-ARC and SciCite annotation frameworks. These results confirm SOFT's value as a clear, reusable annotation standard, improving clarity, consistency, and generalizability for digital libraries and scholarly communication infrastructures. All code and data are publicly available on GitHub (https://github.com/zhiyintan/SOFT).",
    isbn="978-3-032-05409-8"
}
```

## Acknowledgment

[CitationIntentOpenLLM](https://github.com/athenarc/CitationIntentOpenLLM)

[CitePrompt](https://github.com/AvishekLahiri/CitePrompt)