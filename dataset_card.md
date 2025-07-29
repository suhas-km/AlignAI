---
languages:
- en
license: mit
pretty_name: EU AI Act Flagged Dataset
size_categories:
- 100K<n<1M
tags:
- eu-ai-act
- compliance
- policy-detection
- legal
---

# EU AI Act Flagged Dataset

This dataset contains annotated examples for EU AI Act compliance detection, used for training and evaluating the [EU AI Act Policy Detection Model](https://huggingface.co/suhas-km/eu-ai-act-policy-model).

## Dataset Description

### Dataset Summary

This dataset consists of text samples related to AI systems and their deployment, annotated for compliance with the European Union's Artificial Intelligence Act (EU AI Act). Each sample is labeled with:
- Whether it contains a potential violation
- The category of compliance/violation
- Severity level 
- Relevant EU AI Act articles
- Explanatory context

The dataset is designed to help train models that can identify potential regulatory issues in AI system descriptions and documentation.

### Supported Tasks

The dataset supports the following tasks:
- Multi-label classification for EU AI Act compliance categories
- Violation detection (binary and multi-class classification)
- Severity assessment
- Article reference identification

### Dataset Structure

The dataset contains three splits:
- `train`: Training examples
- `validation`: Development/validation examples  
- `test`: Testing examples

Each example contains the following fields:
- `text`: The AI system description or documentation text
- `violation`: Boolean flag or "borderline" indicating whether a violation is present
- `category`: The compliance category (e.g., "transparency", "risk_management", "data_governance")
- `severity`: The severity level of the violation ("none", "low", "medium", "high", "critical", or "borderline")
- `articles`: List of relevant EU AI Act articles
- `explanation`: Textual explanation of the compliance assessment
- `context`: Application domain context
- `ambiguity`: Optional boolean flag indicating cases with ambiguous regulatory interpretation

## Dataset Creation

### Source Data

The dataset was created by the AlignAI project team based on:
- EU AI Act regulatory text
- Expert analysis of potential compliance scenarios
- Synthetic examples covering various AI application domains

### Annotations

The annotations were created by experts in AI ethics and EU regulatory compliance, following a detailed annotation guide based on the EU AI Act provisions.

## Considerations for Using the Data

### Social Impact of the Dataset

This dataset aims to improve AI system compliance with EU regulations, potentially leading to more transparent, fair, and accountable AI systems. It may help reduce harmful impacts of AI by facilitating early detection of compliance issues.

### Discussion of Biases

The dataset may contain biases in terms of:
- Coverage of different AI domains and applications
- Interpretation of regulatory requirements
- Assessment of violation severity

Users should be aware that regulatory interpretation can vary and evolve over time.

### Personal and Sensitive Information

This dataset does not contain personal information. All examples are synthetic or anonymized.

## Additional Information

### Dataset Curators

The dataset was curated by the AlignAI team / Suhas K M.

### Licensing Information

MIT License

### Citation Information

Please cite as:
```
@dataset{euaiactflagged2025,
  author = {Suhas K M},
  title = {EU AI Act Flagged Dataset},
  year = {2025},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/suhas-km/EU-AI-Act-Flagged}},
}
```

### Contributions

Thanks to all who contributed to the creation and annotation of this dataset!
