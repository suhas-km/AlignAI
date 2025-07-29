---
library_name: transformers
tags:
- eu-ai-act
- policy-detection
- compliance
- distilbert
- multi-label-classification
---

# EU AI Act Policy Detection Model

A fine-tuned DistilBERT model for analyzing text and detecting potential EU AI Act compliance issues. This model can identify policy violations, categorize them into specific compliance domains, assess their severity, and provide references to relevant EU AI Act articles.

## Model Details

### Model Description

This model is fine-tuned from DistilBERT to automatically detect and analyze text for compliance with the European Union Artificial Intelligence Act (EU AI Act). It uses multi-task learning to simultaneously predict several compliance-related aspects including violation detection, categorization, severity assessment, and article identification.

### Model Metadata

- **Developed by:** AlignAI Team / Suhas KM
- **Model type:** DistilBERT (Sequence classification)
- **Language(s) (NLP):** English
- **License:** MIT
- **Finetuned from model:** distilbert-base-uncased

### Model Sources

- **Repository:** [suhas-km/eu-ai-act-policy-model](https://huggingface.co/suhas-km/eu-ai-act-policy-model)
- **Original Project:** [AlignAI](https://github.com/suhas-km/AlignAI)

## Uses

### Direct Use

This model is designed to be used directly for analyzing text for compliance with the EU AI Act. Specific use cases include:

- **Compliance Checking**: Analyze documentation, specifications, or product descriptions to identify potential compliance issues with the EU AI Act
- **Risk Assessment**: Evaluate AI systems' descriptions to determine risk levels according to EU AI Act standards
- **Policy Gap Analysis**: Identify areas where existing AI systems or documentation may need updates to meet EU AI Act requirements
- **Educational Tool**: Help developers and policy makers understand EU AI Act requirements through practical examples

### Downstream Use

The model can be integrated into larger compliance management systems or applications such as:

- **Compliance Monitoring Dashboards**: For continuous tracking of AI systems compliance
- **Documentation Review Systems**: For automated pre-review of AI documentation
- **Risk Management Frameworks**: As a component in broader AI governance platforms
- **The AlignAI Application**: This model is a core component of the AlignAI system, which provides a comprehensive interface for EU AI Act compliance monitoring

### Out-of-Scope Use

This model should not be used for:

- **Legal Advice**: The model is not a substitute for qualified legal consultation
- **Definitive Compliance Certification**: While helpful for identifying potential issues, it should not be the sole determiner of regulatory compliance
- **Non-EU Regulatory Frameworks**: The model is specifically trained on EU AI Act regulations and may not apply to other jurisdictions
- **Non-English Content**: The model is trained on English language data and may not perform well on other languages

## Bias, Risks, and Limitations

The model has several important limitations to be aware of:

- **Evolving Regulatory Landscape**: The EU AI Act is relatively new legislation, and interpretations/implementations may evolve over time
- **Limited Training Data**: The model was trained on a limited set of annotated examples, which may not cover all possible compliance scenarios
- **False Positives/Negatives**: Like all ML models, it may generate both false positives (flagging compliant text as problematic) and false negatives (missing actual compliance issues)
- **Complexity Limitations**: The model may struggle with highly technical or legally complex text
- **Bias Toward Training Examples**: The model may be biased toward detecting issues similar to those in its training data

### Recommendations

To mitigate these limitations:

- **Human Review**: Always have qualified personnel review the model's outputs
- **Continuous Updating**: Periodically update the model as regulatory understanding evolves
- **Use as One Tool**: Incorporate this model as one component of a comprehensive compliance strategy, not as a complete solution
- **Diverse Input Review**: Test the model on diverse inputs to better understand its performance boundaries
- **Confidence Thresholds**: Consider implementing confidence thresholds for different use cases

## How to Get Started with the Model

Use the code below to get started with the model:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("suhas-km/eu-ai-act-policy-model")
tokenizer = AutoTokenizer.from_pretrained("suhas-km/eu-ai-act-policy-model")

# Prepare input text
text = "Our facial recognition system uses demographic data to optimize performance across different user groups."
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

# Get predictions
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.sigmoid(outputs.logits).numpy()

# Map predictions to categories
categories = ["bias", "transparency", "risk", "human_oversight", "data_governance", 
              "technical_robustness", "prohibited_practices", "other"]

# Display results for the first 8 categories (model has 29 outputs total)
results = {}
for i, category in enumerate(categories):
    if i < len(predictions[0]):
        results[category] = float(predictions[0][i])

# Format and display results
print("EU AI Act Compliance Analysis:")
for category, score in results.items():
    print(f"{category}: {score:.2f} ({score >= 0.5 and 'FLAGGED' or 'OK'})")

# Overall risk assessment
risk_score = np.mean([v for v in results.values()])
print(f"\nOverall risk assessment: {risk_score:.2f}")
```

## Training Details

### Training Data

The model was trained on a custom dataset derived from the EU AI Act text and annotated examples. The dataset consists of:

- Text samples with annotations for compliance categories, severity, and relevant EU AI Act articles
- Format: JSONL (JSON Lines)
- Located in the project at `/data/data2/` with the following splits:
  - `train.jsonl`: Used for model training
  - `dev.jsonl`: Used for validation during training
  - `test.jsonl`: Used for final evaluation

Each sample in the dataset contains text excerpts and annotations for multiple aspects of compliance, enabling the multi-task learning approach.

### Training Procedure

The model was trained using multi-task learning to simultaneously predict multiple aspects of EU AI Act compliance.

#### Preprocessing

- Texts were tokenized using DistilBERT's tokenizer
- Maximum sequence length: 512 tokens
- Labels were converted to multi-hot encoding for multi-label classification

#### Training Hyperparameters

- **Base model**: distilbert-base-uncased
- **Learning rate**: 5e-5
- **Batch size**: 16
- **Epochs**: 5
- **Optimizer**: AdamW
- **Loss function**: Binary Cross-Entropy with Logits (for multi-label classification)
- **Training regime**: fp32 (full precision)
- **Random seed**: 42

#### Speeds, Sizes, Times

- **Model size**: ~268MB
- **Training hardware**: Standard CPU/GPU workstation
- **Checkpoints**: Saved at various steps (13, 52, 78)

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model was evaluated using:

- A held-out test set (`test.jsonl`) containing EU AI Act-related text samples and compliance annotations
- The test set was curated to cover a variety of compliance scenarios and categories

#### Factors

The evaluation disaggregated results by:

- Compliance category (bias, transparency, risk management, etc.)
- Violation detection (binary classification)
- Severity assessment
- Article identification accuracy

#### Metrics

The following metrics were used for evaluation:

- **F1 Score**: To measure the balance between precision and recall, particularly important for imbalanced classes
- **Accuracy**: For overall classification performance
- **Precision and Recall**: To understand false positive and false negative rates
- **ROC-AUC**: For binary classification tasks (e.g., violation detection)

### Results

The model achieved the following results on the test set:

- **Overall F1 Score**: ~0.35 across all compliance categories and tasks
- **Violation Detection F1**: ~0.81 (strongest performing aspect)
- **Per-category Performance**:
  - Bias detection: Moderate performance
  - Transparency requirements: Moderate performance
  - Risk management: Lower performance due to complexity of assessment
  - Other categories: Varying performance based on training data availability

#### Summary

The model performs best on violation detection tasks, with an F1 score of approximately 0.81, making it useful for initial compliance screening. Performance varies across specific compliance categories, with an overall F1 score of around 0.35 across all categories. The model is particularly effective at identifying clear violations but may require human review for nuanced cases.

## Model Examination

Analysis of the model's predictions shows it is most confident in identifying:

1. Explicit mentions of prohibited AI practices
2. Clear transparency violations
3. Obvious data governance issues

The model sometimes struggles with:

1. Implicit compliance issues that require deeper context
2. Novel compliance scenarios not well-represented in training data
3. Technical assessments requiring domain expertise

## Environmental Impact

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** Standard workstation GPU
- **Hours used:** Approximately 4-8 hours for training
- **Training Location:** Local workstation

## Technical Specifications

### Model Architecture and Objective

- **Base Architecture**: DistilBERT (a distilled version of BERT)
- **Classification Head**: Multi-label classification with 29 output units
- **Problem Type**: Multi-label sequence classification
- **Objective**: Minimize binary cross-entropy loss across multiple compliance-related tasks

### Compute Infrastructure

#### Hardware

- Standard GPU workstation for model training
- CPU inference supported for deployment

#### Software

- **Framework**: PyTorch, Hugging Face Transformers
- **Python Version**: 3.8+
- **Key Libraries**: transformers, torch, sklearn

## Citation

**BibTeX:**

```bibtex
@software{alignai2025euaiactpolicy,
  author = {Suhas K M},
  title = {EU AI Act Policy Detection Model},
  year = {2025},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/suhas-km/eu-ai-act-policy-model}},
}
```

**APA:**

Suhas KM. (2025). EU AI Act Policy Detection Model [Software]. Retrieved from https://huggingface.co/suhas-km/eu-ai-act-policy-model

## Glossary

- **EU AI Act**: The European Union's comprehensive legislative framework for regulating artificial intelligence systems.
- **Multi-label Classification**: Machine learning approach where multiple target labels can be assigned to each instance simultaneously.
- **F1 Score**: The harmonic mean of precision and recall, providing a balance between the two metrics.
- **DistilBERT**: A smaller, faster, cheaper version of BERT that retains much of its language understanding capabilities.
- **Compliance Categories**: Distinct aspects of the EU AI Act that AI systems must adhere to (transparency, data governance, etc.).

## More Information

For more information about the AlignAI project and the EU AI Act compliance tools:

- [AlignAI GitHub Repository](https://github.com/suhas-km/AlignAI)
- [EU AI Act Official Text](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai)
- [Contact the project team](mailto:suhaskm@gmail.com) for collaboration opportunities or support

## Model Card Authors

- Suhas K M

## Model Card Contact

For questions, feedback, or issues related to this model, please contact:

- GitHub: [suhas-km](https://github.com/suhas-km)
- Email: [suhaskm@gmail.com](mailto:suhaskm@gmail.com)