# AlignAI ML Models

This directory contains the trained machine learning models used by AlignAI for detecting bias, PII, and policy violations.

## Directory Structure

- `bias_detection/` - Contains the bias detection model
- `pii_detection/` - Contains the PII detection model
- `policy_detection/` - Contains the policy compliance model

## Deploying Models

To deploy trained models from the Model-Training directory to these folders, use the deployment script:

```bash
python scripts/deploy_models.py
```

You can deploy specific models by using the `--models` flag:

```bash
python scripts/deploy_models.py --models bias_detection pii_detection
```

## Model Fallback Mechanism

AlignAI implements a robust fallback mechanism:

1. When ML models are available, the system uses them for more accurate detection
2. If ML models are unavailable or fail, the system falls back to rule-based detection
3. For policy detection, only ML-based detection is available (no rule-based fallback)

## Configuration

The ML models are initialized when the `ComplianceGuard` is created and can be enabled/disabled 
through the `use_ml` parameter:

```python
# Initialize with ML models enabled
compliance_guard = ComplianceGuard(use_ml=True)

# Disable ML models and use only rule-based detection
compliance_guard = ComplianceGuard(use_ml=False)
```

## Model Details

### Bias Detection Model
- Type: Text classification model
- Base model: DistilBERT
- Training dataset: Custom bias examples

### PII Detection Model
- Type: Token classification (NER) model
- Base model: DistilBERT
- Training dataset: WNUT-17 + synthetic PII data

### Policy Detection Model
- Type: Text classification model
- Base model: DistilBERT
- Training dataset: EU AI Act compliance examples
