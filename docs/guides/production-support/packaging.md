# Package and Deploy NeMo RL Models

This guide covers how to package and deploy NeMo RL models for production use.

## Overview

Model packaging involves creating deployable artifacts that contain your trained model, configuration, and dependencies. This enables easy deployment across different environments and platforms.

## Packaging Formats

### HuggingFace Format

The most common format for deploying NeMo RL models:

```python
from nemo_rl.utils.checkpoint import save_model_for_deployment

# Save model in HuggingFace format
save_model_for_deployment(
    model=model,
    tokenizer=tokenizer,
    output_dir="deployed_model",
    model_name="my-dpo-model",
    model_description="DPO-trained model for math reasoning"
)
```

**Features:**
- Compatible with HuggingFace ecosystem
- Easy integration with existing pipelines
- Support for model cards and metadata
- Version control friendly

### ONNX Format

For optimized inference:

```python
from nemo_rl.utils.checkpoint import export_to_onnx

# Export to ONNX
export_to_onnx(
    model=model,
    output_path="model.onnx",
    input_shape=(1, 512),
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'attention_mask': {0: 'batch_size', 1: 'sequence_length'}
    }
)
```

**Features:**
- Optimized inference performance
- Cross-platform compatibility
- Hardware acceleration support
- Smaller model size

### TorchScript Format

For PyTorch-specific deployments:

```python
from nemo_rl.utils.checkpoint import export_to_torchscript

# Export to TorchScript
export_to_torchscript(
    model=model,
    output_path="model.pt",
    example_inputs=example_inputs
)
```

**Features:**
- PyTorch-native format
- Good performance optimization
- Easy integration with PyTorch applications

## Packaging Configuration

### Model Configuration

```yaml
# packaging_config.yaml
model:
  name: "my-dpo-model"
  version: "1.0.0"
  description: "DPO-trained model for mathematical reasoning"
  author: "Your Name"
  license: "MIT"
  
packaging:
  format: "huggingface"  # or "onnx", "torchscript"
  include_tokenizer: true
  include_config: true
  include_metadata: true
  
  # Model metadata
  metadata:
    task: "text-generation"
    domain: "mathematics"
    tags: ["dpo", "math", "reasoning"]
    
  # Dependencies
  requirements:
    - "torch>=2.0.0"
    - "transformers>=4.30.0"
    - "nemo_rl>=1.0.0"
```

### CLI Packaging

```bash
# Package model using CLI
python -m nemo_rl.package \
  --checkpoint checkpoints/best_model \
  --config packaging_config.yaml \
  --output_dir deployed_model
```

## Model Cards

Create comprehensive model cards for your packaged models:

```markdown
# Model Card for My DPO Model

## Model Details

- **Model Name**: my-dpo-model
- **Version**: 1.0.0
- **Task**: Text Generation
- **Domain**: Mathematics

## Training Data

- **Dataset**: OpenMathInstruct
- **Size**: 1M examples
- **Preprocessing**: Standard tokenization

## Training Configuration

- **Algorithm**: DPO
- **Base Model**: Llama2-7B
- **Training Steps**: 10,000
- **Learning Rate**: 5e-5

## Evaluation Results

- **Accuracy**: 85.2%
- **BLEU Score**: 0.78
- **Human Evaluation**: 4.2/5.0

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("my-dpo-model")
tokenizer = AutoTokenizer.from_pretrained("my-dpo-model")

# Generate text
inputs = tokenizer("Solve: 2x + 5 = 13", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0]))
```

## Limitations

- Limited to mathematical reasoning tasks
- May not generalize to other domains
- Requires careful prompt engineering

## License

MIT License
```

## Deployment Strategies

### Docker Containers

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy model
COPY deployed_model /app/model

# Copy application
COPY app.py .

EXPOSE 8000

CMD ["python", "app.py"]
```

```python
# app.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model
model = AutoModelForCausalLM.from_pretrained("/app/model")
tokenizer = AutoTokenizer.from_pretrained("/app/model")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data["prompt"]
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=128)
    response = tokenizer.decode(outputs[0])
    
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
```

### Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nemo-rl-model
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nemo-rl-model
  template:
    metadata:
      labels:
        app: nemo-rl-model
    spec:
      containers:
      - name: model-server
        image: my-dpo-model:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
---
apiVersion: v1
kind: Service
metadata:
  name: nemo-rl-model-service
spec:
  selector:
    app: nemo-rl-model
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Serverless Deployment

```python
# lambda_function.py
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model (cached between invocations)
model = AutoModelForCausalLM.from_pretrained("/opt/model")
tokenizer = AutoTokenizer.from_pretrained("/opt/model")

def lambda_handler(event, context):
    # Parse request
    body = json.loads(event['body'])
    prompt = body['prompt']
    
    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=128)
    response = tokenizer.decode(outputs[0])
    
    return {
        'statusCode': 200,
        'body': json.dumps({'response': response})
    }
```

## Versioning and Updates

### Semantic Versioning

```python
# version.py
import semver

def bump_version(current_version, bump_type):
    """Bump version according to semantic versioning."""
    version = semver.VersionInfo.parse(current_version)
    
    if bump_type == "major":
        return version.bump_major()
    elif bump_type == "minor":
        return version.bump_minor()
    elif bump_type == "patch":
        return version.bump_patch()
    else:
        raise ValueError(f"Unknown bump type: {bump_type}")
```

### Model Registry

```python
# registry.py
import mlflow
from nemo_rl.utils.checkpoint import save_model_for_deployment

def register_model(model, tokenizer, model_name, version):
    """Register model in MLflow registry."""
    
    # Save model
    model_path = save_model_for_deployment(
        model=model,
        tokenizer=tokenizer,
        output_dir=f"models/{model_name}/{version}",
        model_name=model_name
    )
    
    # Register in MLflow
    mlflow.register_model(
        model_uri=model_path,
        name=model_name,
        tags={
            "version": version,
            "algorithm": "dpo",
            "domain": "mathematics"
        }
    )
```

## Testing Packaged Models

### Unit Tests

```python
# test_packaged_model.py
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_model_loading():
    """Test that packaged model loads correctly."""
    model = AutoModelForCausalLM.from_pretrained("deployed_model")
    tokenizer = AutoTokenizer.from_pretrained("deployed_model")
    
    assert model is not None
    assert tokenizer is not None

def test_model_inference():
    """Test model inference on sample inputs."""
    model = AutoModelForCausalLM.from_pretrained("deployed_model")
    tokenizer = AutoTokenizer.from_pretrained("deployed_model")
    
    # Test input
    prompt = "Solve: 2x + 5 = 13"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate
    outputs = model.generate(**inputs, max_new_tokens=50)
    response = tokenizer.decode(outputs[0])
    
    # Basic validation
    assert len(response) > len(prompt)
    assert "x = 4" in response or "4" in response

def test_model_performance():
    """Test model performance metrics."""
    model = AutoModelForCausalLM.from_pretrained("deployed_model")
    tokenizer = AutoTokenizer.from_pretrained("deployed_model")
    
    import time
    
    # Measure inference time
    prompt = "Solve: 2x + 5 = 13"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    start_time = time.time()
    outputs = model.generate(**inputs, max_new_tokens=50)
    inference_time = time.time() - start_time
    
    # Performance requirements
    assert inference_time < 5.0  # Should complete within 5 seconds
```

### Integration Tests

```python
# test_integration.py
import requests
import json

def test_api_endpoint():
    """Test deployed model API endpoint."""
    
    # Test request
    payload = {
        "prompt": "Solve: 2x + 5 = 13"
    }
    
    response = requests.post(
        "http://localhost:8000/generate",
        json=payload
    )
    
    assert response.status_code == 200
    
    result = response.json()
    assert "response" in result
    assert len(result["response"]) > 0

def test_batch_inference():
    """Test batch inference capabilities."""
    
    payload = {
        "prompts": [
            "Solve: 2x + 5 = 13",
            "What is 3 * 7?",
            "Simplify: 2x + 3x"
        ]
    }
    
    response = requests.post(
        "http://localhost:8000/batch_generate",
        json=payload
    )
    
    assert response.status_code == 200
    
    result = response.json()
    assert "responses" in result
    assert len(result["responses"]) == 3
```

## Best Practices

### 1. Model Optimization

- Use quantization for smaller model size
- Implement caching for repeated requests
- Use batch processing for efficiency

### 2. Security

- Validate all inputs
- Implement rate limiting
- Use HTTPS for API endpoints
- Sanitize outputs

### 3. Monitoring

- Log all requests and responses
- Monitor model performance metrics
- Set up alerts for failures
- Track usage patterns

### 4. Documentation

- Create comprehensive API documentation
- Include usage examples
- Document model limitations
- Provide troubleshooting guides

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Check model file integrity
   - Verify dependencies are installed
   - Ensure sufficient memory

2. **Performance Issues**
   - Enable model optimization
   - Use appropriate hardware
   - Implement caching

3. **Deployment Issues**
   - Check container configuration
   - Verify network connectivity
   - Monitor resource usage

For more help with packaging and deployment, see the [Troubleshoot NeMo RL](troubleshooting.md). 