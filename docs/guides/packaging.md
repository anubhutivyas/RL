# Packaging Guide

This guide covers packaging and deployment strategies for NeMo RL models and applications.

## Overview

NeMo RL provides multiple packaging options to deploy trained models and applications in various environments, from local development to production clusters.

## Model Packaging

### Hugging Face Format

The most common format for sharing and deploying models:

```bash
# Convert checkpoint to Hugging Face format
python examples/converters/convert_megatron_to_hf.py \
    --input_dir /path/to/checkpoint \
    --output_dir /path/to/hf/checkpoint \
    --model_name Qwen/Qwen2.5-1.5B
```

**Features**:
- Compatible with Hugging Face ecosystem
- Easy integration with transformers library
- Support for model cards and metadata
- Version control and sharing

### ONNX Format

For optimized inference deployment:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model
model = AutoModelForCausalLM.from_pretrained("/path/to/checkpoint")
tokenizer = AutoTokenizer.from_pretrained("/path/to/checkpoint")

# Export to ONNX
dummy_input = torch.randint(0, 1000, (1, 10))
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input_ids'],
    output_names=['logits'],
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'logits': {0: 'batch_size', 1: 'sequence_length'}
    }
)
```

### TensorRT Format

For maximum inference performance:

```python
import tensorrt as trt
from transformers import AutoTokenizer

# Load and optimize model
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
config = builder.create_builder_config()
config.max_workspace_size = 1 << 30  # 1GB

# Build engine
engine = builder.build_engine(network, config)

# Save engine
with open("model.trt", "wb") as f:
    f.write(engine.serialize())
```

## Application Packaging

### Docker Containers

Create production-ready containers:

```dockerfile
# Base image
FROM nvcr.io/nvidia/pytorch:23.12-py3

# Install dependencies
RUN pip install nemo-rl transformers accelerate

# Copy application
COPY . /workspace/app
WORKDIR /workspace/app

# Set environment variables
ENV HF_HOME=/workspace/cache
ENV CUDA_VISIBLE_DEVICES=0

# Expose ports
EXPOSE 8000

# Run application
CMD ["python", "app.py"]
```

**Multi-stage build for optimization**:

```dockerfile
# Build stage
FROM nvcr.io/nvidia/pytorch:23.12-py3 as builder

RUN pip install nemo-rl transformers accelerate
COPY . /workspace/app
WORKDIR /workspace/app

# Download and cache models
RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B')"

# Production stage
FROM nvcr.io/nvidia/pytorch:23.12-py3

COPY --from=builder /root/.cache /root/.cache
COPY --from=builder /workspace/app /workspace/app
WORKDIR /workspace/app

ENV HF_HOME=/root/.cache
EXPOSE 8000

CMD ["python", "app.py"]
```

### Kubernetes Deployment

Deploy to Kubernetes clusters:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nemo-rl-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nemo-rl-app
  template:
    metadata:
      labels:
        app: nemo-rl-app
    spec:
      containers:
      - name: nemo-rl
        image: nemo-rl:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "16Gi"
            cpu: "4"
            nvidia.com/gpu: 1
          limits:
            memory: "32Gi"
            cpu: "8"
            nvidia.com/gpu: 1
        env:
        - name: HF_HOME
          value: "/workspace/cache"
        volumeMounts:
        - name: cache-volume
          mountPath: /workspace/cache
      volumes:
      - name: cache-volume
        persistentVolumeClaim:
          claimName: nemo-rl-cache-pvc
```

### Ray Serve

Deploy as a Ray Serve application:

```python
import ray
from ray import serve
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

@serve.deployment(num_replicas=2, ray_actor_options={"num_gpus": 1})
class NeMoRLModel:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained("/path/to/checkpoint")
        self.tokenizer = AutoTokenizer.from_pretrained("/path/to/checkpoint")
        self.model.eval()
    
    async def __call__(self, request):
        data = await request.json()
        prompt = data["prompt"]
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=100,
                temperature=0.7
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"response": response}

# Deploy
serve.run(NeMoRLModel.bind())
```

## Deployment Strategies

### Blue-Green Deployment

Minimize downtime during updates:

```yaml
# Blue deployment (current)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nemo-rl-blue
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: nemo-rl
        image: nemo-rl:v1.0
        ports:
        - containerPort: 8000

# Green deployment (new)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nemo-rl-green
spec:
  replicas: 0  # Start with 0 replicas
  template:
    spec:
      containers:
      - name: nemo-rl
        image: nemo-rl:v1.1
        ports:
        - containerPort: 8000
```

### Canary Deployment

Gradual rollout for testing:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nemo-rl-canary
spec:
  replicas: 1  # Start with 1 replica
  template:
    spec:
      containers:
      - name: nemo-rl
        image: nemo-rl:v1.1
        ports:
        - containerPort: 8000
```

### Rolling Update

Zero-downtime updates:

```bash
kubectl set image deployment/nemo-rl-app nemo-rl=nemo-rl:v1.1
kubectl rollout status deployment/nemo-rl-app
```

## Performance Optimization

### Model Optimization

```python
# Quantization for reduced memory usage
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("/path/to/checkpoint")

# Dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Save quantized model
torch.save(quantized_model.state_dict(), "quantized_model.pt")
```

### Inference Optimization

```python
# Enable optimizations
import torch

# JIT compilation
model = torch.jit.script(model)

# Mixed precision
model = model.half()

# Memory optimization
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
```

### Caching Strategies

```python
# Model caching
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

cache_dir = "/workspace/cache"
os.environ["HF_HOME"] = cache_dir

# Load and cache model
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B",
    cache_dir=cache_dir,
    torch_dtype=torch.bfloat16
)
```

## Monitoring and Observability

### Metrics Collection

```python
import time
import psutil
import torch
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
request_counter = Counter('nemo_rl_requests_total', 'Total requests')
request_duration = Histogram('nemo_rl_request_duration_seconds', 'Request duration')
gpu_memory = Gauge('nemo_rl_gpu_memory_bytes', 'GPU memory usage')

def monitor_request(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        request_counter.inc()
        
        # Monitor GPU memory
        if torch.cuda.is_available():
            gpu_memory.set(torch.cuda.memory_allocated())
        
        result = func(*args, **kwargs)
        
        duration = time.time() - start_time
        request_duration.observe(duration)
        
        return result
    return wrapper
```

### Health Checks

```python
from flask import Flask, jsonify
import torch

app = Flask(__name__)

@app.route('/health')
def health_check():
    status = {
        'status': 'healthy',
        'gpu_available': torch.cuda.is_available(),
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'memory_usage': psutil.virtual_memory().percent
    }
    
    if torch.cuda.is_available():
        status['gpu_memory'] = torch.cuda.memory_allocated() / 1024**3  # GB
    
    return jsonify(status)
```

### Logging

```python
import logging
import json
from datetime import datetime

# Structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def log_request(prompt, response, duration):
    log_entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'prompt_length': len(prompt),
        'response_length': len(response),
        'duration_ms': duration * 1000,
        'gpu_memory_mb': torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    }
    
    logger.info(json.dumps(log_entry))
```

## Security Considerations

### Model Security

```python
# Model encryption
from cryptography.fernet import Fernet
import pickle

# Generate key
key = Fernet.generate_key()
cipher_suite = Fernet(key)

# Encrypt model
with open('model.pkl', 'rb') as f:
    model_data = f.read()

encrypted_data = cipher_suite.encrypt(model_data)

# Save encrypted model
with open('model.encrypted', 'wb') as f:
    f.write(encrypted_data)
```

### Access Control

```python
# API key authentication
from functools import wraps
from flask import request, jsonify

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key != 'your-secret-key':
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/generate')
@require_api_key
def generate_text():
    # Generation logic
    pass
```

### Input Validation

```python
import re
from typing import List

def validate_prompt(prompt: str) -> bool:
    # Check for malicious content
    malicious_patterns = [
        r'<script>',
        r'javascript:',
        r'data:text/html'
    ]
    
    for pattern in malicious_patterns:
        if re.search(pattern, prompt, re.IGNORECASE):
            return False
    
    # Check length
    if len(prompt) > 10000:
        return False
    
    return True

def sanitize_input(text: str) -> str:
    # Remove potentially dangerous characters
    return re.sub(r'[<>"\']', '', text)
```

## Cost Optimization

### Resource Management

```yaml
# Kubernetes resource limits
resources:
  requests:
    memory: "8Gi"
    cpu: "2"
    nvidia.com/gpu: 1
  limits:
    memory: "16Gi"
    cpu: "4"
    nvidia.com/gpu: 1
```

### Auto-scaling

```yaml
# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: nemo-rl-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: nemo-rl-app
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Spot Instances

```yaml
# Use spot instances for cost savings
spec:
  template:
    spec:
      nodeSelector:
        node.kubernetes.io/instance-type: g4dn.xlarge
      tolerations:
      - key: "kubernetes.azure.com/scalesetpriority"
        operator: "Equal"
        value: "spot"
        effect: "NoSchedule"
```

## Best Practices

### Version Management

```python
# Version your models and applications
__version__ = "1.0.0"

# Include version in API responses
@app.route('/version')
def version():
    return jsonify({
        'version': __version__,
        'model_version': 'v1.0.0',
        'framework_version': 'nemo-rl-0.2.1'
    })
```

### Error Handling

```python
import traceback
from flask import jsonify

@app.errorhandler(Exception)
def handle_error(error):
    logger.error(f"Error: {error}")
    logger.error(traceback.format_exc())
    
    return jsonify({
        'error': 'Internal server error',
        'message': str(error)
    }), 500
```

### Testing

```python
import pytest
from unittest.mock import Mock, patch

def test_model_generation():
    with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
        # Test generation logic
        mock_model.return_value.generate.return_value = torch.tensor([[1, 2, 3]])
        
        # Your test logic here
        pass

def test_api_endpoint():
    with app.test_client() as client:
        response = client.post('/generate', json={'prompt': 'Hello'})
        assert response.status_code == 200
        assert 'response' in response.json
```

## Getting Help

- [Deployment Examples](../../examples/deployment/) - Complete deployment examples
- [Configuration Reference](../reference/configuration.md) - Configuration options
- [Troubleshooting Guide](../reference/troubleshooting.md) - Common deployment issues
- [API Reference](../reference/api.md) - Programmatic deployment 