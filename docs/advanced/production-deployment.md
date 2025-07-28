---
description: "Deploy NeMo RL models to production with monitoring, debugging, and serving architectures"
categories: ["advanced"]
tags: ["deployment", "production", "serving", "monitoring", "debugging", "architecture"]
personas: ["mle-focused", "admin-focused"]
difficulty: "advanced"
content_type: "tutorial"
modality: "universal"
---

# Production Deployment

This guide covers how to deploy NeMo RL models to production with monitoring, debugging, and serving architectures.

## Overview

NeMo RL provides comprehensive tools for deploying models to production. This guide focuses on production-ready deployment strategies.

## Model Serving Architecture

### Base Model Server

```python
from nemo_rl.models import ModelServer

class CustomModelServer(ModelServer):
    def __init__(self, config):
        super().__init__(config)
        # Initialize serving components
        
    def load_model(self, model_path):
        # Load model from checkpoint
        pass
        
    def preprocess(self, input_data):
        # Preprocess input data
        pass
        
    def predict(self, input_data):
        # Run model inference
        pass
        
    def postprocess(self, predictions):
        # Postprocess model outputs
        pass
```

### REST API Server

```python
from flask import Flask, request, jsonify
import torch

class RESTModelServer:
    def __init__(self, config):
        self.app = Flask(__name__)
        self.model = self.load_model(config['model_path'])
        self.setup_routes()
        
    def setup_routes(self):
        @self.app.route('/predict', methods=['POST'])
        def predict():
            try:
                # Get input data
                input_data = request.json['input']
                
                # Preprocess
                processed_input = self.preprocess(input_data)
                
                # Run inference
                with torch.no_grad():
                    predictions = self.model(processed_input)
                
                # Postprocess
                output = self.postprocess(predictions)
                
                return jsonify({'predictions': output})
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
                
    def run(self, host='0.0.0.0', port=5000):
        self.app.run(host=host, port=port)
```

### gRPC Server

```python
import grpc
from concurrent import futures
import nemo_rl_pb2
import nemo_rl_pb2_grpc

class GRPCModelServer(nemo_rl_pb2_grpc.ModelServiceServicer):
    def __init__(self, config):
        self.model = self.load_model(config['model_path'])
        
    def Predict(self, request, context):
        try:
            # Extract input from request
            input_data = self.extract_input(request)
            
            # Run inference
            predictions = self.model(input_data)
            
            # Create response
            response = nemo_rl_pb2.PredictResponse()
            response.predictions.extend(predictions.tolist())
            
            return response
            
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return nemo_rl_pb2.PredictResponse()

def serve_grpc(config):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    nemo_rl_pb2_grpc.add_ModelServiceServicer_to_server(
        GRPCModelServer(config), server
    )
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()
```

## Model Optimization for Production

### Model Quantization

```python
import torch.quantization as quantization

class QuantizedModelServer(ModelServer):
    def __init__(self, config):
        super().__init__(config)
        self.quantization_config = config.get('quantization', {})
        
    def quantize_model(self, model):
        """Quantize model for production deployment"""
        
        # Dynamic quantization
        if self.quantization_config.get('dynamic', False):
            model = quantization.quantize_dynamic(
                model, 
                {torch.nn.Linear}, 
                dtype=torch.qint8
            )
            
        # Static quantization
        elif self.quantization_config.get('static', False):
            model.eval()
            model = quantization.quantize_static(
                model,
                self.calibration_data,
                self.calibration_fn,
                dtype=torch.qint8
            )
            
        return model
```

### Model Pruning

```python
import torch.nn.utils.prune as prune

class PrunedModelServer(ModelServer):
    def __init__(self, config):
        super().__init__(config)
        self.pruning_config = config.get('pruning', {})
        
    def prune_model(self, model):
        """Prune model for production deployment"""
        
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Apply pruning
                prune.l1_unstructured(
                    module,
                    name='weight',
                    amount=self.pruning_config.get('amount', 0.3)
                )
                
        return model
```

### Model Compilation

```python
import torch.jit

class CompiledModelServer(ModelServer):
    def __init__(self, config):
        super().__init__(config)
        self.compilation_config = config.get('compilation', {})
        
    def compile_model(self, model):
        """Compile model for production deployment"""
        
        # Trace model
        if self.compilation_config.get('trace', False):
            example_input = torch.randn(1, 512)
            model = torch.jit.trace(model, example_input)
            
        # Script model
        elif self.compilation_config.get('script', False):
            model = torch.jit.script(model)
            
        return model
```

## Monitoring and Observability

### Performance Monitoring

```python
import time
import psutil
import GPUtil

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {}
        
    def monitor_inference(self, func):
        """Decorator to monitor inference performance"""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.virtual_memory().used
            
            # Run inference
            result = func(*args, **kwargs)
            
            # Record metrics
            end_time = time.time()
            end_memory = psutil.virtual_memory().used
            
            inference_time = end_time - start_time
            memory_used = end_memory - start_memory
            
            self.record_metrics({
                'inference_time': inference_time,
                'memory_used': memory_used,
                'timestamp': start_time
            })
            
            return result
        return wrapper
        
    def record_metrics(self, metrics):
        """Record performance metrics"""
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
```

### Health Checks

```python
class HealthChecker:
    def __init__(self, model_server):
        self.model_server = model_server
        self.health_status = 'healthy'
        
    def check_health(self):
        """Perform health check"""
        try:
            # Test model inference
            test_input = self.create_test_input()
            predictions = self.model_server.predict(test_input)
            
            # Check predictions are valid
            if predictions is not None and len(predictions) > 0:
                self.health_status = 'healthy'
                return True
            else:
                self.health_status = 'unhealthy'
                return False
                
        except Exception as e:
            self.health_status = 'error'
            self.last_error = str(e)
            return False
            
    def get_health_status(self):
        """Get current health status"""
        return {
            'status': self.health_status,
            'timestamp': time.time(),
            'error': getattr(self, 'last_error', None)
        }
```

### Metrics Collection

```python
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge

class MetricsCollector:
    def __init__(self):
        # Define metrics
        self.request_counter = Counter(
            'model_requests_total',
            'Total number of model requests',
            ['model_name', 'endpoint']
        )
        
        self.inference_duration = Histogram(
            'model_inference_duration_seconds',
            'Model inference duration in seconds',
            ['model_name']
        )
        
        self.model_memory_usage = Gauge(
            'model_memory_usage_bytes',
            'Model memory usage in bytes',
            ['model_name']
        )
        
    def record_request(self, model_name, endpoint):
        """Record a model request"""
        self.request_counter.labels(model_name=model_name, endpoint=endpoint).inc()
        
    def record_inference_duration(self, model_name, duration):
        """Record inference duration"""
        self.inference_duration.labels(model_name=model_name).observe(duration)
        
    def record_memory_usage(self, model_name, memory_bytes):
        """Record memory usage"""
        self.model_memory_usage.labels(model_name=model_name).set(memory_bytes)
```

## Deployment Strategies

### Blue-Green Deployment

```python
class BlueGreenDeployment:
    def __init__(self, config):
        self.config = config
        self.blue_server = None
        self.green_server = None
        self.current_active = 'blue'
        
    def deploy_new_version(self, model_path):
        """Deploy new model version using blue-green strategy"""
        
        # Determine which environment to deploy to
        if self.current_active == 'blue':
            target_env = 'green'
            self.green_server = self.create_server(model_path)
        else:
            target_env = 'blue'
            self.blue_server = self.create_server(model_path)
            
        # Health check new deployment
        if self.health_check(target_env):
            # Switch traffic
            self.switch_traffic(target_env)
            self.current_active = target_env
            return True
        else:
            # Rollback
            self.rollback(target_env)
            return False
            
    def switch_traffic(self, target_env):
        """Switch traffic to target environment"""
        # Update load balancer configuration
        # This would typically involve updating nginx or similar
        pass
```

### Canary Deployment

```python
class CanaryDeployment:
    def __init__(self, config):
        self.config = config
        self.canary_traffic_percentage = 0.1  # 10% traffic to canary
        
    def deploy_canary(self, model_path):
        """Deploy canary version with gradual traffic increase"""
        
        # Deploy canary version
        canary_server = self.create_server(model_path)
        
        # Start with small traffic percentage
        self.set_traffic_split(canary=0.05, stable=0.95)
        
        # Monitor canary performance
        if self.monitor_canary_performance():
            # Gradually increase traffic
            self.gradually_increase_traffic()
        else:
            # Rollback canary
            self.rollback_canary()
            
    def gradually_increase_traffic(self):
        """Gradually increase traffic to canary"""
        traffic_levels = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
        
        for level in traffic_levels:
            self.set_traffic_split(canary=level, stable=1-level)
            
            # Monitor for specified duration
            time.sleep(self.config.get('monitoring_duration', 300))
            
            if not self.monitor_canary_performance():
                self.rollback_canary()
                return False
                
        return True
```

## Containerization

### Docker Configuration

```dockerfile
# Dockerfile for NeMo RL model serving
FROM pytorch/pytorch:2.0.0-cuda11.8-cudnn8-runtime

# Install dependencies
RUN pip install nemo-rl flask grpcio prometheus-client

# Copy model and code
COPY model/ /app/model/
COPY server.py /app/

# Set working directory
WORKDIR /app

# Expose port
EXPOSE 5000

# Run server
CMD ["python", "server.py"]
```

### Kubernetes Deployment

```yaml
# kubernetes-deployment.yaml
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
        image: nemo-rl-model:latest
        ports:
        - containerPort: 5000
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 5000
          initialDelaySeconds: 5
          periodSeconds: 5
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
    targetPort: 5000
  type: LoadBalancer
```

## Debugging Production Issues

### Model Debugging

```python
class ModelDebugger:
    def __init__(self, model_server):
        self.model_server = model_server
        self.debug_log = []
        
    def debug_inference(self, input_data):
        """Debug model inference step by step"""
        
        debug_info = {
            'input_shape': input_data.shape,
            'input_dtype': input_data.dtype,
            'input_range': (input_data.min().item(), input_data.max().item())
        }
        
        # Debug preprocessing
        try:
            processed_input = self.model_server.preprocess(input_data)
            debug_info['preprocessing_success'] = True
            debug_info['processed_shape'] = processed_input.shape
        except Exception as e:
            debug_info['preprocessing_error'] = str(e)
            return debug_info
            
        # Debug inference
        try:
            with torch.no_grad():
                predictions = self.model_server.model(processed_input)
            debug_info['inference_success'] = True
            debug_info['output_shape'] = predictions.shape
        except Exception as e:
            debug_info['inference_error'] = str(e)
            return debug_info
            
        # Debug postprocessing
        try:
            final_output = self.model_server.postprocess(predictions)
            debug_info['postprocessing_success'] = True
            debug_info['final_output'] = final_output
        except Exception as e:
            debug_info['postprocessing_error'] = str(e)
            
        return debug_info
```

### Performance Profiling

```python
import cProfile
import pstats

class PerformanceProfiler:
    def __init__(self):
        self.profiler = cProfile.Profile()
        
    def profile_inference(self, func):
        """Profile inference function"""
        def wrapper(*args, **kwargs):
            self.profiler.enable()
            result = func(*args, **kwargs)
            self.profiler.disable()
            
            # Get profiling stats
            stats = pstats.Stats(self.profiler)
            stats.sort_stats('cumulative')
            
            # Log top functions by time
            top_functions = []
            for func_name, (cc, nc, tt, ct, callers) in stats.stats.items():
                if ct > 0.01:  # Only log functions taking > 1% of time
                    top_functions.append({
                        'function': func_name,
                        'cumulative_time': ct,
                        'total_time': tt,
                        'call_count': cc
                    })
                    
            return result, top_functions
        return wrapper
```

## Best Practices

### 1. Model Optimization
- Quantize models for faster inference
- Use model compilation for optimization
- Implement proper caching strategies

### 2. Monitoring and Observability
- Set up comprehensive monitoring
- Implement health checks
- Track performance metrics

### 3. Deployment Strategy
- Use blue-green or canary deployments
- Implement proper rollback procedures
- Test deployments in staging first

### 4. Security and Reliability
- Implement proper authentication
- Use HTTPS for API endpoints
- Set up rate limiting and throttling

## Common Patterns

### Model Versioning

```python
class ModelVersionManager:
    def __init__(self):
        self.versions = {}
        
    def register_model(self, version, model_path, metadata):
        """Register a new model version"""
        self.versions[version] = {
            'path': model_path,
            'metadata': metadata,
            'created_at': time.time()
        }
        
    def get_model_info(self, version):
        """Get information about a model version"""
        return self.versions.get(version)
        
    def list_versions(self):
        """List all available model versions"""
        return list(self.versions.keys())
```

### Configuration Management

```python
class DeploymentConfig:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        
    def load_config(self, config_path):
        """Load deployment configuration"""
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def get_model_config(self, model_name):
        """Get configuration for specific model"""
        return self.config.get('models', {}).get(model_name, {})
        
    def get_deployment_config(self, deployment_type):
        """Get deployment-specific configuration"""
        return self.config.get('deployments', {}).get(deployment_type, {})
```

## Next Steps

- Read [Performance and Scaling](performance-scaling) for optimization techniques
- Explore [Custom Loss Functions](custom-loss-functions) for advanced loss design
- Check [Model Validation](model-validation) for evaluation frameworks
- Review [Algorithm Implementation](algorithm-implementation) for custom algorithms 