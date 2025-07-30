---
description: "Deploy models with REST/gRPC APIs and serving architectures for production use"
tags: ["deployment", "serving", "api", "rest", "grpc", "production"]
categories: ["production-deployment"]
---

# Model Serving

This guide covers how to deploy NeMo RL models to production with REST/gRPC APIs and serving architectures.

## Overview

NeMo RL provides comprehensive model serving capabilities that allow you to deploy trained models in production environments with high availability, scalability, and performance.

## Key Components

### REST API Serving

Deploy models with REST APIs using FastAPI:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from nemo_rl.models import load_model
from nemo_rl.serving import ModelServer

app = FastAPI(title="NeMo RL Model Serving API")

class PredictionRequest(BaseModel):
    prompt: str
    max_length: int = 100
    temperature: float = 0.7
    top_p: float = 0.9

class PredictionResponse(BaseModel):
    generated_text: str
    confidence: float
    tokens_used: int

class ModelServer:
    def __init__(self, model_path: str):
        self.model = load_model(model_path)
        self.tokenizer = self.model.tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    async def generate_text(self, request: PredictionRequest) -> PredictionResponse:
        """
        Generate text using the deployed model
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(
                request.prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate text
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=request.max_length,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode output
            generated_text = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            # Calculate confidence (simplified)
            confidence = self.calculate_confidence(outputs[0])
            
            return PredictionResponse(
                generated_text=generated_text,
                confidence=confidence,
                tokens_used=len(outputs[0])
            )
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    def calculate_confidence(self, token_ids):
        """
        Calculate confidence score for generated tokens
        """
        # Simplified confidence calculation
        return 0.85  # Placeholder

# Initialize model server
model_server = ModelServer("path/to/trained/model")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Generate text prediction endpoint
    """
    return await model_server.generate_text(request)

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "model_loaded": True}

@app.get("/model_info")
async def model_info():
    """
    Get model information
    """
    return {
        "model_name": model_server.model.__class__.__name__,
        "parameters": sum(p.numel() for p in model_server.model.parameters()),
        "device": str(model_server.device)
    }
```

### gRPC Serving

Deploy models with gRPC for high-performance serving:

```python
import grpc
from concurrent import futures
import nemo_rl_pb2
import nemo_rl_pb2_grpc
from nemo_rl.models import load_model
import torch

class NeMoRLServicer(nemo_rl_pb2_grpc.NeMoRLServicer):
    def __init__(self, model_path: str):
        self.model = load_model(model_path)
        self.tokenizer = self.model.tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    def GenerateText(self, request, context):
        """
        Generate text using gRPC
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(
                request.prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate text
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=request.max_length,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode output
            generated_text = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            return nemo_rl_pb2.PredictionResponse(
                generated_text=generated_text,
                confidence=0.85,
                tokens_used=len(outputs[0])
            )
        
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return nemo_rl_pb2.PredictionResponse()
    
    def HealthCheck(self, request, context):
        """
        Health check endpoint
        """
        return nemo_rl_pb2.HealthResponse(
            status="healthy",
            model_loaded=True
        )

def serve():
    """
    Start gRPC server
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    nemo_rl_pb2_grpc.add_NeMoRLServicer_to_server(
        NeMoRLServicer("path/to/model"), 
        server
    )
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
```

## Configuration

### Model Serving Configuration

```yaml
# configs/model_serving.yaml
serving:
  enabled: true
  
  # API configuration
  api:
    type: "rest"  # or "grpc"
    host: "0.0.0.0"
    port: 8000
    workers: 4
    
  # Model configuration
  model:
    path: "models/trained_model"
    device: "auto"  # auto, cpu, cuda
    batch_size: 1
    max_concurrent_requests: 10
    
  # Performance settings
  performance:
    enable_batching: true
    max_batch_size: 8
    timeout_seconds: 30
    
  # Monitoring
  monitoring:
    enabled: true
    metrics_endpoint: "/metrics"
    health_check_interval: 30
```

### Production Deployment Configuration

```yaml
# configs/production_deployment.yaml
deployment:
  # Load balancer configuration
  load_balancer:
    enabled: true
    type: "nginx"  # or "haproxy"
    upstream_servers:
      - "http://server1:8000"
      - "http://server2:8000"
      - "http://server3:8000"
    
  # Auto-scaling
  auto_scaling:
    enabled: true
    min_replicas: 2
    max_replicas: 10
    target_cpu_utilization: 70
    target_memory_utilization: 80
    
  # Health checks
  health_checks:
    enabled: true
    path: "/health"
    interval: 30
    timeout: 5
    failure_threshold: 3
```

## Advanced Serving Strategies

### Model Batching

Implement efficient model batching for high-throughput serving:

```python
import asyncio
from typing import List, Dict
import torch

class BatchedModelServer:
    def __init__(self, model_path: str, max_batch_size: int = 8):
        self.model = load_model(model_path)
        self.tokenizer = self.model.tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        self.max_batch_size = max_batch_size
        self.pending_requests = []
        self.batch_timeout = 0.1  # seconds
        
    async def process_batch(self, requests: List[Dict]) -> List[Dict]:
        """
        Process a batch of requests
        """
        # Prepare batch
        prompts = [req['prompt'] for req in requests]
        max_lengths = [req.get('max_length', 100) for req in requests]
        temperatures = [req.get('temperature', 0.7) for req in requests]
        
        # Tokenize batch
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generate batch
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max(max_lengths),
                temperature=temperatures[0],  # Use first temperature for batch
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode outputs
        results = []
        for i, output in enumerate(outputs):
            generated_text = self.tokenizer.decode(
                output, 
                skip_special_tokens=True
            )
            results.append({
                'generated_text': generated_text,
                'confidence': 0.85,
                'tokens_used': len(output)
            })
        
        return results
    
    async def add_request(self, request: Dict) -> asyncio.Future:
        """
        Add request to batch and return future
        """
        future = asyncio.Future()
        self.pending_requests.append((request, future))
        
        # Process batch if full or timeout
        if len(self.pending_requests) >= self.max_batch_size:
            await self.process_pending_batch()
        else:
            # Schedule timeout processing
            asyncio.create_task(self.process_batch_with_timeout())
        
        return future
    
    async def process_batch_with_timeout(self):
        """
        Process batch after timeout
        """
        await asyncio.sleep(self.batch_timeout)
        if self.pending_requests:
            await self.process_pending_batch()
    
    async def process_pending_batch(self):
        """
        Process all pending requests
        """
        if not self.pending_requests:
            return
        
        # Extract requests and futures
        requests = [req for req, _ in self.pending_requests]
        futures = [future for _, future in self.pending_requests]
        
        # Clear pending requests
        self.pending_requests = []
        
        # Process batch
        try:
            results = await self.process_batch(requests)
            
            # Set results for all futures
            for future, result in zip(futures, results):
                future.set_result(result)
        
        except Exception as e:
            # Set exception for all futures
            for future in futures:
                future.set_exception(e)
```

### Model Caching

Implement intelligent model caching:

```python
import redis
import pickle
import hashlib
from typing import Optional

class ModelCache:
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port)
        self.cache_ttl = 3600  # 1 hour
    
    def generate_cache_key(self, prompt: str, params: Dict) -> str:
        """
        Generate cache key for request
        """
        # Create hash of prompt and parameters
        cache_data = {
            'prompt': prompt,
            'max_length': params.get('max_length', 100),
            'temperature': params.get('temperature', 0.7),
            'top_p': params.get('top_p', 0.9)
        }
        
        cache_string = str(cache_data)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def get_cached_result(self, cache_key: str) -> Optional[Dict]:
        """
        Get cached result if available
        """
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return pickle.loads(cached_data)
        except Exception as e:
            print(f"Cache retrieval error: {e}")
        
        return None
    
    def cache_result(self, cache_key: str, result: Dict):
        """
        Cache result
        """
        try:
            serialized_result = pickle.dumps(result)
            self.redis_client.setex(
                cache_key, 
                self.cache_ttl, 
                serialized_result
            )
        except Exception as e:
            print(f"Cache storage error: {e}")

class CachedModelServer(ModelServer):
    def __init__(self, model_path: str, cache_enabled: bool = True):
        super().__init__(model_path)
        self.cache_enabled = cache_enabled
        if cache_enabled:
            self.cache = ModelCache()
    
    async def generate_text(self, request: PredictionRequest) -> PredictionResponse:
        """
        Generate text with caching
        """
        if self.cache_enabled:
            # Generate cache key
            cache_key = self.cache.generate_cache_key(
                request.prompt,
                {
                    'max_length': request.max_length,
                    'temperature': request.temperature,
                    'top_p': request.top_p
                }
            )
            
            # Check cache
            cached_result = self.cache.get_cached_result(cache_key)
            if cached_result:
                return PredictionResponse(**cached_result)
        
        # Generate new result
        result = await super().generate_text(request)
        
        # Cache result
        if self.cache_enabled:
            self.cache.cache_result(cache_key, result.dict())
        
        return result
```

### Load Balancing

Implement intelligent load balancing:

```python
import random
import time
from typing import List, Dict

class LoadBalancer:
    def __init__(self, servers: List[str]):
        self.servers = servers
        self.server_health = {server: True for server in servers}
        self.server_load = {server: 0 for server in servers}
        self.last_health_check = {server: time.time() for server in servers}
    
    def get_next_server(self) -> str:
        """
        Get next available server using round-robin with health checks
        """
        available_servers = [
            server for server in self.servers 
            if self.server_health[server]
        ]
        
        if not available_servers:
            # All servers down, return random server
            return random.choice(self.servers)
        
        # Select server with lowest load
        selected_server = min(
            available_servers,
            key=lambda s: self.server_load[s]
        )
        
        # Increment load
        self.server_load[selected_server] += 1
        
        return selected_server
    
    def mark_request_complete(self, server: str):
        """
        Mark request as complete for load tracking
        """
        if server in self.server_load:
            self.server_load[server] = max(0, self.server_load[server] - 1)
    
    async def health_check_server(self, server: str) -> bool:
        """
        Perform health check on server
        """
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{server}/health", timeout=5) as response:
                    is_healthy = response.status == 200
                    self.server_health[server] = is_healthy
                    self.last_health_check[server] = time.time()
                    return is_healthy
        except Exception as e:
            self.server_health[server] = False
            return False
    
    async def periodic_health_check(self):
        """
        Periodic health check of all servers
        """
        while True:
            for server in self.servers:
                await self.health_check_server(server)
            
            await asyncio.sleep(30)  # Check every 30 seconds

class LoadBalancedModelServer:
    def __init__(self, servers: List[str]):
        self.load_balancer = LoadBalancer(servers)
        self.session = None
    
    async def get_session(self):
        """
        Get aiohttp session
        """
        if self.session is None:
            import aiohttp
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def generate_text(self, request: PredictionRequest) -> PredictionResponse:
        """
        Generate text using load-balanced servers
        """
        session = await self.get_session()
        
        # Get next available server
        server = self.load_balancer.get_next_server()
        
        try:
            # Make request to selected server
            async with session.post(
                f"{server}/predict",
                json=request.dict()
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    self.load_balancer.mark_request_complete(server)
                    return PredictionResponse(**result)
                else:
                    raise Exception(f"Server error: {response.status}")
        
        except Exception as e:
            # Mark server as unhealthy
            self.load_balancer.server_health[server] = False
            raise e
```

## Monitoring and Observability

### Performance Monitoring

Implement comprehensive performance monitoring:

```python
import time
import psutil
from prometheus_client import Counter, Histogram, Gauge

class ModelServingMetrics:
    def __init__(self):
        # Request metrics
        self.request_counter = Counter(
            'nemo_rl_requests_total',
            'Total number of requests',
            ['endpoint', 'status']
        )
        
        self.request_duration = Histogram(
            'nemo_rl_request_duration_seconds',
            'Request duration in seconds',
            ['endpoint']
        )
        
        # Model metrics
        self.model_inference_time = Histogram(
            'nemo_rl_model_inference_seconds',
            'Model inference time in seconds'
        )
        
        self.model_memory_usage = Gauge(
            'nemo_rl_model_memory_bytes',
            'Model memory usage in bytes'
        )
        
        self.gpu_utilization = Gauge(
            'nemo_rl_gpu_utilization_percent',
            'GPU utilization percentage'
        )
    
    def record_request(self, endpoint: str, status: str, duration: float):
        """
        Record request metrics
        """
        self.request_counter.labels(endpoint=endpoint, status=status).inc()
        self.request_duration.labels(endpoint=endpoint).observe(duration)
    
    def record_inference_time(self, duration: float):
        """
        Record model inference time
        """
        self.model_inference_time.observe(duration)
    
    def update_system_metrics(self):
        """
        Update system metrics
        """
        # Memory usage
        memory_usage = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        self.model_memory_usage.set(memory_usage)
        
        # GPU utilization (simplified)
        if torch.cuda.is_available():
            gpu_util = torch.cuda.utilization()
            self.gpu_utilization.set(gpu_util)

class MonitoredModelServer(ModelServer):
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.metrics = ModelServingMetrics()
    
    async def generate_text(self, request: PredictionRequest) -> PredictionResponse:
        """
        Generate text with monitoring
        """
        start_time = time.time()
        
        try:
            # Record inference start
            inference_start = time.time()
            
            # Generate text
            result = await super().generate_text(request)
            
            # Record inference time
            inference_duration = time.time() - inference_start
            self.metrics.record_inference_time(inference_duration)
            
            # Record successful request
            total_duration = time.time() - start_time
            self.metrics.record_request("/predict", "success", total_duration)
            
            return result
        
        except Exception as e:
            # Record failed request
            total_duration = time.time() - start_time
            self.metrics.record_request("/predict", "error", total_duration)
            raise e
    
    async def update_metrics(self):
        """
        Update system metrics periodically
        """
        while True:
            self.metrics.update_system_metrics()
            await asyncio.sleep(10)  # Update every 10 seconds
```

## Best Practices

### 1. Resource Management

Implement proper resource management:

```python
class ResourceManager:
    def __init__(self, max_memory_gb: float = 8.0):
        self.max_memory_gb = max_memory_gb
        self.memory_threshold = 0.9
    
    def check_memory_usage(self) -> bool:
        """
        Check if memory usage is within limits
        """
        if torch.cuda.is_available():
            memory_usage = torch.cuda.memory_allocated() / 1024**3
            return memory_usage < (self.max_memory_gb * self.memory_threshold)
        return True
    
    def cleanup_memory(self):
        """
        Clean up memory
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def monitor_resources(self):
        """
        Monitor system resources
        """
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        if cpu_percent > 90 or memory_percent > 90:
            print(f"Warning: High resource usage - CPU: {cpu_percent}%, Memory: {memory_percent}%")
```

### 2. Error Handling

Implement robust error handling:

```python
class RobustModelServer(ModelServer):
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.error_count = 0
        self.max_errors = 10
    
    async def generate_text(self, request: PredictionRequest) -> PredictionResponse:
        """
        Generate text with robust error handling
        """
        try:
            # Validate input
            if not request.prompt or len(request.prompt.strip()) == 0:
                raise ValueError("Empty prompt provided")
            
            if request.max_length <= 0 or request.max_length > 1000:
                raise ValueError("Invalid max_length")
            
            if request.temperature < 0 or request.temperature > 2:
                raise ValueError("Invalid temperature")
            
            # Generate text
            result = await super().generate_text(request)
            
            # Reset error count on success
            self.error_count = 0
            
            return result
        
        except Exception as e:
            self.error_count += 1
            
            # Log error
            print(f"Error in generate_text: {e}")
            
            # Check if too many errors
            if self.error_count >= self.max_errors:
                print("Too many errors, restarting model server")
                self.restart_model()
            
            raise e
    
    def restart_model(self):
        """
        Restart model (simplified)
        """
        print("Restarting model...")
        # In a real implementation, you would reload the model
        self.error_count = 0
```

### 3. Security

Implement security measures:

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

class SecureModelServer(ModelServer):
    def __init__(self, model_path: str, secret_key: str):
        super().__init__(model_path)
        self.secret_key = secret_key
    
    def verify_token(self, credentials: HTTPAuthorizationCredentials = Depends(security)):
        """
        Verify JWT token
        """
        try:
            payload = jwt.decode(credentials.credentials, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    async def generate_text_secure(self, request: PredictionRequest, token: dict = Depends(verify_token):
        """
        Generate text with authentication
        """
        # Check user permissions
        if not self.check_user_permissions(token):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        
        return await self.generate_text(request)
    
    def check_user_permissions(self, token: dict) -> bool:
        """
        Check if user has required permissions
        """
        # Implement permission checking logic
        return token.get('role') in ['user', 'admin']
```

## Troubleshooting

### Common Issues

1. **High Latency**: Implement batching and caching
2. **Memory Issues**: Monitor memory usage and implement cleanup
3. **Load Balancing**: Ensure proper health checks and failover

### Debugging Tips

```python
# Add debugging to model serving
def debug_model_serving(self):
    """
    Debug model serving issues
    """
    print("=== Model Serving Debug ===")
    
    # Check model status
    print(f"Model loaded: {self.model is not None}")
    print(f"Model device: {self.device}")
    print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
    
    # Check memory usage
    if torch.cuda.is_available():
        memory_usage = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU memory usage: {memory_usage:.2f} GB")
    
    # Check tokenizer
    print(f"Tokenizer loaded: {self.tokenizer is not None}")
    print(f"Vocabulary size: {len(self.tokenizer)}")
    
    print("==========================")
```

## Next Steps

- Learn about [Monitoring & Alerting](monitoring-alerting) for production monitoring
- Review [Performance & Scaling](../performance-scaling/index) for optimization
- Explore [Algorithm Customization](../algorithm-customization/index) for advanced training 