---
description: "Implement comprehensive monitoring and alerting systems for production model performance and health"
tags: ["monitoring", "alerting", "production", "metrics", "health-checks"]
categories: ["production-deployment"]
---

# Monitoring & Alerting

This guide covers how to implement comprehensive monitoring and alerting systems for NeMo RL production deployments.

## Overview

Effective monitoring and alerting are crucial for maintaining reliable production deployments. NeMo RL provides tools and frameworks for monitoring model performance, system health, and business metrics.

## Key Components

### Metrics Collection

Implement comprehensive metrics collection:

```python
import time
import psutil
import torch
from prometheus_client import Counter, Histogram, Gauge, Summary
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    inference_time: float
    memory_usage: float
    gpu_utilization: float
    throughput: float
    error_rate: float

class MetricsCollector:
    def __init__(self):
        # Request metrics
        self.request_counter = Counter(
            'nemo_rl_requests_total',
            'Total number of requests',
            ['endpoint', 'status', 'model_version']
        )
        
        self.request_duration = Histogram(
            'nemo_rl_request_duration_seconds',
            'Request duration in seconds',
            ['endpoint', 'model_version']
        )
        
        # Model performance metrics
        self.inference_time = Histogram(
            'nemo_rl_inference_time_seconds',
            'Model inference time in seconds',
            ['model_version']
        )
        
        self.memory_usage = Gauge(
            'nemo_rl_memory_usage_bytes',
            'Memory usage in bytes',
            ['device', 'model_version']
        )
        
        self.gpu_utilization = Gauge(
            'nemo_rl_gpu_utilization_percent',
            'GPU utilization percentage',
            ['gpu_id', 'model_version']
        )
        
        # Business metrics
        self.response_quality = Histogram(
            'nemo_rl_response_quality_score',
            'Response quality score',
            ['model_version']
        )
        
        self.user_satisfaction = Counter(
            'nemo_rl_user_satisfaction_total',
            'User satisfaction ratings',
            ['rating', 'model_version']
        )
    
    def record_request(self, endpoint: str, status: str, duration: float, model_version: str):
        """
        Record request metrics
        """
        self.request_counter.labels(
            endpoint=endpoint,
            status=status,
            model_version=model_version
        ).inc()
        
        self.request_duration.labels(
            endpoint=endpoint,
            model_version=model_version
        ).observe(duration)
    
    def record_inference(self, duration: float, model_version: str):
        """
        Record inference metrics
        """
        self.inference_time.labels(model_version=model_version).observe(duration)
    
    def update_system_metrics(self, model_version: str):
        """
        Update system metrics
        """
        # Memory usage
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated()
            self.memory_usage.labels(
                device='gpu',
                model_version=model_version
            ).set(gpu_memory)
        
        # CPU memory
        cpu_memory = psutil.virtual_memory().used
        self.memory_usage.labels(
            device='cpu',
            model_version=model_version
        ).set(cpu_memory)
        
        # GPU utilization
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_util = torch.cuda.utilization(i)
                self.gpu_utilization.labels(
                    gpu_id=i,
                    model_version=model_version
                ).set(gpu_util)
    
    def record_quality_metrics(self, quality_score: float, model_version: str):
        """
        Record quality metrics
        """
        self.response_quality.labels(model_version=model_version).observe(quality_score)
    
    def record_user_satisfaction(self, rating: int, model_version: str):
        """
        Record user satisfaction
        """
        self.user_satisfaction.labels(
            rating=str(rating),
            model_version=model_version
        ).inc()
```

### Health Checks

Implement comprehensive health checks:

```python
import asyncio
import aiohttp
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class HealthCheck:
    name: str
    status: HealthStatus
    message: str
    timestamp: float
    duration: float

class HealthChecker:
    def __init__(self, config: Dict):
        self.config = config
        self.health_checks = []
        self.check_interval = config.get('check_interval', 30)
        
    async def perform_health_checks(self) -> List[HealthCheck]:
        """
        Perform all health checks
        """
        checks = []
        
        # Model health check
        model_check = await self.check_model_health()
        checks.append(model_check)
        
        # System health check
        system_check = await self.check_system_health()
        checks.append(system_check)
        
        # Database health check
        db_check = await self.check_database_health()
        checks.append(db_check)
        
        # External service health check
        external_check = await self.check_external_services()
        checks.append(external_check)
        
        self.health_checks = checks
        return checks
    
    async def check_model_health(self) -> HealthCheck:
        """
        Check model health
        """
        start_time = time.time()
        
        try:
            # Test model inference
            test_input = "Hello, how are you?"
            result = await self.model.generate_text(test_input)
            
            # Check if result is valid
            if result and len(result) > 0:
                status = HealthStatus.HEALTHY
                message = "Model inference working correctly"
            else:
                status = HealthStatus.DEGRADED
                message = "Model inference returned empty result"
        
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"Model inference failed: {str(e)}"
        
        duration = time.time() - start_time
        
        return HealthCheck(
            name="model_health",
            status=status,
            message=message,
            timestamp=start_time,
            duration=duration
        )
    
    async def check_system_health(self) -> HealthCheck:
        """
        Check system health
        """
        start_time = time.time()
        
        try:
            # Check CPU usage
            cpu_percent = psutil.cpu_percent()
            
            # Check memory usage
            memory_percent = psutil.virtual_memory().percent
            
            # Check disk usage
            disk_percent = psutil.disk_usage('/').percent
            
            # Determine status based on thresholds
            if cpu_percent > 90 or memory_percent > 90 or disk_percent > 90:
                status = HealthStatus.UNHEALTHY
                message = f"High resource usage - CPU: {cpu_percent}%, Memory: {memory_percent}%, Disk: {disk_percent}%"
            elif cpu_percent > 70 or memory_percent > 70 or disk_percent > 70:
                status = HealthStatus.DEGRADED
                message = f"Elevated resource usage - CPU: {cpu_percent}%, Memory: {memory_percent}%, Disk: {disk_percent}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"System resources normal - CPU: {cpu_percent}%, Memory: {memory_percent}%, Disk: {disk_percent}%"
        
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"System health check failed: {str(e)}"
        
        duration = time.time() - start_time
        
        return HealthCheck(
            name="system_health",
            status=status,
            message=message,
            timestamp=start_time,
            duration=duration
        )
    
    async def check_database_health(self) -> HealthCheck:
        """
        Check database health
        """
        start_time = time.time()
        
        try:
            # Test database connection
            # This is a simplified example - implement based on your database
            db_connection_ok = await self.test_database_connection()
            
            if db_connection_ok:
                status = HealthStatus.HEALTHY
                message = "Database connection healthy"
            else:
                status = HealthStatus.UNHEALTHY
                message = "Database connection failed"
        
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"Database health check failed: {str(e)}"
        
        duration = time.time() - start_time
        
        return HealthCheck(
            name="database_health",
            status=status,
            message=message,
            timestamp=start_time,
            duration=duration
        )
    
    async def check_external_services(self) -> HealthCheck:
        """
        Check external services health
        """
        start_time = time.time()
        
        try:
            # Check external API endpoints
            external_services = self.config.get('external_services', [])
            failed_services = []
            
            for service in external_services:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(service['health_url'], timeout=5) as response:
                            if response.status != 200:
                                failed_services.append(service['name'])
                except Exception:
                    failed_services.append(service['name'])
            
            if not failed_services:
                status = HealthStatus.HEALTHY
                message = "All external services healthy"
            elif len(failed_services) < len(external_services):
                status = HealthStatus.DEGRADED
                message = f"Some external services down: {', '.join(failed_services)}"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"All external services down: {', '.join(failed_services)}"
        
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"External services health check failed: {str(e)}"
        
        duration = time.time() - start_time
        
        return HealthCheck(
            name="external_services_health",
            status=status,
            message=message,
            timestamp=start_time,
            duration=duration
        )
```

## Configuration

### Monitoring Configuration

```yaml
# configs/monitoring.yaml
monitoring:
  enabled: true
  
  # Metrics collection
  metrics:
    enabled: true
    endpoint: "/metrics"
    collection_interval: 10  # seconds
    
  # Health checks
  health_checks:
    enabled: true
    check_interval: 30  # seconds
    timeout: 10  # seconds
    
  # Alerting
  alerting:
    enabled: true
    alert_manager_url: "http://alertmanager:9093"
    
  # Logging
  logging:
    level: "INFO"
    format: "json"
    output: "stdout"
    
  # External services
  external_services:
    - name: "redis"
      health_url: "http://redis:6379/health"
    - name: "database"
      health_url: "http://database:5432/health"
```

### Alerting Configuration

```yaml
# configs/alerting.yaml
alerting:
  # Alert rules
  rules:
    - name: "high_error_rate"
      condition: "error_rate > 0.05"
      severity: "critical"
      duration: "5m"
      
    - name: "high_latency"
      condition: "p95_latency > 2.0"
      severity: "warning"
      duration: "2m"
      
    - name: "high_memory_usage"
      condition: "memory_usage > 0.9"
      severity: "critical"
      duration: "1m"
      
    - name: "model_degradation"
      condition: "quality_score < 0.7"
      severity: "warning"
      duration: "10m"
  
  # Notification channels
  notifications:
    - type: "slack"
      webhook_url: "https://hooks.slack.com/services/..."
      channel: "#alerts"
      
    - type: "email"
      smtp_server: "smtp.gmail.com"
      smtp_port: 587
      username: "alerts@company.com"
      password: "password"
      
    - type: "pagerduty"
      service_key: "your-service-key"
```

## Advanced Monitoring

### Custom Metrics

Implement custom business metrics:

```python
class BusinessMetricsCollector:
    def __init__(self):
        # Custom business metrics
        self.user_engagement = Counter(
            'nemo_rl_user_engagement_total',
            'User engagement events',
            ['event_type', 'user_segment']
        )
        
        self.model_performance = Histogram(
            'nemo_rl_model_performance_score',
            'Model performance scores',
            ['metric_type', 'model_version']
        )
        
        self.business_impact = Gauge(
            'nemo_rl_business_impact_score',
            'Business impact metrics',
            ['impact_type', 'time_period']
        )
    
    def record_user_engagement(self, event_type: str, user_segment: str):
        """
        Record user engagement events
        """
        self.user_engagement.labels(
            event_type=event_type,
            user_segment=user_segment
        ).inc()
    
    def record_model_performance(self, metric_type: str, score: float, model_version: str):
        """
        Record model performance metrics
        """
        self.model_performance.labels(
            metric_type=metric_type,
            model_version=model_version
        ).observe(score)
    
    def update_business_impact(self, impact_type: str, score: float, time_period: str):
        """
        Update business impact metrics
        """
        self.business_impact.labels(
            impact_type=impact_type,
            time_period=time_period
        ).set(score)
```

### Performance Profiling

Implement detailed performance profiling:

```python
import cProfile
import pstats
from functools import wraps

class PerformanceProfiler:
    def __init__(self):
        self.profiler = cProfile.Profile()
        self.stats = None
    
    def profile_function(self, func):
        """
        Decorator to profile a function
        """
        @wraps(func)
        async def wrapper(*args, **kwargs):
            self.profiler.enable()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                self.profiler.disable()
                self.stats = pstats.Stats(self.profiler)
        
        return wrapper
    
    def get_profile_stats(self) -> Dict[str, Any]:
        """
        Get profiling statistics
        """
        if self.stats is None:
            return {}
        
        # Get top functions by cumulative time
        stats_data = {}
        for func, (cc, nc, tt, ct, callers) in self.stats.stats.items():
            stats_data[func] = {
                'calls': nc,
                'total_time': tt,
                'cumulative_time': ct,
                'time_per_call': tt / nc if nc > 0 else 0
            }
        
        return stats_data
    
    def print_profile_summary(self):
        """
        Print profiling summary
        """
        if self.stats:
            self.stats.sort_stats('cumulative')
            self.stats.print_stats(10)  # Top 10 functions

class ProfiledModelServer:
    def __init__(self, model_path: str):
        self.model = load_model(model_path)
        self.profiler = PerformanceProfiler()
    
    @PerformanceProfiler.profile_function
    async def generate_text(self, request: PredictionRequest) -> PredictionResponse:
        """
        Generate text with performance profiling
        """
        # Your existing generation logic here
        pass
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get performance profiling report
        """
        return self.profiler.get_profile_stats()
```

### Distributed Tracing

Implement distributed tracing for complex deployments:

```python
import opentracing
import jaeger_client
from opentracing.ext import tags

class TracingModelServer:
    def __init__(self, model_path: str, jaeger_host: str = "localhost"):
        self.model = load_model(model_path)
        
        # Initialize Jaeger tracer
        config = jaeger_client.Config(
            config={
                'sampler': {'type': 'const', 'param': True},
                'local_agent': {'reporting_host': jaeger_host},
                'logging': True,
            },
            service_name='nemo-rl-model-server'
        )
        self.tracer = config.initialize_tracer()
    
    async def generate_text_traced(self, request: PredictionRequest, parent_span=None) -> PredictionResponse:
        """
        Generate text with distributed tracing
        """
        # Start span
        span = self.tracer.start_span(
            'generate_text',
            child_of=parent_span
        )
        
        try:
            # Add request metadata to span
            span.set_tag('request.prompt_length', len(request.prompt))
            span.set_tag('request.max_length', request.max_length)
            span.set_tag('request.temperature', request.temperature)
            
            # Tokenize with tracing
            tokenize_span = self.tracer.start_span('tokenize', child_of=span)
            inputs = self.tokenizer(
                request.prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            tokenize_span.finish()
            
            # Generate with tracing
            generate_span = self.tracer.start_span('generate', child_of=span)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=request.max_length,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    do_sample=True
                )
            generate_span.finish()
            
            # Decode with tracing
            decode_span = self.tracer.start_span('decode', child_of=span)
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            decode_span.finish()
            
            # Add result metadata to span
            span.set_tag('response.length', len(generated_text))
            span.set_tag('response.tokens_used', len(outputs[0]))
            
            return PredictionResponse(
                generated_text=generated_text,
                confidence=0.85,
                tokens_used=len(outputs[0])
            )
        
        except Exception as e:
            # Record error in span
            span.set_tag(tags.ERROR, True)
            span.log_kv({'event': 'error', 'error.object': e})
            raise
        finally:
            span.finish()
```

## Alerting Systems

### Alert Rules Engine

Implement a flexible alert rules engine:

```python
from typing import Dict, List, Any
import asyncio
import json

class AlertRule:
    def __init__(self, name: str, condition: str, severity: str, duration: str):
        self.name = name
        self.condition = condition
        self.severity = severity
        self.duration = duration
        self.triggered = False
        self.trigger_time = None
    
    def evaluate(self, metrics: Dict[str, Any]) -> bool:
        """
        Evaluate alert condition
        """
        # Simple condition evaluation - in practice, use a proper expression evaluator
        try:
            # Parse condition (simplified)
            if "error_rate > 0.05" in self.condition:
                return metrics.get('error_rate', 0) > 0.05
            elif "p95_latency > 2.0" in self.condition:
                return metrics.get('p95_latency', 0) > 2.0
            elif "memory_usage > 0.9" in self.condition:
                return metrics.get('memory_usage', 0) > 0.9
            else:
                return False
        except Exception:
            return False

class AlertManager:
    def __init__(self, config: Dict):
        self.config = config
        self.rules = self.load_alert_rules()
        self.notification_channels = self.load_notification_channels()
        self.alert_history = []
    
    def load_alert_rules(self) -> List[AlertRule]:
        """
        Load alert rules from configuration
        """
        rules = []
        for rule_config in self.config.get('rules', []):
            rule = AlertRule(
                name=rule_config['name'],
                condition=rule_config['condition'],
                severity=rule_config['severity'],
                duration=rule_config['duration']
            )
            rules.append(rule)
        return rules
    
    def load_notification_channels(self) -> List[Dict]:
        """
        Load notification channels from configuration
        """
        return self.config.get('notifications', [])
    
    async def evaluate_alerts(self, metrics: Dict[str, Any]):
        """
        Evaluate all alert rules
        """
        for rule in self.rules:
            is_triggered = rule.evaluate(metrics)
            
            if is_triggered and not rule.triggered:
                # Alert just triggered
                rule.triggered = True
                rule.trigger_time = time.time()
                await self.send_alert(rule, metrics)
            
            elif not is_triggered and rule.triggered:
                # Alert resolved
                rule.triggered = False
                rule.trigger_time = None
                await self.send_resolution(rule, metrics)
    
    async def send_alert(self, rule: AlertRule, metrics: Dict[str, Any]):
        """
        Send alert notification
        """
        alert_message = {
            'alert_name': rule.name,
            'severity': rule.severity,
            'message': f"Alert {rule.name} triggered: {rule.condition}",
            'metrics': metrics,
            'timestamp': time.time()
        }
        
        # Send to all notification channels
        for channel in self.notification_channels:
            await self.send_notification(channel, alert_message)
        
        # Store in history
        self.alert_history.append(alert_message)
    
    async def send_resolution(self, rule: AlertRule, metrics: Dict[str, Any]):
        """
        Send alert resolution notification
        """
        resolution_message = {
            'alert_name': rule.name,
            'severity': rule.severity,
            'message': f"Alert {rule.name} resolved",
            'metrics': metrics,
            'timestamp': time.time()
        }
        
        # Send to all notification channels
        for channel in self.notification_channels:
            await self.send_notification(channel, resolution_message)
    
    async def send_notification(self, channel: Dict, message: Dict):
        """
        Send notification to a specific channel
        """
        channel_type = channel.get('type')
        
        if channel_type == 'slack':
            await self.send_slack_notification(channel, message)
        elif channel_type == 'email':
            await self.send_email_notification(channel, message)
        elif channel_type == 'pagerduty':
            await self.send_pagerduty_notification(channel, message)
    
    async def send_slack_notification(self, channel: Dict, message: Dict):
        """
        Send Slack notification
        """
        import aiohttp
        
        webhook_url = channel.get('webhook_url')
        if not webhook_url:
            return
        
        slack_message = {
            "text": f"🚨 *{message['severity'].upper()} Alert*: {message['message']}",
            "attachments": [{
                "fields": [
                    {"title": "Alert Name", "value": message['alert_name'], "short": True},
                    {"title": "Severity", "value": message['severity'], "short": True},
                    {"title": "Timestamp", "value": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(message['timestamp'])), "short": True}
                ]
            }]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=slack_message) as response:
                if response.status != 200:
                    print(f"Failed to send Slack notification: {response.status}")
```

## Best Practices

### 1. Comprehensive Monitoring

Implement monitoring at all levels:

```python
class ComprehensiveMonitor:
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.health_checker = HealthChecker({})
        self.alert_manager = AlertManager({})
        self.business_metrics = BusinessMetricsCollector()
    
    async def run_monitoring_cycle(self):
        """
        Run complete monitoring cycle
        """
        # Collect metrics
        metrics = await self.collect_all_metrics()
        
        # Perform health checks
        health_checks = await self.health_checker.perform_health_checks()
        
        # Evaluate alerts
        await self.alert_manager.evaluate_alerts(metrics)
        
        # Update business metrics
        self.update_business_metrics(metrics)
        
        # Log monitoring summary
        self.log_monitoring_summary(metrics, health_checks)
    
    async def collect_all_metrics(self) -> Dict[str, Any]:
        """
        Collect all system metrics
        """
        metrics = {}
        
        # System metrics
        metrics['cpu_usage'] = psutil.cpu_percent()
        metrics['memory_usage'] = psutil.virtual_memory().percent
        metrics['disk_usage'] = psutil.disk_usage('/').percent
        
        # Model metrics
        if torch.cuda.is_available():
            metrics['gpu_memory'] = torch.cuda.memory_allocated() / 1024**3
            metrics['gpu_utilization'] = torch.cuda.utilization()
        
        # Request metrics
        metrics['request_rate'] = self.metrics_collector.request_counter._value.sum()
        metrics['error_rate'] = self.calculate_error_rate()
        metrics['p95_latency'] = self.calculate_p95_latency()
        
        return metrics
```

### 2. Alert Fatigue Prevention

Implement alert fatigue prevention:

```python
class AlertFatiguePrevention:
    def __init__(self):
        self.alert_cooldowns = {}
        self.escalation_rules = {}
        self.alert_grouping = {}
    
    def should_send_alert(self, alert_name: str, severity: str) -> bool:
        """
        Determine if alert should be sent based on cooldown and escalation rules
        """
        current_time = time.time()
        
        # Check cooldown
        if alert_name in self.alert_cooldowns:
            last_alert_time = self.alert_cooldowns[alert_name]
            cooldown_duration = self.get_cooldown_duration(severity)
            
            if current_time - last_alert_time < cooldown_duration:
                return False
        
        # Update cooldown
        self.alert_cooldowns[alert_name] = current_time
        return True
    
    def get_cooldown_duration(self, severity: str) -> int:
        """
        Get cooldown duration based on severity
        """
        cooldowns = {
            'critical': 300,  # 5 minutes
            'warning': 1800,  # 30 minutes
            'info': 3600      # 1 hour
        }
        return cooldowns.get(severity, 1800)
    
    def group_similar_alerts(self, alerts: List[Dict]) -> List[Dict]:
        """
        Group similar alerts to reduce noise
        """
        grouped_alerts = {}
        
        for alert in alerts:
            # Create group key based on alert type and severity
            group_key = f"{alert['alert_name']}_{alert['severity']}"
            
            if group_key not in grouped_alerts:
                grouped_alerts[group_key] = {
                    'alert_name': alert['alert_name'],
                    'severity': alert['severity'],
                    'count': 1,
                    'first_occurrence': alert['timestamp'],
                    'last_occurrence': alert['timestamp']
                }
            else:
                grouped_alerts[group_key]['count'] += 1
                grouped_alerts[group_key]['last_occurrence'] = alert['timestamp']
        
        return list(grouped_alerts.values())
```

### 3. Monitoring Dashboard

Create monitoring dashboard endpoints:

```python
from fastapi import APIRouter
from typing import Dict, List

router = APIRouter()

@router.get("/dashboard/overview")
async def get_dashboard_overview() -> Dict[str, Any]:
    """
    Get dashboard overview
    """
    return {
        "system_health": await get_system_health_summary(),
        "model_performance": await get_model_performance_summary(),
        "business_metrics": await get_business_metrics_summary(),
        "recent_alerts": await get_recent_alerts()
    }

@router.get("/dashboard/metrics")
async def get_metrics_dashboard() -> Dict[str, Any]:
    """
    Get detailed metrics dashboard
    """
    return {
        "request_metrics": {
            "total_requests": get_total_requests(),
            "error_rate": get_error_rate(),
            "average_latency": get_average_latency(),
            "p95_latency": get_p95_latency()
        },
        "system_metrics": {
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "gpu_usage": get_gpu_usage()
        },
        "model_metrics": {
            "inference_time": get_average_inference_time(),
            "throughput": get_requests_per_second(),
            "memory_usage": get_model_memory_usage()
        }
    }

@router.get("/dashboard/alerts")
async def get_alerts_dashboard() -> Dict[str, Any]:
    """
    Get alerts dashboard
    """
    return {
        "active_alerts": get_active_alerts(),
        "alert_history": get_alert_history(),
        "alert_statistics": get_alert_statistics()
    }
```

## Troubleshooting

### Common Monitoring Issues

1. **High Cardinality**: Limit metric labels and use aggregation
2. **Memory Leaks**: Monitor memory usage and implement cleanup
3. **Alert Storms**: Implement alert grouping and cooldowns

### Debugging Tips

```python
# Add debugging to monitoring
def debug_monitoring_system(self):
    """
    Debug monitoring system issues
    """
    print("=== Monitoring System Debug ===")
    
    # Check metrics collection
    print(f"Metrics collector initialized: {self.metrics_collector is not None}")
    print(f"Health checker initialized: {self.health_checker is not None}")
    print(f"Alert manager initialized: {self.alert_manager is not None}")
    
    # Check system resources
    print(f"CPU usage: {psutil.cpu_percent()}%")
    print(f"Memory usage: {psutil.virtual_memory().percent}%")
    
    # Check external dependencies
    print("Checking external dependencies...")
    # Add checks for database, Redis, etc.
    
    print("==============================")
```

## Next Steps

- Learn about [Model Serving](model-serving) for deployment strategies
- Review [Performance & Scaling](../performance-scaling/index) for optimization
- Explore [Algorithm Customization](../algorithm-customization/index) for advanced training 