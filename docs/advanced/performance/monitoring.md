# Performance Monitoring

This guide covers performance monitoring for NeMo RL training, including real-time metrics, alerting, and performance analysis tools.

## Overview

Performance monitoring is crucial for maintaining optimal training performance and detecting issues early. This guide covers monitoring tools, metrics, and best practices for NeMo RL training.

## Key Metrics

### Training Metrics

Monitor essential training metrics:

```python
import time
import torch
from collections import defaultdict

class TrainingMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_time = time.time()
    
    def record_metrics(self, step, loss, learning_rate, batch_size):
        """Record training metrics."""
        self.metrics['step'].append(step)
        self.metrics['loss'].append(loss.item())
        self.metrics['learning_rate'].append(learning_rate)
        self.metrics['batch_size'].append(batch_size)
        self.metrics['timestamp'].append(time.time())
    
    def get_training_stats(self):
        """Get training statistics."""
        if not self.metrics['loss']:
            return {}
        
        return {
            'current_loss': self.metrics['loss'][-1],
            'avg_loss': sum(self.metrics['loss']) / len(self.metrics['loss']),
            'min_loss': min(self.metrics['loss']),
            'max_loss': max(self.metrics['loss']),
            'total_steps': len(self.metrics['step']),
            'training_time': time.time() - self.start_time
        }
```

### Resource Metrics

Monitor system resources:

```python
def monitor_resources():
    """Monitor system resource usage."""
    import psutil
    import torch
    
    # CPU metrics
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()
    
    # Memory metrics
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    memory_available = memory.available / (1024**3)  # GB
    
    # GPU metrics
    gpu_metrics = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_metrics[f'gpu_{i}'] = {
                'memory_allocated': torch.cuda.memory_allocated(i) / (1024**2),
                'memory_reserved': torch.cuda.memory_reserved(i) / (1024**2),
                'max_memory_allocated': torch.cuda.max_memory_allocated(i) / (1024**2)
            }
    
    return {
        'cpu_percent': cpu_percent,
        'cpu_count': cpu_count,
        'memory_percent': memory_percent,
        'memory_available_gb': memory_available,
        'gpu_metrics': gpu_metrics
    }
```

### Ray Cluster Metrics

Monitor Ray cluster performance:

```python
import ray

def monitor_ray_cluster():
    """Monitor Ray cluster resources and performance."""
    # Get cluster resources
    cluster_resources = ray.cluster_resources()
    available_resources = ray.available_resources()
    
    # Calculate utilization
    cpu_utilization = 0
    gpu_utilization = 0
    
    if cluster_resources.get('CPU', 0) > 0:
        cpu_utilization = (cluster_resources['CPU'] - available_resources.get('CPU', 0)) / cluster_resources['CPU']
    
    if cluster_resources.get('GPU', 0) > 0:
        gpu_utilization = (cluster_resources['GPU'] - available_resources.get('GPU', 0)) / cluster_resources['GPU']
    
    return {
        'cluster_resources': cluster_resources,
        'available_resources': available_resources,
        'cpu_utilization': cpu_utilization,
        'gpu_utilization': gpu_utilization,
        'total_nodes': len(ray.nodes())
    }
```

## Monitoring Tools

### TensorBoard Integration

Use TensorBoard for training visualization:

```python
from torch.utils.tensorboard import SummaryWriter
import time

class TensorBoardMonitor:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        self.start_time = time.time()
    
    def log_training_metrics(self, step, metrics):
        """Log training metrics to TensorBoard."""
        for key, value in metrics.items():
            self.writer.add_scalar(f'training/{key}', value, step)
    
    def log_resource_metrics(self, step, metrics):
        """Log resource metrics to TensorBoard."""
        for key, value in metrics.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    self.writer.add_scalar(f'resources/{key}/{subkey}', subvalue, step)
            else:
                self.writer.add_scalar(f'resources/{key}', value, step)
    
    def log_model_parameters(self, step, model):
        """Log model parameter statistics."""
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f'parameters/{name}', param.data, step)
                self.writer.add_histogram(f'gradients/{name}', param.grad.data, step)
    
    def close(self):
        """Close TensorBoard writer."""
        self.writer.close()
```

### Prometheus Integration

Set up Prometheus monitoring:

```python
from prometheus_client import Counter, Gauge, Histogram, start_http_server
import time

class PrometheusMonitor:
    def __init__(self, port=8000):
        # Define metrics
        self.training_loss = Gauge('training_loss', 'Current training loss')
        self.training_steps = Counter('training_steps_total', 'Total training steps')
        self.gpu_memory = Gauge('gpu_memory_mb', 'GPU memory usage in MB', ['gpu_id'])
        self.cpu_usage = Gauge('cpu_usage_percent', 'CPU usage percentage')
        self.training_time = Histogram('training_step_duration_seconds', 'Training step duration')
        
        # Start HTTP server
        start_http_server(port)
    
    def update_metrics(self, loss, gpu_metrics, cpu_percent, step_duration):
        """Update Prometheus metrics."""
        self.training_loss.set(loss)
        self.training_steps.inc()
        self.cpu_usage.set(cpu_percent)
        self.training_time.observe(step_duration)
        
        # Update GPU metrics
        for gpu_id, metrics in gpu_metrics.items():
            self.gpu_memory.labels(gpu_id=gpu_id).set(metrics['memory_allocated'])
```

### Custom Monitoring Dashboard

Create a custom monitoring dashboard:

```python
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd

class MonitoringDashboard:
    def __init__(self):
        self.app = dash.Dash(__name__)
        self.metrics_data = []
        
        self.app.layout = html.Div([
            html.H1('NeMo RL Training Monitor'),
            
            dcc.Graph(id='loss-graph'),
            dcc.Graph(id='resource-graph'),
            
            dcc.Interval(
                id='interval-component',
                interval=5*1000,  # Update every 5 seconds
                n_intervals=0
            )
        ])
        
        @self.app.callback(
            [Output('loss-graph', 'figure'),
             Output('resource-graph', 'figure')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_graphs(n):
            return self.create_loss_graph(), self.create_resource_graph()
    
    def create_loss_graph(self):
        """Create loss visualization."""
        df = pd.DataFrame(self.metrics_data)
        
        return {
            'data': [
                go.Scatter(
                    x=df['timestamp'],
                    y=df['loss'],
                    mode='lines+markers',
                    name='Training Loss'
                )
            ],
            'layout': go.Layout(
                title='Training Loss Over Time',
                xaxis={'title': 'Time'},
                yaxis={'title': 'Loss'}
            )
        }
    
    def create_resource_graph(self):
        """Create resource usage visualization."""
        df = pd.DataFrame(self.metrics_data)
        
        return {
            'data': [
                go.Scatter(
                    x=df['timestamp'],
                    y=df['cpu_percent'],
                    mode='lines',
                    name='CPU Usage'
                ),
                go.Scatter(
                    x=df['timestamp'],
                    y=df['gpu_memory'],
                    mode='lines',
                    name='GPU Memory',
                    yaxis='y2'
                )
            ],
            'layout': go.Layout(
                title='Resource Usage',
                xaxis={'title': 'Time'},
                yaxis={'title': 'CPU Usage (%)'},
                yaxis2={'title': 'GPU Memory (MB)', 'overlaying': 'y', 'side': 'right'}
            )
        }
    
    def run(self, host='localhost', port=8050):
        """Run the monitoring dashboard."""
        self.app.run_server(host=host, port=port, debug=False)
```

## Alerting System

### Performance Alerts

Set up performance-based alerting:

```python
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

class PerformanceAlerting:
    def __init__(self, email_config):
        self.email_config = email_config
        self.alert_thresholds = {
            'loss_increase': 0.1,  # 10% loss increase
            'gpu_memory': 0.9,     # 90% GPU memory usage
            'cpu_usage': 0.95,     # 95% CPU usage
            'training_speed': 0.5   # 50% speed reduction
        }
    
    def check_alerts(self, current_metrics, baseline_metrics):
        """Check for performance alerts."""
        alerts = []
        
        # Check loss increase
        if current_metrics['loss'] > baseline_metrics['loss'] * (1 + self.alert_thresholds['loss_increase']):
            alerts.append({
                'type': 'loss_increase',
                'message': f"Training loss increased by {((current_metrics['loss'] / baseline_metrics['loss']) - 1) * 100:.1f}%",
                'severity': 'warning'
            })
        
        # Check GPU memory
        for gpu_id, memory in current_metrics['gpu_memory'].items():
            if memory['memory_allocated'] / memory['memory_reserved'] > self.alert_thresholds['gpu_memory']:
                alerts.append({
                    'type': 'gpu_memory',
                    'message': f"GPU {gpu_id} memory usage is {memory['memory_allocated'] / memory['memory_reserved'] * 100:.1f}%",
                    'severity': 'warning'
                })
        
        # Check CPU usage
        if current_metrics['cpu_percent'] > self.alert_thresholds['cpu_usage'] * 100:
            alerts.append({
                'type': 'cpu_usage',
                'message': f"CPU usage is {current_metrics['cpu_percent']:.1f}%",
                'severity': 'warning'
            })
        
        return alerts
    
    def send_alert(self, alert):
        """Send alert via email."""
        msg = MIMEText(f"""
        NeMo RL Training Alert
        
        Type: {alert['type']}
        Message: {alert['message']}
        Severity: {alert['severity']}
        Time: {datetime.now()}
        """)
        
        msg['Subject'] = f"NeMo RL Alert: {alert['type']}"
        msg['From'] = self.email_config['from']
        msg['To'] = self.email_config['to']
        
        with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            server.send_message(msg)
```

### Real-time Monitoring

Implement real-time monitoring:

```python
import threading
import time

class RealTimeMonitor:
    def __init__(self, alerting_system):
        self.alerting_system = alerting_system
        self.monitoring_thread = None
        self.is_monitoring = False
        self.baseline_metrics = None
    
    def start_monitoring(self, interval=30):
        """Start real-time monitoring."""
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
    
    def _monitor_loop(self, interval):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Get current metrics
                current_metrics = self._get_current_metrics()
                
                # Set baseline if not set
                if self.baseline_metrics is None:
                    self.baseline_metrics = current_metrics
                
                # Check for alerts
                alerts = self.alerting_system.check_alerts(current_metrics, self.baseline_metrics)
                
                # Send alerts
                for alert in alerts:
                    self.alerting_system.send_alert(alert)
                
                # Log metrics
                self._log_metrics(current_metrics)
                
                time.sleep(interval)
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(interval)
    
    def _get_current_metrics(self):
        """Get current system metrics."""
        return {
            'loss': self._get_current_loss(),
            'gpu_memory': self._get_gpu_memory(),
            'cpu_percent': self._get_cpu_usage(),
            'timestamp': time.time()
        }
    
    def _log_metrics(self, metrics):
        """Log metrics to storage."""
        # Implementation depends on storage backend
        pass
```

## Performance Analysis

### Trend Analysis

Analyze performance trends:

```python
import numpy as np
from scipy import stats

class PerformanceAnalyzer:
    def __init__(self, metrics_history):
        self.metrics_history = metrics_history
    
    def analyze_loss_trend(self):
        """Analyze loss trend over time."""
        losses = [m['loss'] for m in self.metrics_history]
        timestamps = [m['timestamp'] for m in self.metrics_history]
        
        # Calculate trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(timestamps, losses)
        
        return {
            'slope': slope,
            'r_squared': r_value**2,
            'p_value': p_value,
            'trend': 'decreasing' if slope < 0 else 'increasing',
            'significance': 'significant' if p_value < 0.05 else 'not_significant'
        }
    
    def analyze_resource_usage(self):
        """Analyze resource usage patterns."""
        cpu_usage = [m['cpu_percent'] for m in self.metrics_history]
        gpu_memory = [m['gpu_memory'] for m in self.metrics_history]
        
        return {
            'cpu_stats': {
                'mean': np.mean(cpu_usage),
                'std': np.std(cpu_usage),
                'max': np.max(cpu_usage),
                'min': np.min(cpu_usage)
            },
            'gpu_stats': {
                'mean': np.mean(gpu_memory),
                'std': np.std(gpu_memory),
                'max': np.max(gpu_memory),
                'min': np.min(gpu_memory)
            }
        }
    
    def detect_anomalies(self):
        """Detect performance anomalies."""
        losses = [m['loss'] for m in self.metrics_history]
        
        # Use z-score to detect anomalies
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)
        z_scores = [(loss - mean_loss) / std_loss for loss in losses]
        
        anomalies = []
        for i, z_score in enumerate(z_scores):
            if abs(z_score) > 2:  # 2 standard deviations
                anomalies.append({
                    'index': i,
                    'timestamp': self.metrics_history[i]['timestamp'],
                    'loss': losses[i],
                    'z_score': z_score
                })
        
        return anomalies
```

## Configuration Examples

### Basic Monitoring

```yaml
monitoring:
  enabled: true
  interval: 30  # seconds
  
  metrics:
    - training_loss
    - gpu_memory
    - cpu_usage
    - training_speed
  
  visualization:
    tensorboard: true
    dashboard: true
    prometheus: false
  
  alerting:
    enabled: true
    email_notifications: true
    thresholds:
      loss_increase: 0.1
      gpu_memory: 0.9
      cpu_usage: 0.95
```

### Advanced Monitoring

```yaml
monitoring:
  enabled: true
  interval: 10  # seconds
  
  metrics:
    - training_loss
    - gpu_memory
    - cpu_usage
    - training_speed
    - model_parameters
    - gradient_norms
    - learning_rate
  
  visualization:
    tensorboard: true
    dashboard: true
    prometheus: true
    grafana: true
  
  alerting:
    enabled: true
    email_notifications: true
    slack_notifications: true
    thresholds:
      loss_increase: 0.05
      gpu_memory: 0.85
      cpu_usage: 0.9
      training_speed: 0.7
  
  analysis:
    trend_analysis: true
    anomaly_detection: true
    performance_regression: true
```

## Best Practices

### Monitoring Guidelines

1. **Comprehensive Coverage**
   - Monitor all critical metrics
   - Include system and application metrics
   - Track both real-time and historical data

2. **Appropriate Intervals**
   - Use shorter intervals for critical metrics
   - Balance monitoring overhead with detail
   - Adjust based on training phase

3. **Alert Management**
   - Set appropriate thresholds
   - Avoid alert fatigue
   - Provide actionable alerts

### Performance Optimization

1. **Baseline Establishment**
   - Establish performance baselines
   - Document normal operating ranges
   - Set realistic performance targets

2. **Continuous Improvement**
   - Monitor performance trends
   - Identify optimization opportunities
   - Track improvement impact

3. **Proactive Monitoring**
   - Detect issues before they impact training
   - Monitor resource usage patterns
   - Plan for capacity needs

## Next Steps

After setting up monitoring:

1. **Establish Baselines**: Create performance baselines for your models
2. **Configure Alerts**: Set up appropriate alerting thresholds
3. **Analyze Trends**: Use monitoring data to optimize performance
4. **Scale Monitoring**: Extend monitoring to larger deployments

For more advanced topics, see:
- [Performance Profiling](profiling.md) - Detailed profiling techniques
- [Benchmarking](benchmarking.md) - Performance benchmarking tools
- [Memory Optimization](memory-optimization.md) - Memory monitoring and optimization 