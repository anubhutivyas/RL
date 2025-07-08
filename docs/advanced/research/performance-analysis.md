# Performance Analysis

This guide covers performance analysis techniques for NeMo RL research, including statistical analysis, visualization, and interpretation of results.

## Overview

Performance analysis is essential for understanding algorithm behavior, comparing methods, and drawing meaningful conclusions from experimental results. This guide covers comprehensive analysis techniques.

## Statistical Analysis

### Basic Statistics

Compute fundamental statistical measures:

```python
import numpy as np
import pandas as pd
from scipy import stats

class PerformanceAnalyzer:
    def __init__(self, results_data):
        self.data = results_data
        self.analyzed_metrics = {}
    
    def compute_basic_statistics(self, metric_name):
        """Compute basic statistics for a metric."""
        values = self.data[metric_name]
        
        stats_dict = {
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'q25': np.percentile(values, 25),
            'q75': np.percentile(values, 75),
            'count': len(values)
        }
        
        return stats_dict
    
    def analyze_all_metrics(self):
        """Analyze all metrics in the dataset."""
        for metric in self.data.columns:
            if metric != 'step' and metric != 'timestamp':
                self.analyzed_metrics[metric] = self.compute_basic_statistics(metric)
        
        return self.analyzed_metrics
```

### Trend Analysis

Analyze performance trends over time:

```python
def analyze_trends(self, metric_name, window_size=100):
    """Analyze trends in performance metrics."""
    values = self.data[metric_name]
    steps = self.data['step']
    
    # Moving average
    moving_avg = pd.Series(values).rolling(window=window_size).mean()
    
    # Linear trend
    slope, intercept, r_value, p_value, std_err = stats.linregress(steps, values)
    
    # Detect significant changes
    change_points = self.detect_change_points(values)
    
    trend_analysis = {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'p_value': p_value,
        'trend_direction': 'increasing' if slope > 0 else 'decreasing',
        'significance': 'significant' if p_value < 0.05 else 'not_significant',
        'moving_average': moving_avg.tolist(),
        'change_points': change_points
    }
    
    return trend_analysis

def detect_change_points(self, values, threshold=2.0):
    """Detect significant change points in time series."""
    change_points = []
    
    for i in range(1, len(values)):
        # Calculate z-score of change
        change = values[i] - values[i-1]
        mean_change = np.mean(np.diff(values))
        std_change = np.std(np.diff(values))
        
        if std_change > 0:
            z_score = abs(change - mean_change) / std_change
            
            if z_score > threshold:
                change_points.append({
                    'index': i,
                    'value': values[i],
                    'change': change,
                    'z_score': z_score
                })
    
    return change_points
```

### Comparative Analysis

Compare multiple algorithms or configurations:

```python
def compare_algorithms(self, algorithm_results):
    """Compare performance across multiple algorithms."""
    comparison_results = {}
    
    for algorithm_name, results in algorithm_results.items():
        # Compute statistics for each algorithm
        stats = {}
        for metric in ['loss', 'reward', 'accuracy']:
            if metric in results.columns:
                stats[metric] = self.compute_basic_statistics(metric)
        
        comparison_results[algorithm_name] = stats
    
    # Statistical significance tests
    significance_tests = self.perform_significance_tests(algorithm_results)
    comparison_results['significance_tests'] = significance_tests
    
    return comparison_results

def perform_significance_tests(self, algorithm_results):
    """Perform statistical significance tests."""
    tests = {}
    
    # Get all algorithm names
    algorithms = list(algorithm_results.keys())
    
    # Perform pairwise comparisons
    for i, alg1 in enumerate(algorithms):
        for j, alg2 in enumerate(algorithms[i+1:], i+1):
            comparison_name = f"{alg1}_vs_{alg2}"
            
            # Compare final performance
            final_perf1 = algorithm_results[alg1]['reward'].iloc[-100:].mean()
            final_perf2 = algorithm_results[alg2]['reward'].iloc[-100:].mean()
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(
                algorithm_results[alg1]['reward'].iloc[-100:],
                algorithm_results[alg2]['reward'].iloc[-100:]
            )
            
            tests[comparison_name] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'effect_size': (final_perf1 - final_perf2) / np.sqrt(
                    (algorithm_results[alg1]['reward'].var() + algorithm_results[alg2]['reward'].var()) / 2
                )
            }
    
    return tests
```

## Visualization

### Performance Plots

Create comprehensive performance visualizations:

```python
import matplotlib.pyplot as plt
import seaborn as sns

class PerformanceVisualizer:
    def __init__(self, results_data):
        self.data = results_data
        self.setup_plotting_style()
    
    def setup_plotting_style(self):
        """Setup consistent plotting style."""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_learning_curves(self, metrics=['loss', 'reward', 'accuracy'], 
                            algorithms=None, figsize=(15, 10)):
        """Plot learning curves for multiple metrics and algorithms."""
        fig, axes = plt.subplots(len(metrics), 1, figsize=figsize)
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            if algorithms:
                for alg_name, alg_data in algorithms.items():
                    if metric in alg_data.columns:
                        ax.plot(alg_data['step'], alg_data[metric], 
                               label=alg_name, linewidth=2)
            else:
                if metric in self.data.columns:
                    ax.plot(self.data['step'], self.data[metric], linewidth=2)
            
            ax.set_xlabel('Training Steps')
            ax.set_ylabel(metric.title())
            ax.set_title(f'{metric.title()} Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_performance_distribution(self, metric='reward', algorithms=None):
        """Plot distribution of performance metrics."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if algorithms:
            data_to_plot = []
            labels = []
            
            for alg_name, alg_data in algorithms.items():
                if metric in alg_data.columns:
                    # Use final performance values
                    final_values = alg_data[metric].iloc[-100:]
                    data_to_plot.append(final_values)
                    labels.append(alg_name)
            
            ax.boxplot(data_to_plot, labels=labels)
        else:
            if metric in self.data.columns:
                final_values = self.data[metric].iloc[-100:]
                ax.hist(final_values, bins=30, alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Algorithm' if algorithms else metric.title())
        ax.set_ylabel('Frequency' if not algorithms else metric.title())
        ax.set_title(f'Distribution of {metric.title()}')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_correlation_matrix(self, metrics=None):
        """Plot correlation matrix between metrics."""
        if metrics is None:
            metrics = ['loss', 'reward', 'accuracy', 'learning_rate']
        
        # Select relevant columns
        relevant_data = self.data[metrics].dropna()
        
        # Compute correlation matrix
        corr_matrix = relevant_data.corr()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax)
        ax.set_title('Metric Correlation Matrix')
        
        return fig
    
    def plot_rolling_statistics(self, metric='reward', window=100):
        """Plot rolling statistics for a metric."""
        values = self.data[metric]
        rolling_mean = values.rolling(window=window).mean()
        rolling_std = values.rolling(window=window).std()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot original data and rolling mean
        ax1.plot(self.data['step'], values, alpha=0.5, label='Original')
        ax1.plot(self.data['step'], rolling_mean, linewidth=2, label=f'Rolling Mean (w={window})')
        ax1.fill_between(self.data['step'], 
                        rolling_mean - rolling_std, 
                        rolling_mean + rolling_std, 
                        alpha=0.3, label='±1 Std Dev')
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel(metric.title())
        ax1.set_title(f'{metric.title()} with Rolling Statistics')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot rolling standard deviation
        ax2.plot(self.data['step'], rolling_std, color='red')
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Standard Deviation')
        ax2.set_title(f'Rolling Standard Deviation of {metric.title()}')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
```

### Advanced Visualizations

Create advanced analysis plots:

```python
def plot_convergence_analysis(self, metric='reward', window=100):
    """Analyze convergence behavior."""
    values = self.data[metric]
    steps = self.data['step']
    
    # Compute convergence metrics
    rolling_mean = values.rolling(window=window).mean()
    rolling_std = values.rolling(window=window).std()
    
    # Detect convergence
    convergence_threshold = 0.01
    converged = rolling_std < convergence_threshold
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot performance with convergence regions
    ax1.plot(steps, values, alpha=0.5, label='Original')
    ax1.plot(steps, rolling_mean, linewidth=2, label='Rolling Mean')
    
    # Highlight converged regions
    converged_steps = steps[converged]
    converged_values = values[converged]
    ax1.scatter(converged_steps, converged_values, 
                color='red', alpha=0.7, label='Converged', s=20)
    
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel(metric.title())
    ax1.set_title(f'{metric.title()} Convergence Analysis')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot convergence indicator
    ax2.plot(steps, rolling_std, color='blue', label='Rolling Std')
    ax2.axhline(y=convergence_threshold, color='red', linestyle='--', 
                label=f'Convergence Threshold ({convergence_threshold})')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Standard Deviation')
    ax2.set_title('Convergence Indicator')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_algorithm_comparison(self, algorithm_results, metric='reward'):
    """Create comprehensive algorithm comparison plot."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Learning curves
    for alg_name, alg_data in algorithm_results.items():
        if metric in alg_data.columns:
            ax1.plot(alg_data['step'], alg_data[metric], label=alg_name, linewidth=2)
    
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel(metric.title())
    ax1.set_title('Learning Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Final performance distribution
    final_performances = []
    labels = []
    
    for alg_name, alg_data in algorithm_results.items():
        if metric in alg_data.columns:
            final_values = alg_data[metric].iloc[-100:]
            final_performances.append(final_values)
            labels.append(alg_name)
    
    ax2.boxplot(final_performances, labels=labels)
    ax2.set_ylabel(metric.title())
    ax2.set_title('Final Performance Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Performance over time (smoothed)
    for alg_name, alg_data in algorithm_results.items():
        if metric in alg_data.columns:
            smoothed = alg_data[metric].rolling(window=50).mean()
            ax3.plot(alg_data['step'], smoothed, label=alg_name, linewidth=2)
    
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel(f'{metric.title()} (Smoothed)')
    ax3.set_title('Smoothed Learning Curves')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Performance improvement
    for alg_name, alg_data in algorithm_results.items():
        if metric in alg_data.columns:
            improvement = alg_data[metric] - alg_data[metric].iloc[0]
            ax4.plot(alg_data['step'], improvement, label=alg_name, linewidth=2)
    
    ax4.set_xlabel('Training Steps')
    ax4.set_ylabel(f'{metric.title()} Improvement')
    ax4.set_title('Performance Improvement Over Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
```

## Performance Metrics

### Training Metrics

Define comprehensive training metrics:

```python
class TrainingMetrics:
    def __init__(self):
        self.metrics = {}
    
    def compute_training_metrics(self, results_data):
        """Compute comprehensive training metrics."""
        metrics = {}
        
        # Convergence metrics
        metrics.update(self.compute_convergence_metrics(results_data))
        
        # Efficiency metrics
        metrics.update(self.compute_efficiency_metrics(results_data))
        
        # Stability metrics
        metrics.update(self.compute_stability_metrics(results_data))
        
        # Final performance metrics
        metrics.update(self.compute_final_performance_metrics(results_data))
        
        return metrics
    
    def compute_convergence_metrics(self, data):
        """Compute convergence-related metrics."""
        metrics = {}
        
        for metric in ['loss', 'reward', 'accuracy']:
            if metric in data.columns:
                values = data[metric]
                
                # Convergence time
                convergence_threshold = 0.01
                rolling_std = values.rolling(window=100).std()
                converged_steps = rolling_std[rolling_std < convergence_threshold]
                
                if len(converged_steps) > 0:
                    convergence_step = converged_steps.index[0]
                    metrics[f'{metric}_convergence_step'] = convergence_step
                else:
                    metrics[f'{metric}_convergence_step'] = None
                
                # Final performance
                final_performance = values.iloc[-100:].mean()
                metrics[f'{metric}_final_performance'] = final_performance
                
                # Performance improvement
                initial_performance = values.iloc[:100].mean()
                improvement = final_performance - initial_performance
                metrics[f'{metric}_improvement'] = improvement
        
        return metrics
    
    def compute_efficiency_metrics(self, data):
        """Compute efficiency-related metrics."""
        metrics = {}
        
        # Sample efficiency
        if 'reward' in data.columns:
            rewards = data['reward']
            
            # Steps to reach 50% of final performance
            final_performance = rewards.iloc[-100:].mean()
            target_performance = final_performance * 0.5
            
            steps_to_target = rewards[rewards >= target_performance]
            if len(steps_to_target) > 0:
                metrics['steps_to_50_percent'] = steps_to_target.index[0]
            else:
                metrics['steps_to_50_percent'] = None
        
        return metrics
    
    def compute_stability_metrics(self, data):
        """Compute stability-related metrics."""
        metrics = {}
        
        for metric in ['loss', 'reward', 'accuracy']:
            if metric in data.columns:
                values = data[metric]
                
                # Variance in final performance
                final_variance = values.iloc[-100:].var()
                metrics[f'{metric}_final_variance'] = final_variance
                
                # Coefficient of variation
                final_mean = values.iloc[-100:].mean()
                if final_mean != 0:
                    cv = np.sqrt(final_variance) / abs(final_mean)
                    metrics[f'{metric}_coefficient_of_variation'] = cv
                else:
                    metrics[f'{metric}_coefficient_of_variation'] = float('inf')
        
        return metrics
    
    def compute_final_performance_metrics(self, data):
        """Compute final performance metrics."""
        metrics = {}
        
        for metric in ['loss', 'reward', 'accuracy']:
            if metric in data.columns:
                values = data[metric]
                final_values = values.iloc[-100:]
                
                metrics[f'{metric}_final_mean'] = final_values.mean()
                metrics[f'{metric}_final_std'] = final_values.std()
                metrics[f'{metric}_final_min'] = final_values.min()
                metrics[f'{metric}_final_max'] = final_values.max()
                metrics[f'{metric}_final_median'] = final_values.median()
        
        return metrics
```

### Comparative Metrics

Define metrics for comparing algorithms:

```python
class ComparativeMetrics:
    def __init__(self):
        self.metrics = {}
    
    def compare_algorithms(self, algorithm_results, baseline_algorithm=None):
        """Compare multiple algorithms."""
        comparison = {}
        
        # Set baseline
        if baseline_algorithm is None:
            baseline_algorithm = list(algorithm_results.keys())[0]
        
        baseline_data = algorithm_results[baseline_algorithm]
        
        for alg_name, alg_data in algorithm_results.items():
            if alg_name == baseline_algorithm:
                continue
            
            comparison[alg_name] = self.compute_relative_metrics(
                alg_data, baseline_data
            )
        
        return comparison
    
    def compute_relative_metrics(self, algorithm_data, baseline_data):
        """Compute metrics relative to baseline."""
        metrics = {}
        
        for metric in ['loss', 'reward', 'accuracy']:
            if metric in algorithm_data.columns and metric in baseline_data.columns:
                alg_final = algorithm_data[metric].iloc[-100:].mean()
                baseline_final = baseline_data[metric].iloc[-100:].mean()
                
                # Relative performance
                if baseline_final != 0:
                    relative_performance = (alg_final - baseline_final) / abs(baseline_final)
                    metrics[f'{metric}_relative_performance'] = relative_performance
                else:
                    metrics[f'{metric}_relative_performance'] = float('inf')
                
                # Performance ratio
                if baseline_final != 0:
                    performance_ratio = alg_final / baseline_final
                    metrics[f'{metric}_performance_ratio'] = performance_ratio
                else:
                    metrics[f'{metric}_performance_ratio'] = float('inf')
        
        return metrics
```

## Reporting

### Automated Reports

Generate comprehensive performance reports:

```python
class PerformanceReporter:
    def __init__(self, results_data, algorithm_name):
        self.data = results_data
        self.algorithm_name = algorithm_name
        self.analyzer = PerformanceAnalyzer(results_data)
        self.visualizer = PerformanceVisualizer(results_data)
        self.metrics_calculator = TrainingMetrics()
    
    def generate_report(self, output_path=None):
        """Generate comprehensive performance report."""
        report = {
            'algorithm_name': self.algorithm_name,
            'summary_statistics': self.analyzer.analyze_all_metrics(),
            'training_metrics': self.metrics_calculator.compute_training_metrics(self.data),
            'figures': self.generate_figures()
        }
        
        if output_path:
            self.save_report(report, output_path)
        
        return report
    
    def generate_figures(self):
        """Generate all analysis figures."""
        figures = {}
        
        # Learning curves
        figures['learning_curves'] = self.visualizer.plot_learning_curves()
        
        # Performance distributions
        figures['performance_distribution'] = self.visualizer.plot_performance_distribution()
        
        # Rolling statistics
        figures['rolling_statistics'] = self.visualizer.plot_rolling_statistics()
        
        # Correlation matrix
        figures['correlation_matrix'] = self.visualizer.plot_correlation_matrix()
        
        return figures
    
    def save_report(self, report, output_path):
        """Save report to file."""
        import json
        
        # Save text report
        text_report = {
            'algorithm_name': report['algorithm_name'],
            'summary_statistics': report['summary_statistics'],
            'training_metrics': report['training_metrics']
        }
        
        with open(f"{output_path}_report.json", 'w') as f:
            json.dump(text_report, f, indent=2, default=str)
        
        # Save figures
        for name, fig in report['figures'].items():
            fig.savefig(f"{output_path}_{name}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
```

### Comparative Reports

Generate comparative analysis reports:

```python
class ComparativeReporter:
    def __init__(self, algorithm_results):
        self.algorithm_results = algorithm_results
        self.comparative_metrics = ComparativeMetrics()
    
    def generate_comparative_report(self, output_path=None):
        """Generate comparative analysis report."""
        report = {
            'algorithm_comparison': self.comparative_metrics.compare_algorithms(
                self.algorithm_results
            ),
            'statistical_tests': self.perform_statistical_tests(),
            'figures': self.generate_comparative_figures()
        }
        
        if output_path:
            self.save_comparative_report(report, output_path)
        
        return report
    
    def perform_statistical_tests(self):
        """Perform statistical significance tests."""
        tests = {}
        
        algorithms = list(self.algorithm_results.keys())
        
        for i, alg1 in enumerate(algorithms):
            for j, alg2 in enumerate(algorithms[i+1:], i+1):
                comparison_name = f"{alg1}_vs_{alg2}"
                
                # Compare final performance
                final_perf1 = self.algorithm_results[alg1]['reward'].iloc[-100:]
                final_perf2 = self.algorithm_results[alg2]['reward'].iloc[-100:]
                
                # T-test
                t_stat, p_value = stats.ttest_ind(final_perf1, final_perf2)
                
                # Mann-Whitney U test
                u_stat, u_p_value = stats.mannwhitneyu(final_perf1, final_perf2)
                
                tests[comparison_name] = {
                    't_test': {
                        'statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    },
                    'mann_whitney': {
                        'statistic': u_stat,
                        'p_value': u_p_value,
                        'significant': u_p_value < 0.05
                    }
                }
        
        return tests
    
    def generate_comparative_figures(self):
        """Generate comparative analysis figures."""
        figures = {}
        
        # Algorithm comparison plot
        figures['algorithm_comparison'] = self.visualizer.plot_algorithm_comparison(
            self.algorithm_results
        )
        
        return figures
```

## Best Practices

### Analysis Guidelines

1. **Comprehensive Metrics**
   - Use multiple performance indicators
   - Consider both final performance and learning speed
   - Account for stability and robustness

2. **Statistical Rigor**
   - Perform appropriate statistical tests
   - Report confidence intervals
   - Consider effect sizes

3. **Visualization Quality**
   - Use clear, informative plots
   - Include error bars and confidence intervals
   - Maintain consistent styling

### Reporting Standards

1. **Transparency**
   - Report all experimental conditions
   - Include uncertainty measures
   - Document analysis methods

2. **Reproducibility**
   - Provide code and data
   - Document random seeds
   - Include environment details

3. **Interpretation**
   - Provide clear conclusions
   - Discuss limitations
   - Suggest future work

## Next Steps

After implementing performance analysis:

1. **Validate Methods**: Ensure analysis methods are appropriate
2. **Automate Reports**: Create automated reporting pipelines
3. **Extend Analysis**: Add domain-specific metrics
4. **Share Results**: Publish findings and contribute to community

For more advanced topics, see:
- [Custom Algorithms](custom-algorithms.md) - Implementing custom algorithms
- [Research Methodologies](index.md) - Research best practices
- [Hyperparameter Optimization](hyperparameter-optimization.md) - Optimizing algorithm parameters 