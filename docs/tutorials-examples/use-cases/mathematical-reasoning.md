# Mathematical Reasoning

Train NeMo RL models to solve complex mathematical problems with step-by-step reasoning, proof generation, and advanced mathematical concepts. This use case covers architectural patterns for building mathematical reasoning systems that can handle algebra, calculus, geometry, and formal proofs.

## Overview

Mathematical reasoning with RL involves training models to understand mathematical concepts, perform step-by-step problem solving, and generate formal proofs. Unlike traditional approaches that focus on answer prediction, RL enables models to learn from feedback about reasoning quality, correctness, and solution elegance.

## Key Challenges

### Problem Complexity
- **Multi-step reasoning**: Breaking down complex problems into manageable steps
- **Proof generation**: Creating formal mathematical proofs with logical flow
- **Concept integration**: Combining multiple mathematical concepts in single problems
- **Symbolic manipulation**: Handling mathematical notation and symbolic expressions

### Evaluation Metrics
- **Correctness**: Mathematical accuracy of solutions and proofs
- **Reasoning quality**: Logical flow and step-by-step clarity
- **Completeness**: Full problem coverage without missing steps
- **Elegance**: Efficiency and elegance of solution approaches

### Domain Coverage
- **Algebra**: Linear equations, polynomials, systems of equations
- **Calculus**: Derivatives, integrals, limits, series
- **Geometry**: Geometric proofs, spatial reasoning, coordinate systems
- **Number theory**: Prime factorization, modular arithmetic, divisibility

## Architecture Patterns

### Multi-Stage Reasoning Pipeline

```python
class MathematicalReasoningPipeline:
    def __init__(self):
        self.problem_analyzer = ProblemAnalyzer()
        self.reasoning_engine = ReasoningEngine()
        self.proof_generator = ProofGenerator()
        self.verifier = MathematicalVerifier()
    
    def solve_problem(self, problem_text):
        """Solve mathematical problem with step-by-step reasoning"""
        # Stage 1: Problem analysis
        problem_type = self.problem_analyzer.classify(problem_text)
        key_concepts = self.problem_analyzer.extract_concepts(problem_text)
        
        # Stage 2: Solution planning
        solution_plan = self.reasoning_engine.plan_solution(
            problem_type, key_concepts
        )
        
        # Stage 3: Step-by-step execution
        solution_steps = []
        for step in solution_plan:
            step_result = self.reasoning_engine.execute_step(step)
            solution_steps.append(step_result)
        
        # Stage 4: Proof generation
        proof = self.proof_generator.generate_proof(solution_steps)
        
        # Stage 5: Verification
        verification_result = self.verifier.verify_solution(
            problem_text, solution_steps, proof
        )
        
        return {
            'solution_steps': solution_steps,
            'proof': proof,
            'verification': verification_result
        }
```

### Environment Design for Mathematical Problems

```python
class MathematicalEnvironment:
    def __init__(self, problem_domain="algebra"):
        self.domain = problem_domain
        self.symbolic_solver = SymbolicSolver()
        self.proof_checker = ProofChecker()
        self.step_validator = StepValidator()
    
    def step(self, action):
        """Process a reasoning step and return reward"""
        try:
            # Parse the mathematical action
            parsed_action = self.parse_mathematical_action(action)
            
            # Validate the step
            step_validity = self.step_validator.validate(parsed_action)
            
            # Check mathematical correctness
            mathematical_correctness = self.symbolic_solver.verify(parsed_action)
            
            # Assess reasoning quality
            reasoning_quality = self.assess_reasoning_quality(parsed_action)
            
            # Calculate reward
            reward = self.calculate_reward(
                step_validity=step_validity,
                mathematical_correctness=mathematical_correctness,
                reasoning_quality=reasoning_quality
            )
            
            return reward, {
                'step_valid': step_validity,
                'mathematically_correct': mathematical_correctness,
                'reasoning_quality': reasoning_quality
            }
        except Exception as e:
            return -1.0, {"error": str(e)}
    
    def calculate_reward(self, step_validity, mathematical_correctness, reasoning_quality):
        reward = 0.0
        
        # Step validity reward
        if step_validity:
            reward += 0.3
        else:
            reward -= 0.5
        
        # Mathematical correctness reward
        if mathematical_correctness:
            reward += 0.4
        else:
            reward -= 0.8
        
        # Reasoning quality reward
        reward += reasoning_quality * 0.3
        
        return reward
```

## Implementation Considerations

### Problem Representation

```python
class MathematicalProblem:
    def __init__(self, problem_text, domain="algebra"):
        self.text = problem_text
        self.domain = domain
        self.concepts = self.extract_concepts()
        self.constraints = self.extract_constraints()
        self.expected_output = self.determine_expected_output()
    
    def extract_concepts(self):
        """Extract mathematical concepts from problem text"""
        concepts = []
        
        # Identify mathematical concepts using NLP
        concept_patterns = {
            'equation': r'[=<>≤≥]',
            'function': r'f\(x\)|g\(x\)',
            'derivative': r'd/dx|f\'\(x\)',
            'integral': r'∫|\\int',
            'limit': r'lim|\\lim',
            'summation': r'∑|\\sum',
            'product': r'∏|\\prod'
        }
        
        for concept, pattern in concept_patterns.items():
            if re.search(pattern, self.text):
                concepts.append(concept)
        
        return concepts
    
    def extract_constraints(self):
        """Extract problem constraints and conditions"""
        constraints = {
            'domain': None,
            'range': None,
            'conditions': [],
            'assumptions': []
        }
        
        # Parse constraints from problem text
        # Implementation details...
        
        return constraints
```

### Reward Function for Mathematical Reasoning

```python
class MathematicalReward:
    def __init__(self):
        self.metrics = {
            'correctness': MathematicalCorrectnessMetric(),
            'reasoning': ReasoningQualityMetric(),
            'completeness': CompletenessMetric(),
            'elegance': EleganceMetric()
        }
    
    def calculate_reward(self, solution_steps, problem, expected_solution):
        total_reward = 0.0
        weights = {
            'correctness': 0.4,
            'reasoning': 0.3,
            'completeness': 0.2,
            'elegance': 0.1
        }
        
        for metric_name, metric in self.metrics.items():
            score = metric.evaluate(solution_steps, problem, expected_solution)
            total_reward += score * weights[metric_name]
        
        return total_reward
```

## Advanced Mathematical Concepts

### Proof Generation

```python
class ProofGenerator:
    def __init__(self):
        self.logic_engine = LogicEngine()
        self.theorem_database = TheoremDatabase()
        self.proof_strategies = ProofStrategies()
    
    def generate_proof(self, problem, solution_steps):
        """Generate formal mathematical proof"""
        proof_structure = {
            'hypothesis': self.extract_hypothesis(problem),
            'conclusion': self.extract_conclusion(problem),
            'steps': [],
            'theorems_used': [],
            'logical_flow': []
        }
        
        for step in solution_steps:
            # Identify applicable theorems
            applicable_theorems = self.theorem_database.find_applicable(step)
            
            # Generate proof step
            proof_step = self.logic_engine.generate_proof_step(
                step, applicable_theorems
            )
            
            proof_structure['steps'].append(proof_step)
            proof_structure['theorems_used'].extend(applicable_theorems)
        
        return proof_structure
```

### Symbolic Computation Integration

```python
class SymbolicComputation:
    def __init__(self):
        self.sympy_engine = sympy.SymbolicEngine()
        self.mathematica_engine = MathematicaEngine()
        self.wolfram_engine = WolframEngine()
    
    def verify_solution(self, problem, solution):
        """Verify mathematical solution using symbolic computation"""
        try:
            # Convert to symbolic form
            symbolic_problem = self.sympy_engine.parse(problem)
            symbolic_solution = self.sympy_engine.parse(solution)
            
            # Verify equality
            verification_result = self.sympy_engine.verify_equality(
                symbolic_problem, symbolic_solution
            )
            
            return verification_result
        except Exception as e:
            return {'error': str(e), 'verified': False}
```

## Production Deployment

### Mathematical Reasoning Service

```python
class MathematicalReasoningService:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)
        self.symbolic_verifier = SymbolicVerifier()
        self.proof_checker = ProofChecker()
    
    def solve_problem(self, problem_text, include_proof=True):
        """Solve mathematical problem with reasoning"""
        try:
            # Generate solution with reasoning
            solution = self.model.generate_solution(problem_text)
            
            # Verify mathematical correctness
            verification = self.symbolic_verifier.verify(solution)
            
            # Generate proof if requested
            proof = None
            if include_proof:
                proof = self.proof_checker.generate_proof(solution)
            
            return {
                'solution': solution,
                'verification': verification,
                'proof': proof,
                'reasoning_steps': solution.get('steps', [])
            }
        except Exception as e:
            return {'error': str(e)}
```

### Multi-Domain Support

```python
class MultiDomainMathematicalService:
    def __init__(self):
        self.domain_models = {
            'algebra': self.load_model('algebra_model'),
            'calculus': self.load_model('calculus_model'),
            'geometry': self.load_model('geometry_model'),
            'number_theory': self.load_model('number_theory_model')
        }
        self.domain_classifier = DomainClassifier()
    
    def solve_problem(self, problem_text):
        """Route problem to appropriate domain model"""
        # Classify problem domain
        domain = self.domain_classifier.classify(problem_text)
        
        # Select appropriate model
        model = self.domain_models.get(domain)
        if not model:
            return {'error': f'Unsupported domain: {domain}'}
        
        # Solve using domain-specific model
        return model.solve_problem(problem_text)
```

## Real-World Examples

### Algebra Problem Solving

```python
# Example: Solving quadratic equations with step-by-step reasoning
problem = "Solve the quadratic equation: x² + 5x + 6 = 0"

config = {
    'task': 'quadratic_equation_solving',
    'domain': 'algebra',
    'reasoning_style': 'step_by_step',
    'include_proof': True,
    'verification_method': 'symbolic_computation'
}

# Training configuration
training_config = {
    'reward_function': 'mathematical_reasoning_reward',
    'evaluation_metrics': ['correctness', 'reasoning_quality', 'completeness'],
    'curriculum_learning': True,
    'difficulty_progression': ['simple', 'medium', 'complex']
}

# Expected solution structure
expected_solution = {
    'steps': [
        'Identify coefficients: a=1, b=5, c=6',
        'Apply quadratic formula: x = (-b ± √(b²-4ac))/(2a)',
        'Calculate discriminant: b²-4ac = 25-24 = 1',
        'Substitute values: x = (-5 ± √1)/2',
        'Simplify: x = (-5 ± 1)/2',
        'Final answer: x = -3 or x = -2'
    ],
    'proof': 'Verification by substitution...',
    'verification': True
}
```

### Calculus Problem Solving

```python
# Example: Computing derivatives with chain rule
problem = "Find the derivative of f(x) = sin(x² + 3x)"

config = {
    'task': 'derivative_computation',
    'domain': 'calculus',
    'techniques': ['chain_rule', 'power_rule', 'trigonometric_derivatives'],
    'step_by_step': True
}

# Training with curriculum learning
curriculum = [
    {'stage': 1, 'problems': 'basic_derivatives', 'difficulty': 'simple'},
    {'stage': 2, 'problems': 'chain_rule', 'difficulty': 'medium'},
    {'stage': 3, 'problems': 'complex_derivatives', 'difficulty': 'complex'}
]
```

## Best Practices

### Problem Design
- **Progressive difficulty**: Start with simple problems and gradually increase complexity
- **Concept integration**: Combine multiple mathematical concepts in single problems
- **Real-world context**: Include problems with practical applications
- **Multiple solution paths**: Design problems that can be solved using different approaches

### Model Training
- **Curriculum learning**: Progress from simple to complex mathematical concepts
- **Multi-task learning**: Train on multiple mathematical domains simultaneously
- **Reasoning quality**: Focus on step-by-step reasoning rather than just answer prediction
- **Proof generation**: Include formal proof generation in training objectives

### Evaluation
- **Symbolic verification**: Use computer algebra systems to verify solutions
- **Human evaluation**: Expert mathematicians evaluate reasoning quality
- **Completeness checking**: Ensure all problem aspects are addressed
- **Elegance assessment**: Evaluate solution efficiency and elegance

### Production Considerations
- **Domain specialization**: Deploy specialized models for different mathematical domains
- **Verification pipeline**: Implement automated verification for all solutions
- **Explanation generation**: Provide clear explanations for each reasoning step
- **Error handling**: Graceful handling of unsolvable or ambiguous problems

## Monitoring and Observability

```python
class MathematicalReasoningMonitor:
    def __init__(self):
        self.metrics = {
            'solution_accuracy': 0,
            'reasoning_quality': 0,
            'proof_correctness': 0,
            'response_time_p95': 0,
            'domain_distribution': {},
            'error_rate': 0
        }
    
    def log_solution(self, problem, solution, verification_result, latency):
        """Log mathematical reasoning metrics"""
        # Update accuracy metrics
        self.metrics['solution_accuracy'] += 1 if verification_result['correct'] else 0
        self.metrics['reasoning_quality'] += solution.get('reasoning_quality', 0)
        self.metrics['proof_correctness'] += 1 if solution.get('proof_correct') else 0
        
        # Update performance metrics
        self.metrics['response_time_p95'] = max(
            self.metrics['response_time_p95'], latency
        )
        
        # Update domain distribution
        domain = self.classify_domain(problem)
        self.metrics['domain_distribution'][domain] = \
            self.metrics['domain_distribution'].get(domain, 0) + 1
        
        # Send to monitoring system
        self.send_metrics(self.metrics)
```

This use case provides a comprehensive framework for building production-ready mathematical reasoning systems with NeMo RL, covering everything from problem representation to proof generation and deployment. 