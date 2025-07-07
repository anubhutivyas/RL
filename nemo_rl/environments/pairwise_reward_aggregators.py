# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
import numpy as np


class PairwiseRewardAggregator(ABC):
    """Abstract base class for aggregating pairwise comparison results into scalar rewards."""
    
    def __init__(self, **kwargs):
        """Default initialization that accepts any keyword arguments for compatibility with factory function."""
        # Accept any kwargs to maintain compatibility with the factory function
        # Subclasses can override this if they need specific configuration parameters
        pass
    
    @abstractmethod
    def aggregate_scores(
        self,
        comparison_results: List[Tuple[str, float, float, float]],  # (request_id, score_1, score_2, ranking_score)
        comparison_metadata: List[Tuple[str, int, int]],  # (prompt_key, resp_i, resp_j)
        prompt_groups: Dict[str, Dict[str, Any]]  # {prompt_key: {"conversations": [...], "metadata": {...}, "indices": [...]}}
    ) -> Dict[str, List[float]]:
        """Aggregate pairwise comparison results into final rewards for each response."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the aggregation method."""
        pass


class WinRateAggregator(PairwiseRewardAggregator):
    """Current approach: Binary win-loss converted to win rates."""
    
    def aggregate_scores(
        self,
        comparison_results: List[Tuple[str, float, float, float]],
        comparison_metadata: List[Tuple[str, int, int]],
        prompt_groups: Dict[str, Dict[str, Any]]
    ) -> Dict[str, List[float]]:
        """Aggregate using binary win-loss counting."""
        
        # Initialize win counts for each response group
        win_counts = {}
        total_comparisons = {}
        
        for prompt_key, group_data in prompt_groups.items():
            num_responses = len(group_data["conversations"])
            win_counts[prompt_key] = [0 for _ in range(num_responses)]
            total_comparisons[prompt_key] = [0 for _ in range(num_responses)]
        
        # Process each comparison result
        for (request_id, score_1, score_2, ranking_score), (prompt_key, resp_i, resp_j) in zip(
            comparison_results, comparison_metadata
        ):
            # Convert ranking score to binary win/loss
            # Ranking scores: 1-3 means response_1 is better, 4-6 means response_2 is better
            if ranking_score <= 3.0:
                win_counts[prompt_key][resp_i] += 1
            else:
                win_counts[prompt_key][resp_j] += 1
                
            # Update total comparisons for both responses
            total_comparisons[prompt_key][resp_i] += 1
            total_comparisons[prompt_key][resp_j] += 1
        
        # Calculate final scores as win rates
        final_scores = {}
        for prompt_key, group_data in prompt_groups.items():
            num_responses = len(group_data["conversations"])
            final_scores[prompt_key] = []
            for resp_idx in range(num_responses):
                if total_comparisons[prompt_key][resp_idx] > 0:
                    win_rate = win_counts[prompt_key][resp_idx] / total_comparisons[prompt_key][resp_idx]
                else:
                    win_rate = 0.5  # Neutral score if no comparisons
                final_scores[prompt_key].append(win_rate)
            
        return final_scores
    
    @property
    def name(self) -> str:
        return "win_rate"


class IndividualScoreAggregator(PairwiseRewardAggregator):
    """Use GenRM's individual helpfulness scores directly."""
    
    def __init__(self, score_range: Tuple[float, float] = (1.0, 5.0)):
        """
        Args:
            score_range: (min_score, max_score) for normalization
        """
        self.min_score, self.max_score = score_range
    
    def aggregate_scores(
        self,
        comparison_results: List[Tuple[str, float, float, float]],
        comparison_metadata: List[Tuple[str, int, int]],
        prompt_groups: Dict[str, Dict[str, Any]]
    ) -> Dict[str, List[float]]:
        """Aggregate using individual helpfulness scores."""
        
        # Collect individual scores for each response
        individual_scores = {}
        score_counts = {}
        
        for prompt_key, group_data in prompt_groups.items():
            num_responses = len(group_data["conversations"])
            individual_scores[prompt_key] = [0.0 for _ in range(num_responses)]
            score_counts[prompt_key] = [0 for _ in range(num_responses)]
        
        # Process each comparison result
        for (request_id, score_1, score_2, ranking_score), (prompt_key, resp_i, resp_j) in zip(
            comparison_results, comparison_metadata
        ):
            # Accumulate individual scores
            individual_scores[prompt_key][resp_i] += score_1
            individual_scores[prompt_key][resp_j] += score_2
            score_counts[prompt_key][resp_i] += 1
            score_counts[prompt_key][resp_j] += 1
        
        # Calculate average individual scores and normalize
        final_scores = {}
        for prompt_key, group_data in prompt_groups.items():
            num_responses = len(group_data["conversations"])
            final_scores[prompt_key] = []
            for resp_idx in range(num_responses):
                if score_counts[prompt_key][resp_idx] > 0:
                    avg_score = individual_scores[prompt_key][resp_idx] / score_counts[prompt_key][resp_idx]
                else:
                    avg_score = (self.min_score + self.max_score) / 2  # Neutral score if no comparisons
                final_scores[prompt_key].append(avg_score)
            
        return final_scores
    
    @property
    def name(self) -> str:
        return "individual_scores"


class EloRatingAggregator(PairwiseRewardAggregator):
    """Use Elo rating system to aggregate pairwise comparisons."""
    
    def __init__(self, k_factor: float = 32.0, initial_rating: float = 1500.0):
        """
        Args:
            k_factor: Elo K-factor (higher = more volatile updates)
            initial_rating: Starting Elo rating for all responses
        """
        self.k_factor = k_factor
        self.initial_rating = initial_rating
    
    def aggregate_scores(
        self,
        comparison_results: List[Tuple[str, float, float, float]],
        comparison_metadata: List[Tuple[str, int, int]],
        prompt_groups: Dict[str, Dict[str, Any]]
    ) -> Dict[str, List[float]]:
        """Aggregate using Elo rating system."""
        
        # Initialize Elo ratings
        elo_ratings = {}
        for prompt_key, group_data in prompt_groups.items():
            num_responses = len(group_data["conversations"])
            elo_ratings[prompt_key] = [self.initial_rating for _ in range(num_responses)]
        
        # Process each comparison result
        for (request_id, score_1, score_2, ranking_score), (prompt_key, resp_i, resp_j) in zip(
            comparison_results, comparison_metadata
        ):
            # Determine winner and loser
            if ranking_score <= 3.0:
                winner_idx, loser_idx = resp_i, resp_j
            else:
                winner_idx, loser_idx = resp_j, resp_i
            
            # Update Elo ratings
            self._update_elo_ratings(elo_ratings[prompt_key], winner_idx, loser_idx)
        
        # Convert Elo ratings to normalized rewards
        final_scores = {}
        for prompt_key, ratings in elo_ratings.items():
            # Normalize ratings to [0, 1] using min-max scaling within the group
            min_rating = min(ratings)
            max_rating = max(ratings)
            
            if max_rating > min_rating:
                normalized_ratings = [(r - min_rating) / (max_rating - min_rating) for r in ratings]
            else:
                # All ratings are the same, assign neutral scores
                normalized_ratings = [0.5 for _ in ratings]
            
            final_scores[prompt_key] = normalized_ratings
        
        return final_scores
    
    def _update_elo_ratings(self, ratings: List[float], winner_idx: int, loser_idx: int):
        """Update Elo ratings based on a single comparison."""
        winner_rating = ratings[winner_idx]
        loser_rating = ratings[loser_idx]
        
        # Calculate expected scores
        expected_winner = 1 / (1 + 10**((loser_rating - winner_rating) / 400))
        expected_loser = 1 - expected_winner
        
        # Update ratings
        ratings[winner_idx] += self.k_factor * (1 - expected_winner)
        ratings[loser_idx] += self.k_factor * (0 - expected_loser)
    
    @property
    def name(self) -> str:
        return "elo_rating"


class WeightedWinLossAggregator(PairwiseRewardAggregator):
    """Use weighted scores based on the magnitude of GenRM ranking preferences."""
    
    def __init__(self, score_mapping: Dict[int, float] = None):
        """
        Args:
            score_mapping: Mapping from ranking scores (1-6) to weighted points
        """
        self.score_mapping = score_mapping or {
            1: 1.0,    # Much better
            2: 0.75,   # Better  
            3: 0.6,    # Slightly better
            4: 0.4,    # Slightly worse
            5: 0.25,   # Worse
            6: 0.0     # Much worse
        }
    
    def aggregate_scores(
        self,
        comparison_results: List[Tuple[str, float, float, float]],
        comparison_metadata: List[Tuple[str, int, int]],
        prompt_groups: Dict[str, Dict[str, Any]]
    ) -> Dict[str, List[float]]:
        """Aggregate using weighted win-loss scores."""
        
        # Initialize weighted scores
        weighted_scores = {}
        total_comparisons = {}
        
        for prompt_key, group_data in prompt_groups.items():
            num_responses = len(group_data["conversations"])
            weighted_scores[prompt_key] = [0.0 for _ in range(num_responses)]
            total_comparisons[prompt_key] = [0 for _ in range(num_responses)]
        
        # Process each comparison result
        for (request_id, score_1, score_2, ranking_score), (prompt_key, resp_i, resp_j) in zip(
            comparison_results, comparison_metadata
        ):
            # Convert ranking score to weighted points
            ranking_int = int(round(ranking_score))
            
            if ranking_int <= 3:
                # Response i wins with different magnitudes
                weight_i = self.score_mapping.get(ranking_int, 0.5)
                weight_j = 1.0 - weight_i
            else:
                # Response j wins with different magnitudes
                weight_j = self.score_mapping.get(ranking_int, 0.5)
                weight_i = 1.0 - weight_j
            
            # Accumulate weighted scores
            weighted_scores[prompt_key][resp_i] += weight_i
            weighted_scores[prompt_key][resp_j] += weight_j
            total_comparisons[prompt_key][resp_i] += 1
            total_comparisons[prompt_key][resp_j] += 1
        
        # Calculate final scores as weighted averages
        final_scores = {}
        for prompt_key, group_data in prompt_groups.items():
            num_responses = len(group_data["conversations"])
            final_scores[prompt_key] = []
            for resp_idx in range(num_responses):
                if total_comparisons[prompt_key][resp_idx] > 0:
                    avg_weighted_score = weighted_scores[prompt_key][resp_idx] / total_comparisons[prompt_key][resp_idx]
                else:
                    avg_weighted_score = 0.5  # Neutral score if no comparisons
                final_scores[prompt_key].append(avg_weighted_score)
            
        return final_scores
    
    @property
    def name(self) -> str:
        return "weighted_win_loss"


class CombinedAggregator(PairwiseRewardAggregator):
    """Combine individual scores with pairwise rankings."""
    
    def __init__(self, alpha: float = 0.5, individual_range: Tuple[float, float] = (1.0, 5.0)):
        """
        Args:
            alpha: Weight for individual component (1-alpha for pairwise component)
            individual_range: (min_score, max_score) for individual score normalization
        """
        self.alpha = alpha
        self.individual_aggregator = IndividualScoreAggregator(individual_range)
        self.pairwise_aggregator = WeightedWinLossAggregator()
    
    def aggregate_scores(
        self,
        comparison_results: List[Tuple[str, float, float, float]],
        comparison_metadata: List[Tuple[str, int, int]],
        prompt_groups: Dict[str, Dict[str, Any]]
    ) -> Dict[str, List[float]]:
        """Aggregate using combined individual + pairwise approach."""
        
        # Get scores from both methods
        individual_scores = self.individual_aggregator.aggregate_scores(
            comparison_results, comparison_metadata, prompt_groups
        )
        pairwise_scores = self.pairwise_aggregator.aggregate_scores(
            comparison_results, comparison_metadata, prompt_groups
        )
        
        # Combine the scores
        final_scores = {}
        for prompt_key in prompt_groups.keys():
            combined = []
            for i in range(len(individual_scores[prompt_key])):
                individual_component = individual_scores[prompt_key][i]
                pairwise_component = pairwise_scores[prompt_key][i]
                combined_score = self.alpha * individual_component + (1 - self.alpha) * pairwise_component
                combined.append(combined_score)
            final_scores[prompt_key] = combined
        
        return final_scores
    
    @property
    def name(self) -> str:
        return f"combined_alpha_{self.alpha}"


class BradleyTerryAggregator(PairwiseRewardAggregator):
    """Use Bradley-Terry model to estimate response strengths from pairwise comparisons."""
    
    def __init__(self, max_iterations: int = 100, tolerance: float = 1e-6):
        """
        Args:
            max_iterations: Maximum iterations for iterative estimation
            tolerance: Convergence tolerance
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def aggregate_scores(
        self,
        comparison_results: List[Tuple[str, float, float, float]],
        comparison_metadata: List[Tuple[str, int, int]],
        prompt_groups: Dict[str, Dict[str, Any]]
    ) -> Dict[str, List[float]]:
        """Aggregate using Bradley-Terry model."""
        
        final_scores = {}
        
        for prompt_key, group_data in prompt_groups.items():
            num_responses = len(group_data["conversations"])
            
            # Build comparison matrix
            wins = np.zeros((num_responses, num_responses))
            total = np.zeros((num_responses, num_responses))
            
            # Fill comparison matrix
            for (request_id, score_1, score_2, ranking_score), (pk, resp_i, resp_j) in zip(
                comparison_results, comparison_metadata
            ):
                if pk != prompt_key:
                    continue
                
                total[resp_i, resp_j] += 1
                total[resp_j, resp_i] += 1
                
                if ranking_score <= 3.0:
                    wins[resp_i, resp_j] += 1
                else:
                    wins[resp_j, resp_i] += 1
            
            # Estimate Bradley-Terry parameters
            strengths = self._estimate_bradley_terry_strengths(wins, total)
            
            # Normalize to [0, 1]
            min_strength = min(strengths)
            max_strength = max(strengths)
            
            if max_strength > min_strength:
                normalized_strengths = [(s - min_strength) / (max_strength - min_strength) for s in strengths]
            else:
                normalized_strengths = [0.5 for _ in strengths]
            
            final_scores[prompt_key] = normalized_strengths
        
        return final_scores
    
    def _estimate_bradley_terry_strengths(self, wins: np.ndarray, total: np.ndarray) -> List[float]:
        """Estimate Bradley-Terry model parameters using iterative algorithm."""
        n = wins.shape[0]
        strengths = np.ones(n)  # Initialize with equal strengths
        
        for iteration in range(self.max_iterations):
            old_strengths = strengths.copy()
            
            for i in range(n):
                numerator = 0
                denominator = 0
                
                for j in range(n):
                    if i != j and total[i, j] > 0:
                        numerator += wins[i, j]
                        denominator += total[i, j] * strengths[i] / (strengths[i] + strengths[j])
                
                if denominator > 0:
                    strengths[i] = numerator / denominator
            
            # Normalize to prevent drift
            strengths = strengths / np.mean(strengths)
            
            # Check convergence
            if np.max(np.abs(strengths - old_strengths)) < self.tolerance:
                logging.info(f"Bradley-Terry converged after {iteration + 1} iterations")
                break
        
        return strengths.tolist()
    
    @property
    def name(self) -> str:
        return "bradley_terry"


class SimpleTiebreakerAggregator(PairwiseRewardAggregator):
    """Use individual scores primarily, with simple ranking-based tiebreaking when scores are equal."""
    
    def __init__(self, score_range: Tuple[float, float] = (1.0, 5.0)):
        """
        Args:
            score_range: (min_score, max_score) for normalization
        """
        self.min_score, self.max_score = score_range
    
    def aggregate_scores(
        self,
        comparison_results: List[Tuple[str, float, float, float]],
        comparison_metadata: List[Tuple[str, int, int]],
        prompt_groups: Dict[str, Dict[str, Any]]
    ) -> Dict[str, List[float]]:
        """Aggregate using individual scores with simple ranking tiebreaking."""
        
        # Collect individual scores for each response
        individual_scores = {}
        score_counts = {}
        
        for prompt_key, group_data in prompt_groups.items():
            num_responses = len(group_data["conversations"])
            individual_scores[prompt_key] = [0.0 for _ in range(num_responses)]
            score_counts[prompt_key] = [0 for _ in range(num_responses)]
        
        # Process each comparison result
        for (request_id, score_1, score_2, ranking_score), (prompt_key, resp_i, resp_j) in zip(
            comparison_results, comparison_metadata
        ):
            # Apply simple tiebreaking logic when scores are equal
            if score_1 == score_2:
                # When individual scores are equal, use ranking to break ties
                score_1 = score_1 + 3.5 - ranking_score  # Response 1 adjustment
                score_2 = score_2 + ranking_score - 3.5  # Response 2 adjustment
            
            # Accumulate individual scores (with tiebreaking adjustments if applied)
            individual_scores[prompt_key][resp_i] += score_1
            individual_scores[prompt_key][resp_j] += score_2
            score_counts[prompt_key][resp_i] += 1
            score_counts[prompt_key][resp_j] += 1
        
        # Calculate average individual scores
        final_scores = {}
        for prompt_key, group_data in prompt_groups.items():
            num_responses = len(group_data["conversations"])
            final_scores[prompt_key] = []
            for resp_idx in range(num_responses):
                if score_counts[prompt_key][resp_idx] > 0:
                    avg_score = individual_scores[prompt_key][resp_idx] / score_counts[prompt_key][resp_idx]
                else:
                    avg_score = (self.min_score + self.max_score) / 2  # Neutral score if no comparisons
                final_scores[prompt_key].append(avg_score)
            
        return final_scores
    
    @property
    def name(self) -> str:
        return "simple_tiebreaker"


# Factory function to create aggregators
def create_aggregator(method: str, **kwargs) -> PairwiseRewardAggregator:
    """Create a reward aggregator by name."""
    aggregators = {
        "win_rate": WinRateAggregator,
        "individual_scores": IndividualScoreAggregator,
        "elo_rating": EloRatingAggregator,
        "weighted_win_loss": WeightedWinLossAggregator,
        "combined": CombinedAggregator,
        "bradley_terry": BradleyTerryAggregator,
        "simple_tiebreaker": SimpleTiebreakerAggregator,
    }
    
    if method not in aggregators:
        raise ValueError(f"Unknown aggregation method: {method}. Available: {list(aggregators.keys())}")
    
    return aggregators[method](**kwargs)


# Example usage and comparison
def compare_aggregation_methods(
    comparison_results: List[Tuple[str, float, float, float]],
    comparison_metadata: List[Tuple[str, int, int]],
    prompt_groups: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, List[float]]]:
    """Compare different aggregation methods on the same data."""
    
    methods = [
        ("win_rate", {}),
        ("individual_scores", {}),
        ("elo_rating", {"k_factor": 32}),
        ("weighted_win_loss", {}),
        ("combined", {"alpha": 0.3}),
        ("bradley_terry", {}),
        ("simple_tiebreaker", {}),
    ]
    
    results = {}
    for method_name, kwargs in methods:
        try:
            aggregator = create_aggregator(method_name, **kwargs)
            scores = aggregator.aggregate_scores(comparison_results, comparison_metadata, prompt_groups)
            results[aggregator.name] = scores
            logging.info(f"Successfully computed scores using {aggregator.name}")
        except Exception as e:
            logging.error(f"Error with {method_name}: {e}")
            results[method_name] = None
    
    return results 