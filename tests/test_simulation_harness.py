"""
Synthetic Simulation Harness for Bayesian Cortex.

Verifies that the multi-armed bandit algorithms converge correctly under extreme conditions:
1. Beta-Binomial (Context Clustering) Router non-stationary drift adaptation speed.
2. LinTS & LinUCB modes under continuous multidimensional noise using the O(d) diagonal covariance approximation.
"""

from typing import Dict, List, Tuple

import numpy as np

from bayesian_cortex.router import BayesianRouter
from bayesian_cortex.storage import InMemoryStorage


class SimulatedContextEmbedder:
    """Mock embedder returning a noisy context vector."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.current_vector = np.zeros(dimension, dtype=np.float32)

    def embed_query(self, text: str) -> List[float]:
        return list(self.current_vector)

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)


def find_crossover(
    selections: List[str], window_size: int = 200, min_stable_steps: int = 100
) -> int:
    """
    Finds the number of steps after iteration 5000 at which the rolling selection rate
    of the new winner (arm_1) consistently dominates the old winner (arm_0).
    """
    num_total = len(selections)
    for i in range(5000, num_total):
        window = selections[i - window_size + 1 : i + 1]
        arm_0_count = window.count("arm_0")
        arm_1_count = window.count("arm_1")
        if arm_1_count > arm_0_count:
            # Check if it remains dominant for min_stable_steps to filter out noise
            is_stable = True
            for j in range(i, min(i + min_stable_steps, num_total)):
                sub_window = selections[j - window_size + 1 : j + 1]
                if sub_window.count("arm_1") < sub_window.count("arm_0"):
                    is_stable = False
                    break
            if is_stable:
                return i - 5000
    return -1


def run_beta_binomial_simulation(
    decay_factor: float, num_iterations: int = 10000
) -> Tuple[int, List[str], List[float]]:
    candidates = [f"arm_{i}" for i in range(10)]
    storage = InMemoryStorage()
    router = BayesianRouter(
        storage=storage, mode="clustering", decay_factor=decay_factor, embedder=None
    )

    selections = []
    rewards = []

    for t in range(num_iterations):
        # Drift success probabilities halfway
        if t < 5000:
            prob_map = {f"arm_{i}": 0.1 for i in range(10)}
            prob_map["arm_0"] = 0.8  # Arm A is winner
            prob_map["arm_1"] = 0.2  # Arm B is loser
        else:
            prob_map = {f"arm_{i}": 0.1 for i in range(10)}
            prob_map["arm_0"] = 0.1  # Arm A is loser
            prob_map["arm_1"] = 0.9  # Arm B is new winner

        chosen_candidate, trace_id = router.route_with_trace(
            "simulated_context", candidates
        )

        # Sample reward
        success_prob = prob_map[chosen_candidate]
        reward = 1.0 if np.random.rand() < success_prob else 0.0

        router.feedback_by_trace(trace_id, reward=reward)
        selections.append(chosen_candidate)
        rewards.append(reward)

    crossover = find_crossover(selections, window_size=200, min_stable_steps=100)
    return crossover, selections, rewards


def run_linear_simulation(
    mode: str,
    diagonal_covariance: bool,
    decay_factor: float,
    num_iterations: int = 10000,
) -> Tuple[int, List[str], List[float], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    candidates = [f"arm_{i}" for i in range(10)]
    dimension = 5
    embedder = SimulatedContextEmbedder(dimension=dimension)
    storage = InMemoryStorage()

    router = BayesianRouter(
        storage=storage,
        embedder=embedder,
        mode=mode,
        decay_factor=decay_factor,
        diagonal_covariance=diagonal_covariance,
        exploration_weight=0.5,
        lambda_val=1.0,
        similarity_threshold=0.0,  # Force all queries to map to a single context cluster
    )

    # Base context vector
    mu = np.array([1.0, 0.5, 0.2, 0.1, 0.0], dtype=np.float32)
    mu_norm_sq = np.sum(mu**2)

    selections = []
    rewards = []

    for t in range(num_iterations):
        # Generate context vector with continuous noise
        noise = np.random.normal(0, 0.05, size=dimension).astype(np.float32)
        x_t = mu + noise
        embedder.current_vector = x_t

        # Define true hidden weight vectors theta*
        if t < 5000:
            c_vals = {f"arm_{i}": 0.1 for i in range(10)}
            c_vals["arm_0"] = 0.8
            c_vals["arm_1"] = 0.2
        else:
            c_vals = {f"arm_{i}": 0.1 for i in range(10)}
            c_vals["arm_0"] = 0.1
            c_vals["arm_1"] = 0.9

        # Map c_vals to true theta_a* projection
        theta_stars = {arm: (c_vals[arm] / mu_norm_sq) * mu for arm in candidates}

        chosen_candidate, trace_id = router.route_with_trace(
            "simulated_context", candidates
        )

        # Linear expected reward: x_t^T * theta*
        expected_reward = np.dot(x_t, theta_stars[chosen_candidate])
        success_prob = np.clip(expected_reward, 0.0, 1.0)

        # Sample reward
        reward = 1.0 if np.random.rand() < success_prob else 0.0

        router.feedback_by_trace(trace_id, reward=reward)
        selections.append(chosen_candidate)
        rewards.append(reward)

    crossover = find_crossover(selections, window_size=200, min_stable_steps=100)

    # Retrieve final parameters from storage
    final_params = {}
    for c in candidates:
        precision, reward_vector = storage.get_linear_params(c)
        if precision is not None:
            final_params[c] = (precision, reward_vector)

    return crossover, selections, rewards, final_params


def test_beta_binomial_adaptation():
    """
    Verify the adaptation speed of Beta-Binomial router under non-stationary drift.
    Ensures that decay_factor < 1.0 adapts significantly faster than decay_factor = 1.0.
    """
    print("\n--- Running Beta-Binomial (Context Clustering) Simulation ---")

    # Test multiple decay factors
    decay_factors = [1.0, 0.99, 0.95, 0.90]
    results = {}

    for df in decay_factors:
        crossover, selections, rewards = run_beta_binomial_simulation(decay_factor=df)
        results[df] = crossover
        print(f"Decay Factor {df:.2f}: Crossover at {crossover} steps after drift")

    # Assertions
    # With decay_factor = 1.0, the router has accumulated 4000+ successes for arm_0.
    # It takes a substantial number of steps to decay the large counts (usually >800 steps).
    assert (
        results[1.0] == -1 or results[1.0] > 800
    ), f"Decay factor 1.0 adapted unexpectedly fast: {results[1.0]} steps"

    # With decay_factor = 0.99, effective window is ~100 steps. It should adapt quickly.
    assert (
        0 < results[0.99] < 500
    ), f"Decay factor 0.99 adaptation took too long: {results[0.99]} steps"

    # With decay_factor = 0.95, effective window is ~20 steps. It should adapt extremely quickly.
    assert (
        0 < results[0.95] < 300
    ), f"Decay factor 0.95 adaptation took too long: {results[0.95]} steps"

    # Verify that decaying models adapted strictly faster than the non-decaying model (decay_factor = 1.0)
    if results[1.0] != -1:
        assert (
            results[0.99] < results[1.0]
        ), f"Decay factor 0.99 ({results[0.99]}) was not faster than 1.0 ({results[1.0]})"
        assert (
            results[0.95] < results[1.0]
        ), f"Decay factor 0.95 ({results[0.95]}) was not faster than 1.0 ({results[1.0]})"


def test_linear_bandits_stability_and_drift():
    """
    Verify LinTS and LinUCB stability and adaptation under continuous noise
    with O(d) diagonal covariance approximation.
    """
    print("\n--- Running Linear Bandits (LinTS / LinUCB) Simulation ---")

    for mode in ["lints", "linucb"]:
        crossover, selections, rewards, final_params = run_linear_simulation(
            mode=mode, diagonal_covariance=True, decay_factor=0.99, num_iterations=10000
        )

        print(
            f"Mode: {mode} (Diagonal Covariance) -> Crossover at {crossover} steps after drift"
        )

        # Verify adaptation
        assert (
            0 < crossover < 1500
        ), f"Linear mode {mode} failed to adapt timely: {crossover} steps"

        # Verify covariance stability for all candidates that have parameters
        assert len(final_params) > 0, "No linear parameters found in storage"

        for candidate, (precision, reward_vector) in final_params.items():
            # Check shape: 5-dim query context + 1-dim intercept = 6 elements
            assert precision.shape == (6,)
            assert reward_vector.shape == (6,)

            # Ensure no NaNs or Infinities
            assert np.all(
                np.isfinite(precision)
            ), f"NaN/Inf precision found for {candidate} in {mode}"
            assert np.all(
                np.isfinite(reward_vector)
            ), f"NaN/Inf reward vector found for {candidate} in {mode}"

            # Ensure diagonal precision elements do not diverge or drop below lambda (1.0)
            assert np.all(
                precision >= 1.0
            ), f"Precision dropped below lambda for {candidate} in {mode}"

            # Theoretical bound check: under decay_factor=0.99, lambda=1.0, and |x| <= 1.5,
            # precision should stabilize below ~301.0. Let's assert a safe margin of 500.0.
            assert np.all(
                precision <= 500.0
            ), f"Precision diverged/blew up for {candidate} in {mode}: {precision}"


if __name__ == "__main__":
    # If run directly, run simulations and print comparison summary
    np.random.seed(42)

    print("=" * 70)
    print("STATISTICAL SIMULATION HARNESS FOR BAYESIAN CORTEX")
    print("=" * 70)

    # 1. Run Beta-Binomial clustering mode comparison
    print("\n1. Beta-Binomial (Context Clustering) Adaptation Speed:")
    print("-" * 65)
    print(
        f"{'Decay Factor':<15} | {'Crossover (Steps)':<20} | {'Adaptation Status':<20}"
    )
    print("-" * 65)
    for df in [1.0, 0.99, 0.95, 0.90]:
        crossover, _, _ = run_beta_binomial_simulation(decay_factor=df)
        status = (
            "NO ADAPTATION" if crossover == -1 else f"SUCCESS (< {crossover} steps)"
        )
        print(f"{df:<15.2f} | {str(crossover):<20} | {status:<20}")

    # 2. Run Linear bandits comparison (Diagonal vs Full Covariance)
    print("\n2. Linear Contextual Bandits Stability & Adaptation Speed:")
    print("-" * 80)
    print(
        f"{'Mode':<10} | {'Covariance':<10} | {'Crossover (Steps)':<20} | {'Min Precision':<15} | {'Max Precision':<15}"
    )
    print("-" * 80)

    for mode in ["lints", "linucb"]:
        for diag in [True, False]:
            crossover, selections, _, final_params = run_linear_simulation(
                mode=mode,
                diagonal_covariance=diag,
                decay_factor=0.99,
                num_iterations=10000,
            )

            # Get min/max precision across candidates
            all_prec_min = float("inf")
            all_prec_max = float("-inf")
            for prec, _ in final_params.values():
                if diag:
                    all_prec_min = min(all_prec_min, prec.min())
                    all_prec_max = max(all_prec_max, prec.max())
                else:
                    # For full covariance, precision is a 2D matrix
                    eigvals = np.linalg.eigvalsh(prec)
                    all_prec_min = min(all_prec_min, eigvals.min())
                    all_prec_max = max(all_prec_max, eigvals.max())

            cov_type = "Diagonal" if diag else "Full"
            print(
                f"{mode:<10} | {cov_type:<10} | {str(crossover):<20} | {all_prec_min:<15.4f} | {all_prec_max:<15.4f}"
            )

    print("=" * 70)
