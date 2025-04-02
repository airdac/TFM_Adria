import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import make_swiss_roll
import os
from ...private_lmds import lmds_R, lmds_R_optimized
from ...utils import benchmark


def test_lmds_performance():
    """
    Test function to compare the performance of lmds_R and lmds_R_optimized 
    on the Swiss roll dataset.
    """
    # Parameters
    n_samples = [500, 750]
    k_values = [8, 10]
    tau_values = [0.1, 0.5]
    random_state = 42

    # Results
    results = []

    print("Starting performance comparison between lmds_R and lmds_R_optimized...")
    print("=" * 80)

    for n in n_samples:
        print(f"\nTesting with {n} samples")
        print("-" * 60)

        # Generate Swiss roll dataset
        X, color = make_swiss_roll(
            n_samples=n, random_state=random_state)

        # Compute distance matrix
        distances = squareform(pdist(X))

        for k in k_values:
            for tau in tau_values:
                print(f"  Parameters: k={k}, tau={tau}")

                # Time lmds_R
                embedding_original, original_time = benchmark(lmds_R,
                    distances, d=2, k=k, tau=tau, verbose=2)

                # Time lmds_R_optimized
                embedding_optimized, optimized_time = benchmark(lmds_R_optimized,
                    distances, d=2, k=k, tau=tau, verbose=2)

                # Calculate speedup
                speedup = original_time / optimized_time

                results.append({
                    'n_samples': n,
                    'k': k,
                    'tau': tau,
                    'original_time': original_time,
                    'optimized_time': optimized_time,
                    'speedup': speedup
                })

                print(
                    f"    Original: {original_time:.4f}s, Optimized: {optimized_time:.4f}s")
                print(
                    f"    Speedup: {speedup:.2f}x")

    # Save results to CSV
    results_df = np.array([
        [r['n_samples'], r['k'], r['tau'], r['original_time'],
         r['optimized_time'], r['speedup']]
        for r in results
    ])
    header = "n_samples,k,tau,original_time,optimized_time,speedup"
    results_path = os.path.join('d_and_c', 'LMDS', 'optimization_benchmark')
    os.makedirs(results_path, exist_ok=True)
    np.savetxt(os.path.join(results_path, 'LMDS_optimization_results.csv'),
               results_df, delimiter=',', header=header, comments='')

    # Plot summary
    plt.figure(figsize=(12, 8))

    # Group by n_samples
    for n in n_samples:
        n_results = [r for r in results if r['n_samples'] == n]
        speedups = [r['speedup'] for r in n_results]
        param_labels = [f"k={r['k']},τ={r['tau']}" for r in n_results]

        x = np.arange(len(n_results))
        plt.bar(x + 0.1 * (n_samples.index(n) - 1), speedups, width=0.1,
                label=f'n={n}', alpha=0.7)

    plt.ylabel('Speedup Factor (higher is better)')
    plt.title('Performance Comparison: lmds_R_optimized vs lmds_R')
    plt.xticks(np.arange(len(param_labels)), param_labels, rotation=45)
    plt.axhline(y=1.0, linestyle='--', color='r')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(
        results_path, 'LMDS_speedup_results.png'), dpi=300)

    # Create a detailed visualization for the largest sample size
    n = n_samples[-1]
    _, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original data
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], c=color, s=20, alpha=0.7)
    ax.set_title(f'Original Swiss Roll (n={n})')

    # lmds_R result
    best_params = sorted([r for r in results if r['n_samples'] == n],
                         key=lambda x: x['original_time'])[0]
    k, tau = best_params['k'], best_params['tau']

    distances = squareform(pdist(X))
    embedding_original = lmds_R(distances, d=2, k=k, tau=tau, verbose=0)

    ax = axes[1]
    ax.scatter(
        embedding_original[:, 0], embedding_original[:, 1], c=color, s=20, alpha=0.7)
    ax.set_title(
        f'lmds_R (k={k}, τ={tau})\nTime: {best_params["original_time"]:.2f}s')

    # lmds_R_optimized result
    embedding_optimized = lmds_R_optimized(
        distances, d=2, k=k, tau=tau, verbose=0)

    ax = axes[2]
    ax.scatter(
        embedding_optimized[:, 0], embedding_optimized[:, 1], c=color, s=20, alpha=0.7)
    ax.set_title(
        f'lmds_R_optimized\nTime: {best_params["optimized_time"]:.2f}s\nSpeedup: {best_params["speedup"]:.2f}x')

    plt.tight_layout()
    plt.savefig(os.path.join(
        results_path, 'LMDS_optimized_embedding_comparison.png'), dpi=300)

    print("\nPerformance test completed.")
    print(f"Average speedup: {np.mean([r['speedup'] for r in results]):.2f}x")
    print(f"Maximum speedup: {np.max([r['speedup'] for r in results]):.2f}x")
    print(f"Results saved to 'lmds_performance_results.csv'")
    print(f"Plots saved to 'lmds_performance_comparison.png' and 'lmds_swiss_roll_comparison.png'")


if __name__ == "__main__":
    test_lmds_performance()
