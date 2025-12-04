import glob
import numpy as np
import matplotlib.pyplot as plt


def load_runs(pattern):
    """Load multiple .npy files and return mean + min/max over runs."""
    runs = [np.load(f) for f in glob.glob(pattern)]
    # Make sure all runs have the same length (truncate to shortest if needed)
    min_len = min(r.shape[0] for r in runs)
    runs = np.stack([r[:min_len] for r in runs])   # shape: (n_runs, n_episodes)

    mean = runs.mean(axis=0)
    low  = runs.min(axis=0)   # lower envelope (range)
    high = runs.max(axis=0)   # upper envelope (range)
    episodes = np.arange(min_len)
    return episodes, mean, low, high


plt.figure(figsize=(4, 3))
ax = plt.gca()

# Example: one algorithm (blue)
eps, mean, low, high = load_runs("logs/run_*_reward_history.npy")
ax.plot(eps, mean, color="C0")
ax.fill_between(eps, low, high, color="C0", alpha=0.3)

# # Example: another algorithm (red)
# eps2, mean2, low2, high2 = load_runs("runs/alien_algo2_*.npy")
# ax.plot(eps2, mean2, color="C1")
# ax.fill_between(eps2, low2, high2, color="C1", alpha=0.3)

ax.set_xlabel("Episode")
ax.set_ylabel("rewards of compressibility")
ax.set_title("DDPO IS")
plt.tight_layout()
plt.savefig("plt/reward_plot_ddpo_is_compress.png", bbox_inches="tight")

