import glob, re
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


plt.figure(figsize=(10, 8))
ax = plt.gca()

# Example: one algorithm (blue)
EPISODE_SIZE = 16

# --- Algorithm 1: IS ---
eps, mean, low, high = load_runs("logs/run_[0-9]*_aes_is_reward_history.npy")
samples = eps * EPISODE_SIZE
ax.plot(samples, mean, color="C0", label="IS")
ax.fill_between(samples, low, high, color="C0", alpha=0.3)

# --- Algorithm 2: SF / no IS ---
eps2, mean2, low2, high2 = load_runs("logs/run_[0-9]*_aes_sf_reward_history.npy")
samples2 = eps2 * EPISODE_SIZE
ax.plot(samples2, mean2, color="C1", label="SF (no IS)")
ax.fill_between(samples2, low2, high2, color="C1", alpha=0.3)

# --- Algorithm 3: IS + KL reference ---
eps3, mean3, low3, high3 = load_runs("logs/run_[0-9]*_aes_iskl_reward_history.npy")
samples3 = eps3 * EPISODE_SIZE
ax.plot(samples3, mean3, color="C2", label="ISKL Ref")
ax.fill_between(samples3, low3, high3, color="C2", alpha=0.3)

# Labels and formatting
ax.set_xlabel("Samples")
ax.set_ylabel("Rewards (aesthetics)")
ax.set_title("DDPO: IS vs SF vs ISKL")
ax.legend()  # <-- add legend

plt.tight_layout()
plt.savefig("plt/reward_plot_ddpo_aesthetics.png", bbox_inches="tight")

