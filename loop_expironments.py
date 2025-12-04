import re
import subprocess

CONFIG_PATH = "./config/base_is.py"

def update_config(run_name, seed):
    """Modify config.run_name and config.seed inside base_is.py."""
    with open(CONFIG_PATH, "r") as f:
        content = f.read()

    # Replace config.run_name
    content = re.sub(
        r'config\.run_name\s*=\s*".*?"',
        f'config.run_name = "{run_name}"',
        content
    )

    # Replace config.seed
    content = re.sub(
        r'config\.seed\s*=\s*\d+',
        f'config.seed = {seed}',
        content
    )

    with open(CONFIG_PATH, "w") as f:
        f.write(content)

    print(f"[INFO] Updated config: run_name={run_name}, seed={seed}")

def main():
    start_run_idx = 1
    start_seed = 40

    for i in range(5):
        run_name = f"run_{start_run_idx + i}_kl_ref"
        seed = start_seed + i

        update_config(run_name, seed)

        print(f"[INFO] Launching training: {run_name}")
        subprocess.run(["accelerate", "launch", "scripts/train.py"])

if __name__ == "__main__":
    main()
