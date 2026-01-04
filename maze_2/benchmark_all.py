import os
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add current directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.train_experiment import train

# Configuration
ENVS = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6_RANDOM']
ALGOS = ['PPO', 'DQN', 'A2C', 'SARSA']
TOTAL_TIMESTEPS = 10_000_000
REWARD_ID = 'simple'
OBS_MODE = 'full_map'

RESULTS_DIR = "results/benchmark"
os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    print(f"Starting benchmark with {TOTAL_TIMESTEPS} steps per model...")
    print(f"Environments: {ENVS}")
    print(f"Algorithms: {ALGOS}")
    
    training_times = []

    for env_id in ENVS:
        for algo in ALGOS:
            print(f"\n" + "="*50)
            print(f"Running {algo} on {env_id}...")
            print("="*50)
            
            start_time = time.time()
            
            try:
                model, model_path = train(
                    env_id=env_id,
                    reward_id=REWARD_ID,
                    algo=algo,
                    obs_mode=OBS_MODE,
                    total_timesteps=TOTAL_TIMESTEPS,
                    checkpoint_freq=1_000_000,
                    save_dir=os.path.join(RESULTS_DIR, "models"),
                    log_dir=os.path.join(RESULTS_DIR, "logs"),
                    seed=42
                )
                
                duration = time.time() - start_time
                training_times.append({
                    'Environment': env_id,
                    'Algorithm': algo,
                    'Time (s)': duration,
                    'Model Path': model_path
                })
                
                # Free memory
                del model
                
            except Exception as e:
                print(f"Failed to train {algo} on {env_id}: {e}")
                import traceback
                traceback.print_exc()

    # Save times to CSV
    if training_times:
        df_times = pd.DataFrame(training_times)
        df_times.to_csv(os.path.join(RESULTS_DIR, "training_times.csv"), index=False)
        print(f"\nTraining times saved to {os.path.join(RESULTS_DIR, 'training_times.csv')}")

    # Plotting
    print("\nGenerating plots...")
    all_data = []

    for env_id in ENVS:
        for algo in ALGOS:
            # Find monitor file
            # Pattern: {env_id}_{reward_id}_{algo}_{obs_mode}_*
            prefix = f"{env_id}_{REWARD_ID}_{algo}_{OBS_MODE}"
            model_base_dir = os.path.join(RESULTS_DIR, "models")
            
            # Find matching directory (get the latest one if multiple)
            found_dir = None
            if os.path.exists(model_base_dir):
                candidates = [d for d in os.listdir(model_base_dir) if d.startswith(prefix)]
                if candidates:
                    # Sort by timestamp (last part of name)
                    candidates.sort()
                    found_dir = os.path.join(model_base_dir, candidates[-1])
            
            if found_dir:
                # Check for monitor.monitor.csv (common with some SB3 versions/wrappers) or monitor.csv
                monitor_path = os.path.join(found_dir, "monitor.monitor.csv")
                if not os.path.exists(monitor_path):
                    monitor_path = os.path.join(found_dir, "monitor.csv")
                
                if os.path.exists(monitor_path):
                    try:
                        # Skip first line (metadata)
                        df = pd.read_csv(monitor_path, skiprows=1)
                        if len(df) > 0:
                            df['Environment'] = env_id
                            df['Algorithm'] = algo
                            df['Timesteps'] = df['l'].cumsum()
                            # Rolling mean for smoother plots
                            df['Reward'] = df['r'].rolling(window=1000, min_periods=1).mean()
                            all_data.append(df)
                            print(f"Loaded data from {monitor_path}: {len(df)} episodes")
                        else:
                            print(f"Found {monitor_path} but it was empty (no episodes finished?)")
                    except Exception as e:
                        print(f"Error reading {monitor_path}: {e}")
                else:
                    print(f"No monitor file found in {found_dir}")

    if all_data:
        full_df = pd.concat(all_data)
        
        # Plot
        try:
            sns.set_theme()
            g = sns.FacetGrid(full_df, col="Environment", col_wrap=3, sharex=False, sharey=False, height=4, aspect=1.5)
            g.map_dataframe(sns.lineplot, x="Timesteps", y="Reward", hue="Algorithm")
            g.add_legend()
            
            plt.savefig(os.path.join(RESULTS_DIR, "reward_progression.png"))
            print(f"Plot saved to {os.path.join(RESULTS_DIR, 'reward_progression.png')}")
        except Exception as e:
            print(f"Error generating plot: {e}")
    else:
        print("No data found for plotting.")

if __name__ == "__main__":
    main()
