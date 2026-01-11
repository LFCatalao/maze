
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Configuration
LOGS_DIR = os.path.join("results", "benchmark", "logs")
MODELS_DIR = os.path.join("results", "benchmark", "models")
OUTPUT_FILE = os.path.join("results", "benchmark", "training_tims_final.csv")
PLOTS_DIR = os.path.join("results", "benchmark", "plots_generated")

# Ensure plots directory exists
os.makedirs(PLOTS_DIR, exist_ok=True)

def get_training_data(log_dir, tags_to_extract=None):
    """
    Extract training duration and curves from TensorBoard logs.
    tags_to_extract: list of scalar tags to retrieve. Defaults to ['rollout/ep_rew_mean']
    Returns: (duration, data_dict)
             where data_dict is {tag: {'steps': [], 'values': []}}
    """
    if tags_to_extract is None:
        tags_to_extract = ['rollout/ep_rew_mean']

    # Find event files
    event_files = glob.glob(os.path.join(log_dir, "**", "events.out.tfevents*"), recursive=True)
    
    # Initialize data structures
    data_dict = {tag: {'steps': [], 'values': []} for tag in tags_to_extract}
    start_times = []
    end_times = []
    
    if not event_files:
        return 0, data_dict
    
    for ef in event_files:
        try:
            # Load scalars
            ea = EventAccumulator(ef, size_guidance={'scalars': 0})
            ea.Reload()
            
            # 1. Calculate Duration
            # Find any scalar tag to get wall_time range
            available_tags = ea.Tags()['scalars']
            if available_tags:
                # Use the first scalar to get timestamps
                sample_events = ea.Scalars(available_tags[0])
                if sample_events:
                    start_times.append(sample_events[0].wall_time)
                    end_times.append(sample_events[-1].wall_time)
            
            # 2. Extract Data for requested tags
            for tag in tags_to_extract:
                if tag in available_tags:
                    events = ea.Scalars(tag)
                    for e in events:
                        data_dict[tag]['steps'].append(e.step)
                        data_dict[tag]['values'].append(e.value)
                
        except Exception as e:
            print(f"    Error reading {ef}: {e}")
            
    if not start_times:
        duration = 0
    else:
        duration = max(end_times) - min(start_times)
        
    # Sort samples by step for each tag
    for tag in data_dict:
        if data_dict[tag]['steps']:
            sorted_pairs = sorted(zip(data_dict[tag]['steps'], data_dict[tag]['values']))
            steps, values = zip(*sorted_pairs)
            data_dict[tag]['steps'] = list(steps)
            data_dict[tag]['values'] = list(values)
        
    return duration, data_dict

def parse_dirname(dirname):
    """
    Parse directory name to extract Environment and Algorithm.
    Expected format: {ENV}_{REWARD}_{ALGO}_{OBS}_{TIMESTAMP}
    """
    algos = ["PPO", "DQN", "A2C", "SARSA"]
    
    parts = dirname.split('_')
    
    # Identify Algorithm
    found_algo = "Unknown"
    algo_idx = -1
    
    for i, part in enumerate(parts):
        if part in algos:
            found_algo = part
            algo_idx = i
            break
            
    if algo_idx != -1:
        # Assuming REWARD is immediately before ALGO
        # ENV is everything before REWARD
        # This handles ENV names with underscores like E3_FIXED
        
        # Check if index is valid (needs at least 1 part before algo for reward)
        if algo_idx > 0:
            # We assume the part before algo is Reward ID (e.g. "simple")
            # So Env is parts[:algo_idx-1]
            
            env_parts = parts[:algo_idx-1]
            env_id = "_".join(env_parts)
            return env_id, found_algo
            
    return dirname, found_algo

def plot_environment_rewards(env_name, data_records, output_dir):
    """Generate a plot for a specific environment with all algorithms."""
    plt.figure(figsize=(10, 6))
    
    # data_records is a list of dicts: {'Algorithm': ..., 'Steps': [...], 'Rewards': [...]}
    
    # Plot each algorithm
    has_data = False
    
    # Define a color palette manually or use seaborn's default
    colors = sns.color_palette("bright", 10)
    algo_colors = {
        "PPO": colors[0],
        "DQN": colors[1],
        "A2C": colors[2],
        "SARSA": colors[3]
    }
    
    for i, rec in enumerate(data_records):
        algo_name = rec['Algorithm']
        steps = rec['Steps']
        rewards = rec['Rewards']
        
        if not steps:
            print(f"  Warning: No step data for {algo_name} in {env_name}")
            continue
            
        has_data = True
        
        # Smooth data for cleaner plots (moving average)
        window = 50
        if len(rewards) > window * 2:
            rewards_series = pd.Series(rewards)
            smooth_rewards = rewards_series.rolling(window=window, min_periods=1).mean()
            # Plot raw transparently and smooth solidly? Or just smooth? 
            # Let's plot smooth for clarity
            plt.plot(steps, smooth_rewards, label=f"{algo_name}", linewidth=2, color=algo_colors.get(algo_name, None))
        else:
            plt.plot(steps, rewards, label=f"{algo_name}", linewidth=2, color=algo_colors.get(algo_name, None))

    if has_data:
        plt.title(f"Learning Curve: {env_name} (Smoothed)", fontsize=14)
        plt.xlabel("Timesteps", fontsize=12)
        plt.ylabel("Mean Episode Reward", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        filename = f"{env_name}_rewards.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved plot to {filepath}")
    else:
        plt.close()
        print(f"  No data to plot for {env_name}")

def main():
    if not os.path.exists(LOGS_DIR):
        print(f"Directory not found: {LOGS_DIR}")
        return

    # Data collection
    time_records = []
    # Structure for plotting: dict mapping Env -> list of algo records
    plot_data = {} 
    
    print(f"Reading logs from {os.path.abspath(LOGS_DIR)}...")
    
    entries = os.listdir(LOGS_DIR)
    print(f"Found {len(entries)} entries")

    for dirname in entries:
        dirpath = os.path.join(LOGS_DIR, dirname)
        if not os.path.isdir(dirpath):
            continue
            
        print(f"Processing {dirname}...")
        
        # Get data
        tags_to_fetch = ['rollout/ep_rew_mean', 'rollout/ep_len_mean']
        duration, data_dict = get_training_data(dirpath, tags_to_extract=tags_to_fetch)
        
        # Process rewards
        rewards_data = data_dict['rollout/ep_rew_mean']
        steps = rewards_data['steps']
        rewards = rewards_data['values']
        
        # Process lengths
        lengths_data = data_dict['rollout/ep_len_mean']
        ep_len_values = lengths_data['values']
        
        # Extract last values
        last_reward = rewards[-1] if rewards else None
        last_length = ep_len_values[-1] if ep_len_values else None

        # Parse info
        env, algo = parse_dirname(dirname)
        model_path = os.path.join(MODELS_DIR, dirname)
        
        print(f"  -> Env: {env}, Algo: {algo}, Time: {duration:.2f}s, Points: {len(steps)}")
        
        # Store time record
        time_records.append({
            "Environment": env,
            "Algorithm": algo,
            "Time (s)": duration,
            "Mean Reward (last 100)": last_reward,
            "Mean Length (last 100)": last_length,
            "Model Path": model_path
        })
        
        # Store plot data
        if env not in plot_data:
            plot_data[env] = []
        plot_data[env].append({
            "Algorithm": algo,
            "Steps": steps,
            "Rewards": rewards
        })
        
    # Save Time CSV
    if time_records:
        df = pd.DataFrame(time_records)
        # Reorder columns
        cols = ["Environment", "Algorithm", "Time (s)", "Mean Reward (last 100)", "Mean Length (last 100)", "Model Path"]
        # Filter for existing columns in case something is missing
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
        
        df = df.sort_values(by=["Environment", "Algorithm"])
        
        print(f"\nSaving training times to {OUTPUT_FILE}")
        df.to_csv(OUTPUT_FILE, index=False)
        print(df.to_string())
    else:
        print("No records found.")

    # Generate Plots
    print(f"\nGenerating plots in {PLOTS_DIR}...")
    # Set style
    sns.set_theme(style="whitegrid")
    
    for env_name, records in plot_data.items():
        print(f"Plotting {env_name}...")
        plot_environment_rewards(env_name, records, PLOTS_DIR)

    print("\nDone.")

if __name__ == "__main__":
    main()
