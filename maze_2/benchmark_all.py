import os
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add current directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.train_experiment import train
from analysis.generate_figures import (
    load_all_experiments,
    figure_1_learning_curves,
    figure_2_final_performance,
    figure_3_phase_completion,
    figure_4_heatmap,
    generate_summary_table,
)
from analysis.compare_experiments import load_metrics, print_summary_table
import analysis.compare_experiments as compare_exp
import plot_results

# Configuration
ENVS = ["E5_FIXED", "E6_RANDOM"]  #'E1' #'E2', 'E3_FIXED', 'E4', 'E5_FIXED', 'E6_RANDOM'
ALGOS = ["PPO", "DQN", "A2C"]  # 'SARSA' #"DQN", "A2C"
TOTAL_TIMESTEPS = 20_000_000
REWARD_ID = "simple"
OBS_MODE = "full_map"

RESULTS_DIR = "results/benchmark"
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def main():
    print(f"Starting benchmark with {TOTAL_TIMESTEPS} steps per model...")
    print(f"Environments: {ENVS}")
    print(f"Algorithms: {ALGOS}")

    training_times = []

    for env_id in ENVS:
        for algo in ALGOS:
            print(f"\n" + "=" * 50)
            print(f"Running {algo} on {env_id}...")
            print("=" * 50)

            start_time = time.time()

            try:
                model, model_path = train(
                    env_id=env_id,
                    reward_id=REWARD_ID,
                    algo=algo,
                    obs_mode=OBS_MODE,
                    total_timesteps=TOTAL_TIMESTEPS,
                    checkpoint_freq=2_000_000,
                    save_dir=MODELS_DIR,
                    log_dir=LOGS_DIR,
                    seed=42,
                )

                duration = time.time() - start_time
                training_times.append(
                    {
                        "Environment": env_id,
                        "Algorithm": algo,
                        "Time (s)": duration,
                        "Model Path": model_path,
                    }
                )

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

    # =========================================================================
    # ANALYSIS & PLOTTING
    # =========================================================================
    print("\n" + "=" * 50)
    print("GENERATING ANALYSIS")
    print("=" * 50)

    # 1. Generate Figures (Publication Ready)
    print("\n--- Running generate_figures analysis ---")
    try:
        df = load_all_experiments(MODELS_DIR)
        if df is not None and not df.empty:
            figure_1_learning_curves(df, FIGURES_DIR)
            figure_2_final_performance(df, FIGURES_DIR)
            figure_3_phase_completion(df, FIGURES_DIR)
            figure_4_heatmap(df, FIGURES_DIR)
            generate_summary_table(df, FIGURES_DIR)
        else:
            print("No data found for generate_figures.")
    except Exception as e:
        print(f"Error in generate_figures: {e}")
        import traceback

        traceback.print_exc()

    # 2. Compare Experiments (Summary Table)
    print("\n--- Running compare_experiments analysis ---")
    try:
        experiments = load_metrics(MODELS_DIR)
        if experiments:
            print_summary_table(experiments)

            # Monkeypatch plt.show to avoid blocking if running headless
            original_show = plt.show
            plt.show = lambda: None

            # We can't easily change where it saves "comparison_curves.png",
            # so we might move it after generation or just let it be.
            # It saves to current working directory.
            compare_exp.plot_learning_curves(experiments)

            # Restore plt.show
            plt.show = original_show

            # Move the file if it exists
            if os.path.exists("comparison_curves.png"):
                import shutil

                shutil.move(
                    "comparison_curves.png", os.path.join(FIGURES_DIR, "comparison_curves.png")
                )
                print(f"Moved comparison_curves.png to {FIGURES_DIR}")
        else:
            print("No data found for compare_experiments.")
    except Exception as e:
        print(f"Error in compare_experiments: {e}")
        import traceback

        traceback.print_exc()

    # 3. Plot Results (Alternative Plots)
    print("\n--- Running plot_results analysis ---")
    try:
        # Configure plot_results globals
        plot_results.RESULTS_DIR = MODELS_DIR
        plot_results.OUTPUT_DIR = PLOTS_DIR

        data = plot_results.load_all_metrics(MODELS_DIR)
        if data:
            plot_results.plot_learning_curves(data)
            plot_results.plot_final_comparison(data)
        else:
            print("No data found for plot_results.")
    except Exception as e:
        print(f"Error in plot_results: {e}")
        import traceback

        traceback.print_exc()

    print("\nBenchmark and Analysis Complete!")


if __name__ == "__main__":
    main()
