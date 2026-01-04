"""
Custom callbacks for tracking experiment metrics.
"""

import os
import json
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class NumpyEncoder(json.JSONEncoder):
    """
    Custom encoder to handle NumPy data types in JSON.
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class MetricsCallback(BaseCallback):
    """
    Tracks detailed metrics per episode for later analysis.
    """

    def __init__(self, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path

        # Episode tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []

        # Phase tracking (for E3+ with key/door)
        self.key_pickups = []
        self.door_opens = []

        # Current episode stats
        self.current_reward = 0
        self.current_length = 0
        self.picked_key = False
        self.opened_door = False

        # Rolling stats for logging
        self.last_100_successes = []

    def _on_step(self) -> bool:
        # Accumulate rewards
        self.current_reward += self.locals["rewards"][0]
        self.current_length += 1

        # Check for key pickup (reward spike of ~20)
        if self.locals["rewards"][0] >= 15 and not self.picked_key:
            self.picked_key = True

        # Check for door open (another reward spike)
        if self.locals["rewards"][0] >= 15 and self.picked_key and not self.opened_door:
            self.opened_door = True

        # Episode finished
        dones = self.locals.get("dones", self.locals.get("done", [False]))
        if dones[0]:
            # Check if goal was reached (reward of 100)
            success = self.current_reward >= 90

            self.episode_rewards.append(self.current_reward)
            self.episode_lengths.append(self.current_length)
            self.episode_successes.append(1 if success else 0)
            self.key_pickups.append(1 if self.picked_key else 0)
            self.door_opens.append(1 if self.opened_door else 0)

            # Rolling success rate
            self.last_100_successes.append(1 if success else 0)
            if len(self.last_100_successes) > 100:
                self.last_100_successes.pop(0)

            # Log to TensorBoard
            if len(self.episode_rewards) % 10 == 0:
                self.logger.record("custom/episode_reward", self.current_reward)
                self.logger.record("custom/episode_length", self.current_length)
                self.logger.record("custom/success_rate_100", np.mean(self.last_100_successes))
                self.logger.record("custom/key_pickup_rate_100", np.mean(self.key_pickups[-100:]))
                self.logger.record("custom/door_open_rate_100", np.mean(self.door_opens[-100:]))

            # Reset for next episode
            self.current_reward = 0
            self.current_length = 0
            self.picked_key = False
            self.opened_door = False

        return True

    def _on_training_end(self):
        """Save all metrics to JSON for later analysis."""
        metrics = {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "episode_successes": self.episode_successes,
            "key_pickups": self.key_pickups,
            "door_opens": self.door_opens,
            "summary": {
                "total_episodes": len(self.episode_rewards),
                "mean_reward": float(np.mean(self.episode_rewards)),
                "std_reward": float(np.std(self.episode_rewards)),
                "mean_length": float(np.mean(self.episode_lengths)),
                "success_rate": float(np.mean(self.episode_successes)),
                "key_pickup_rate": float(np.mean(self.key_pickups)),
                "door_open_rate": float(np.mean(self.door_opens)),
                "final_100_success_rate": float(np.mean(self.episode_successes[-100:])),
            },
        }

        filepath = os.path.join(self.save_path, "metrics.json")
        with open(filepath, "w") as f:
            # Pass the custom encoder here
            json.dump(metrics, f, indent=2, cls=NumpyEncoder)

        if self.verbose:
            print(f"\nMetrics saved to {filepath}")
            print(
                f"Final success rate (last 100): {metrics['summary']['final_100_success_rate']:.2%}"
            )
