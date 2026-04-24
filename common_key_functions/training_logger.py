import numpy as np
import os
import torch

class TrainingLogger:
    def __init__(self, 
                 metric_names: list, 
                 log_file: str | None = "training_log.npz", 
                 max_steps: int | None = None):
        """
        Args:
            metric_names: Metrics names.
            log_file:     Path for the output .npz file.
            max_steps:    Total number of optimization steps. If provided, arrays are pre-allocated.
        """
        self.metric_names = metric_names
        self.log_file = log_file
        self._finalized = False

        if max_steps is not None:
            self.steps_arr = np.zeros(max_steps + 1, dtype=np.int32)
            self.metrics_arr = np.zeros((max_steps + 1, len(metric_names)), dtype=np.float32)
            self._index = 0
            self._preallocated = True
        else:
            self.steps_arr = []
            self.metrics_arr = []
            self._preallocated = False

    def log(self, step, *values):
        """
        Record metrics for a given step.

        Call as:
            `logger.log(step, loss_kl, loss_mse, loss_tv, ...)`
        where each loss is `float` or `torch.Tensor` (scalar).
        """
        if self._finalized:
            raise RuntimeError("Logger has already been closed.")

        assert len(values) == len(self.metric_names), \
            f"Expected values number the same as metrics ({len(self.metric_names)}), got {len(values)}"

        # Convert torch tensors to Python floats
        row = [v.item() if isinstance(v, torch.Tensor) else float(v) for v in values]

        if self._preallocated:
            self.steps_arr[self._index] = step
            self.metrics_arr[self._index, :] = row
            self._index += 1
        else:
            self.steps_arr.append(step)
            self.metrics_arr.append(row)

    def close(self):
        """Finalize and write the log to disk."""
        if self._finalized:
            return

        # Convert dynamic lists to arrays
        if not self._preallocated:
            self.steps_arr = np.array(self.steps_arr, dtype=np.int32)
            self.metrics_arr = np.array(self.metrics_arr, dtype=np.float32)

        os.makedirs(os.path.dirname(self.log_file) or '.', exist_ok=True)
        np.savez_compressed(
            self.log_file,
            steps=self.steps_arr,
            metrics=self.metrics_arr,
            metric_names=np.array(self.metric_names)
        )
        self._finalized = True
        print(f"Logs saved to {self.log_file}")