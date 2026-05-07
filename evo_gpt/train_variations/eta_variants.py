# train_variations/eta_variants.py
"""
ETA helpers for training-loop progress reporting.

The class below keeps **two independent moving-average models**

* **iteration model** – latency of the last *N* plain training steps
* **eval-cycle model** – latency of the last *M* *(training + eval)* cycles

`update()` returns a small named-tuple with everything the caller (train.py)
needs to keep the rich-progress bar in sync.
"""

from __future__ import annotations

import time
from collections import deque
from datetime import datetime, timedelta
from typing import NamedTuple


class ETAUpdate(NamedTuple):
    progress_advance: int
    iter_latency_avg: float # in ms
    time_remaining_ms: float
    formatted_completion_eta: str


class ETAEstimator:
    """Common interface for both *iteration* and *eval-cycle* ETA modes."""

    def __init__(self, args, start_time: float, evaluations_remaining: int, formatted_completion_eta: str) -> None:
        self.args = args
        self.start_time = start_time

        # iteration-based moving window
        self.iter_window = deque(maxlen=args.iteration_window)
        self.iter_latency_avg = 0.0                       # ms

        # eval-cycle moving window
        self.cycle_window = deque(maxlen=args.eval_cycle_window)
        self.cycle_latency_avg = 0.0                      # ms
        self.last_cycle_end_time: float | None = None

        self.evaluations_remaining = evaluations_remaining

        self.time_remaining_ms = None
        self.formatted_completion_eta = formatted_completion_eta
        self.warmup_eval_cycles = 2

    # --------------------------------------------------------------------- #
    # public API
    # --------------------------------------------------------------------- #
    def update(
        self,
        *,
        iter_num: int,
        now: float,
        dt: float,
        is_eval_boundary: bool,
    ) -> ETAUpdate:
        """
        Parameters
        ----------
        iter_num          current *training* iteration (0-based)
        now               wall-clock timestamp at the end of the iter        (time.time())
        dt                duration of *this* training step in seconds
        is_eval_boundary  True iff `iter_num % eval_interval == 0`
        """

        # ------------------------------------------------------------------ #
        # 1)   Update latency statistics & «progress_advance»
        # ------------------------------------------------------------------ #
        progress_advance = 1

        if is_eval_boundary:
            # we just finished the first training step *after* eval
            progress_advance += self.args.eval_iters
            self.evaluations_remaining -= 1

            if self.last_cycle_end_time is not None:
                cycle_ms = (now - self.last_cycle_end_time) * 1000.0
                self.cycle_window.append(cycle_ms)
                self.cycle_latency_avg = sum(self.cycle_window) / len(self.cycle_window)

            self.last_cycle_end_time = now

        else:
            # a plain training step – collect dt for the iteration window
            if iter_num:                                   # skip the very first iter (outlier)
                self.iter_window.append(dt)
                self.iter_latency_avg = (
                    sum(self.iter_window) / len(self.iter_window) * 1000.0
                )

        # ------------------------------------------------------------------ #
        # 2)   Remaining-time forecast
        # ------------------------------------------------------------------ #
        if self.args.eta_variant == "iteration" or not self.cycle_latency_avg:
            # (fallbacks to iteration mode until we have enough cycles)
            iter_ms = self.iter_latency_avg or (dt * 1000.0)

            self.time_remaining_ms = (
                (self.args.max_iters - iter_num) * iter_ms
                + self.evaluations_remaining * self.args.eval_iters * iter_ms
            )

            eta_dt = timedelta(milliseconds=self.time_remaining_ms)
            self.formatted_completion_eta = (datetime.now() + eta_dt).strftime("%Y-%m-%d %H:%M:%S")

        if self.warmup_eval_cycles > 0:
                self.warmup_eval_cycles -= 1
        else:
            if (self.args.eta_variant == "eval_cycle") and is_eval_boundary:
                    self.time_remaining_ms = self.evaluations_remaining * self.cycle_latency_avg
                    # subtract the current iteration (already completed)
                    self.time_remaining_ms = max(0.0, self.time_remaining_ms - self.iter_latency_avg)

            if (self.time_remaining_ms is not None) and is_eval_boundary:
                eta_dt = timedelta(milliseconds=self.time_remaining_ms)
                self.formatted_completion_eta = (datetime.now() + eta_dt).strftime("%Y-%m-%d %H:%M:%S")

        return ETAUpdate(
            progress_advance=progress_advance,
            iter_latency_avg=self.iter_latency_avg,
            time_remaining_ms=self.time_remaining_ms,
            formatted_completion_eta=self.formatted_completion_eta,
        )


# convenience factory if you ever want different variants
def build_eta_estimator(args, start_time, evaluations_remaining, formatted_completion_eta) -> ETAEstimator:
    return ETAEstimator(args, start_time, evaluations_remaining, formatted_completion_eta)

