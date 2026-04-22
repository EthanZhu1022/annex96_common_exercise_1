from __future__ import annotations

from time import perf_counter


def format_duration(seconds: float) -> str:
    seconds_i = max(int(round(seconds)), 0)
    hours, rem = divmod(seconds_i, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


class ProgressTimer:
    def __init__(self, total: int, unit: str, label: str = "progress", log_every: int = 1) -> None:
        self.total = max(int(total), 1)
        self.unit = unit
        self.label = label
        self.log_every = max(int(log_every), 1)
        self.start = perf_counter()
        self.last = self.start

    def step(self, current: int) -> None:
        current = min(max(int(current), 1), self.total)
        now = perf_counter()
        last_elapsed = now - self.last
        elapsed = now - self.start
        avg = elapsed / current
        remaining = max(self.total - current, 0) * avg
        estimated_total = avg * self.total
        self.last = now

        if current == 1 or current == self.total or current % self.log_every == 0:
            print(
                f"[{self.label}] {self.unit} {current}/{self.total} | "
                f"last {format_duration(last_elapsed)} | "
                f"avg {format_duration(avg)}/{self.unit.lower()} | "
                f"elapsed {format_duration(elapsed)} | "
                f"ETA {format_duration(remaining)} | "
                f"est_total {format_duration(estimated_total)}",
                flush=True,
            )
