"""SolveTracker: tracks which required test paths have been solved."""

from __future__ import annotations

from models import EnvoiCall, TestingState


class SolveTracker:
    def __init__(self, required_paths: list[str]) -> None:
        self.required_paths = required_paths
        self.required_paths_set = set(required_paths)
        self.solved: set[str] = set()
        self.all_calls: list[EnvoiCall] = []
        self._seen_call_keys: set[str] = set()

    def _call_key(self, call: EnvoiCall) -> str:
        return (
            f"{call.path}|{call.timestamp}"
            f"|{call.status_code}|{call.duration_ms}"
        )

    def update(self, envoi_calls: list[EnvoiCall]) -> None:
        for call in envoi_calls:
            key = self._call_key(call)
            if key in self._seen_call_keys:
                continue
            self._seen_call_keys.add(key)
            self.all_calls.append(call)
            if (
                call.result
                and call.result.total > 0
                and call.result.passed == call.result.total
            ):
                self.solved.add(call.path)

    def get_unsolved_paths(self) -> list[str]:
        return [p for p in self.required_paths if p not in self.solved]

    def get_latest_call_for_path(self, path: str) -> EnvoiCall | None:
        for call in reversed(self.all_calls):
            if call.path == path:
                return call
        return None

    def snapshot(self) -> TestingState:
        latest = self.all_calls[-1] if self.all_calls else None
        latest_passed = (
            latest.result.passed if latest and latest.result else None
        )
        latest_total = (
            latest.result.total if latest and latest.result else None
        )
        return TestingState(
            solved_paths=len(self.solved),
            total_paths=len(self.required_paths),
            latest_path=latest.path if latest else None,
            latest_passed=latest_passed,
            latest_total=latest_total,
            latest_status_code=latest.status_code if latest else None,
            latest_error=latest.error if latest else None,
        )
