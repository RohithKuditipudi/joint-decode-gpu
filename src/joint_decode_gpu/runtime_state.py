from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WorkerCommands:
    admit: list[str] = field(default_factory=list)
    abort: str | None = None


@dataclass
class JointDecodeRuntimeState:
    pending_tokens: dict[str, list[int]] = field(default_factory=dict)
    latest_commands: WorkerCommands | None = None

    def reset(self) -> None:
        self.pending_tokens.clear()
        self.latest_commands = None

    def publish_commands(self, *, admit: list[str] | None = None, abort: str | None = None) -> None:
        if not admit and abort is None:
            return
        self.latest_commands = WorkerCommands(admit=list(admit or []), abort=abort)

    def drain_commands(self) -> WorkerCommands:
        commands = self.latest_commands or WorkerCommands()
        self.latest_commands = None
        return commands


runtime_state = JointDecodeRuntimeState()
