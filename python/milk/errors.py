"""The exception hierarchy of the ``milk`` package.

Every exception class of the package lives here, under :class:`MilkError`. See design-v7 §4.7
(``specs/pipelines/design-v7.md``) for the contract. pyMilk exceptions are never wrapped.
"""

from __future__ import annotations

_CONTEXT_FIELDS = ("pipeline", "compu", "task", "path", "cmd", "stderr")


class MilkError(Exception):
    """Root of the package hierarchy; carries optional structured context.

    The non-empty context fields are appended to ``str(err)``, one ``name: value`` per line, so
    that a caller (or a log) sees where the failure happened without parsing the message.
    """

    def __init__(
        self,
        message: str = "",
        *,
        pipeline: str | None = None,
        compu: str | None = None,
        task: str | None = None,
        path: object | None = None,
        cmd: object | None = None,
        stderr: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.pipeline = pipeline
        self.compu = compu
        self.task = task
        self.path = path
        self.cmd = cmd
        self.stderr = stderr

    def __str__(self) -> str:
        lines = [self.message]
        for field in _CONTEXT_FIELDS:
            value = getattr(self, field)
            if value is None or value == "":
                continue
            lines.append(f"{field}: {value}")
        return "\n".join(lines)


class EnvError(MilkError):
    """A required ``MILK_*`` variable or milk shared library is missing."""


class ConfigError(MilkError):
    """A pipeline configuration folder or file is missing or unusable."""


class TaskError(MilkError):
    """A pipeline task failed."""


class TaskCannotRunError(TaskError):
    """A task's preconditions do not hold, so it cannot run."""


class TaskDidNotSucceedError(TaskError):
    """A task ran but its success check still fails."""


class ComputeUnitError(MilkError):
    """A compute unit subprocess failed."""


class LinkageError(MilkError):
    """A system linkage is unknown, or cannot be applied or removed safely."""


class BestEffortError(MilkError):
    """Aggregate of the failures of a best-effort loop that attempted every item.

    Several causes exist, so none is chained; they are in :attr:`failures`.
    """

    def __init__(
        self,
        failures: list[tuple[str, Exception]],
        message: str = "Best-effort steps failed",
        *,
        pipeline: str | None = None,
    ) -> None:
        self.failures = failures
        detail = "\n".join(f"  {label}: {error}" for label, error in failures)
        super().__init__(
            f"{message}:\n{detail}" if detail else message, pipeline=pipeline
        )
