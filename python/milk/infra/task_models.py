from __future__ import annotations

import typing as typ

import os

import abc

"""
pipe = pipe(taskA)(taskB)(taskC)(taskD) could be a nice syntax.
"""

if typ.TYPE_CHECKING:
    from ..pipeline import Pipeline


class SimpleTask(abc.ABC):  # This can be a great wrapper for AORUN

    pipeline: Pipeline

    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline

    @abc.abstractmethod
    def can(self) -> bool:
        """
        can as in "can perform the forward operation"
        """
        ...

    @abc.abstractmethod
    def success(self) -> bool:
        """
        Did the forward task succeed
        """
        ...

    @abc.abstractmethod
    def forward(self) -> None:  # Raises TaskException?
        """
        Perform the forward task
        Raise if not can before
        Raise if not success after
        """
        ...


class ReversibleTask(SimpleTask):

    @abc.abstractmethod
    def reverse(self):
        """
        Perform the opposite of the task
        Raise if not success before
        Raise if not can after
        """
        ...

    @abc.abstractmethod
    def best_effort_reverse(self):
        """
        Perform the opposite of the task
        The state may be mingled, dont raise.
        """
        ...
