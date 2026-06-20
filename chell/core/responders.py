from __future__ import annotations

from abc import ABC, abstractmethod

from chell.core.types import ClarificationQuery, UserResponse


class UserResponder(ABC):
    """Decouple the answer source from the pipeline so eval reuses production code."""

    @abstractmethod
    def answer(self, query: ClarificationQuery) -> UserResponse:
        ...


class InteractiveResponder(UserResponder):
    """Read answers from stdin — used for the live demo."""

    def answer(self, query: ClarificationQuery) -> UserResponse:
        print(f"\n[Chell] {query.question}")
        if query.options:
            for i, opt in enumerate(query.options):
                print(f"  {i + 1}. {opt}")
            raw = input("Your choice (number or free text): ").strip()
            try:
                idx = int(raw) - 1
                if 0 <= idx < len(query.options):
                    return UserResponse(text=query.options[idx], selection=idx)
            except ValueError:
                pass
            return UserResponse(text=raw)
        else:
            raw = input("Your answer: ").strip()
            return UserResponse(text=raw)


class SimulatedResponder(UserResponder):
    """Feed pre-recorded answers — used by the eval harness."""

    def __init__(self, expected_response: str, selection: int | None = None) -> None:
        self._response = expected_response
        self._selection = selection

    def answer(self, query: ClarificationQuery) -> UserResponse:
        return UserResponse(text=self._response, selection=self._selection)
