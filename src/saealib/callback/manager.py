"""CallbackManager: manages event handler registration and dispatch."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import TypeVar

from saealib.callback.events import Event

E = TypeVar("E", bound=Event)


class CallbackManager:
    """
    Manages event handlers.

    Handlers are registered per concrete event type and called in
    registration order when an event of that type is dispatched.

    Attributes
    ----------
    handlers : defaultdict[type[Event], list[Callable]]
        Mapping from event class to list of registered handler functions.
    """

    def __init__(self) -> None:
        """Initialize CallbackManager."""
        self.handlers: defaultdict[type[Event], list] = defaultdict(list)

    def register(self, event_type: type[E], func: Callable[[E], None]) -> None:
        """
        Register a handler for an event type.

        Parameters
        ----------
        event_type : type[E]
            The concrete event class to listen for.
        func : Callable[[E], None]
            Handler function. Receives the event object and returns nothing.

        Returns
        -------
        None
        """
        self.handlers[event_type].append(func)

    def dispatch(self, event: Event) -> None:
        """
        Invoke all handlers registered for the type of *event*.

        Parameters
        ----------
        event : Event
            The event object to dispatch. Handlers receive this object
            directly and may modify its mutable fields.

        Returns
        -------
        None
        """
        for handler in self.handlers[type(event)]:
            handler(event)

    def unregister(self, event_type: type[E], func: Callable[[E], None]) -> None:
        """
        Remove a previously registered handler.

        Parameters
        ----------
        event_type : type[E]
            The event class the handler was registered for.
        func : Callable[[E], None]
            The handler to remove. Raises ``ValueError`` if not found.

        Returns
        -------
        None
        """
        self.handlers[event_type].remove(func)

    def replace(
        self,
        event_type: type[E],
        old: Callable[[E], None],
        new: Callable[[E], None],
    ) -> None:
        """
        Replace a registered handler with another.

        Parameters
        ----------
        event_type : type[E]
            The event class whose handler list to modify.
        old : Callable[[E], None]
            The handler to replace. Raises ``ValueError`` if not found.
        new : Callable[[E], None]
            The replacement handler.

        Returns
        -------
        None
        """
        idx = self.handlers[event_type].index(old)
        self.handlers[event_type][idx] = new
