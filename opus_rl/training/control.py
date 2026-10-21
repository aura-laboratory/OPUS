"""Cooperative controls at training-operation boundaries."""

import threading
import time


class TrainingControl:
    def __init__(self, stop_event=None):
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        self.condition = threading.Condition()
        self.paused = False
        self.waiting = False
        self.permits = 0
        self.stepping = False
        self._paused_seconds = 0.0
        self._waiting_since = None

    @property
    def paused_seconds(self):
        with self.condition:
            return self._paused_seconds + (time.perf_counter() - self._waiting_since
                                           if self._waiting_since is not None else 0)

    @property
    def status(self):
        with self.condition:
            if self.stop_event.is_set():
                return "stopping"
            if not self.paused:
                return "running"
            if self.stepping or self.permits:
                return "stepping"
            return "paused" if self.waiting else "pausing"

    def pause(self):
        with self.condition:
            self.paused = True
            self.permits = 0
            self.stepping = False

    def resume(self):
        with self.condition:
            self.paused = False
            self.permits = 0
            self.stepping = False
            self.condition.notify_all()

    def step(self):
        with self.condition:
            if not self.paused or not self.waiting or self.permits or self.stepping:
                raise ValueError("Pause training and wait for Paused before stepping.")
            self.permits = 1
            self.condition.notify_all()

    def stop(self):
        with self.condition:
            self.stop_event.set()
            self.condition.notify_all()

    def wait(self, changed=None, consume=False):
        """A permit covers pending processing, then exactly one training interaction.

        Evaluation and optimization never consume a step permit. Callbacks run
        outside the condition, avoiding lock inversion with the server state.
        """
        announced = False
        try:
            while True:
                with self.condition:
                    if self.stop_event.is_set():
                        return False
                    if not self.paused:
                        return True
                    if self.permits:
                        if consume:
                            self.permits -= 1
                        self.stepping = True
                        return True
                    self.waiting = True
                    self.stepping = False
                    if self._waiting_since is None:
                        self._waiting_since = time.perf_counter()
                if not announced:
                    announced = True
                    if changed:
                        changed()
                with self.condition:
                    # The timeout also supports callers setting an existing stop Event directly.
                    if self.paused and not self.permits and not self.stop_event.is_set():
                        self.condition.wait(timeout=0.25)
        finally:
            with self.condition:
                self.waiting = False
                if self._waiting_since is not None:
                    self._paused_seconds += time.perf_counter() - self._waiting_since
                    self._waiting_since = None
            if announced and changed:
                changed()
