import os
import time


class Timer:

    @staticmethod
    def _zero():
        return 0

    def __init__(self, clock=time.time, enabled=True):
        self.start = 0
        self.stop = 0
        self._time = clock if enabled else Timer._zero
        self._tick = 0

    def __enter__(self):
        self.start = self._tick = self._time()
        return self

    def __exit__(self, *args):
        self.stop = self._tick = self._time()

    @property
    def tick(self):
        if self.stop > 0:
            return -1
        now = self._time()
        tick = now - self._tick
        self._tick = now
        return tick

    @property
    def duration(self):
        if self.stop > 0:
            return self.stop - self.start
        return self._time() - self.start


def install_lightautoml():
    os.system("python ./scripts/poetry_fix.py -c")
    os.system("""
        pip install .
    """
    )

        # poetry config virtualenvs.create false --local
        # poetry run python ./scripts/poetry_fix.py -c
        # ls -la
        # poetry run pip install pillow==9.2.0
        # poetry install
        # poetry run pip freeze
        # poetry run python -c "import sys; print(sys.path)"