"""Utils for running experiments."""

import os
import time


class Timer:  # noqa: D101
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
        """Make one tick."""
        if self.stop > 0:
            return -1
        now = self._time()
        tick = now - self._tick
        self._tick = now
        return tick

    @property
    def duration(self):
        """Get dureation in seconds."""
        if self.stop > 0:
            return self.stop - self.start
        return self._time() - self.start


def install_lightautoml():
    """Install lightautoml using pip."""
    # os.system("pwd")
    # print("SYSTEM FREEZED")
    # time.sleep(1800)
    os.system("curl -sSL https://install.python-poetry.org | ../../bin/python - --version 1.5.1")
    # print("python unfreezed")
    os.system("/root/.local/bin/poetry build")
    os.system("../../bin/pip install dist/lightautoml-0.3.8b1-py3-none-any.whl")
    # print("SUCCESS")


#        .pip install --upgrade pip
# poetry config virtualenvs.create false --local
# poetry run python ./scripts/poetry_fix.py -c
# ls -la
# poetry run pip install pillow==9.2.0
# poetry install
# poetry run pip freeze
# poetry run python -c "import sys; print(sys.path)"
