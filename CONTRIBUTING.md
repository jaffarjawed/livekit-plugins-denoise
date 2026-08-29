# Contributing

Thanks for helping improve `livekit-plugins-denoise`.

1. Open an issue before starting a substantial feature so we can agree on the
   API and real-time audio constraints.
2. Keep pull requests focused and add or update tests for behavioural changes.
3. Run the package checks before opening a pull request:

   ```bash
   python -m pip install -e ".[dev]"
   python -m pytest -q
   python -m build
   python -m twine check dist/*
   ```

Audio regressions are especially valuable to report. If possible, include the
sample rate, channel count, frame duration, enhancer, and whether echo
reference audio is attached. Do not include recordings containing private or
identifiable speech.
