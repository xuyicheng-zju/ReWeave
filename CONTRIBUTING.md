# Contributing

Use a separate branch and install the test dependencies.

```bash
python -m pip install -e '.[test,graspgen]'
python -m pytest
```

Include a minimal test case for behavioral changes and describe the simulator,
model and asset revisions for benchmark reports. Keep credentials, native
session files, checkpoints and run outputs outside commits. When adding
upstream source, retain its license and attribution in [NOTICE](NOTICE).
