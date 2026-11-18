# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""Exercise RoboDojo's namespace binding across an Isaac/OpenCV path change."""

import subprocess
import sys

import pytest

from reweave.benchmarks.robodojo.env_server import prepare_source_imports


@pytest.mark.parametrize("foreign_loaded_first", [False, True])
def test_source_binding_survives_later_regular_utils_package(
    tmp_path, foreign_loaded_first
):
    source = tmp_path / "robodojo"
    foreign = tmp_path / "opencv"
    for root in (source, foreign):
        (root / "utils").mkdir(parents=True)
        (root / "utils/load_file.py").write_text(f"ORIGIN = {str(root)!r}\n")
    (foreign / "utils/__init__.py").write_text("")
    script = f"""
import importlib
import sys
from reweave.benchmarks.robodojo.env_server import prepare_source_imports
source, foreign = {str(source)!r}, {str(foreign)!r}
if {foreign_loaded_first!r}:
    sys.path.insert(0, foreign)
    original = importlib.import_module('utils.load_file')
    try:
        prepare_source_imports(source)
    except ImportError as error:
        assert 'outside its source' in str(error)
    else:
        raise AssertionError('accepted a foreign utils module')
    assert sys.modules['utils.load_file'] is original
else:
    assert prepare_source_imports(source) == source
    sys.path.insert(0, foreign)
    assert importlib.import_module('utils.load_file').ORIGIN == source
assert not any(name.startswith('isaacsim') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_missing_source_fails_before_simulator_start(tmp_path):
    with pytest.raises(FileNotFoundError, match="RoboDojo source is missing"):
        prepare_source_imports(str(tmp_path))
