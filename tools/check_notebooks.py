"""Execute both preview notebooks in a temporary directory, without downloads."""
from pathlib import Path
import tempfile
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    # Python objects still communicate through localhost kernel sockets. This
    # guard forbids the library calls that could fetch recordings or model weights.
    guard = nbformat.v4.new_code_cell('''
import huggingface_hub

def forbidden_download(*args, **kwargs):
    raise AssertionError("Preview attempted a download")
huggingface_hub.hf_hub_download = forbidden_download
huggingface_hub.snapshot_download = forbidden_download
''')
    notebook.cells.insert(0, guard)
    with tempfile.TemporaryDirectory(prefix="simpleb2t-preview-") as work:
        NotebookClient(notebook, timeout=180, kernel_name="python3",
                       resources={"metadata": {"path": work}}).execute()
        assert not (Path(work) / "raw_data").exists()
        assert not (Path(work) / "runs").exists()
    print(f"PASS: {path.name}", flush=True)
