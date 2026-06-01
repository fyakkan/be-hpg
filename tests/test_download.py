"""Phase 1 gate: download URL builder + tar extraction (no network)."""
import tarfile

from src.data.download import task_url, extract_tar, DEFAULT_S3_BASE


def test_task_url():
    assert task_url("Task09_Spleen") == f"{DEFAULT_S3_BASE}/Task09_Spleen.tar"
    assert task_url("Task07_Pancreas", "https://x/") == "https://x/Task07_Pancreas.tar"


def test_extract_tar(tmp_path):
    inner = tmp_path / "Task00_Toy"
    inner.mkdir()
    (inner / "hello.txt").write_text("hi")
    tar_path = tmp_path / "Task00_Toy.tar"
    with tarfile.open(tar_path, "w") as t:
        t.add(str(inner), arcname="Task00_Toy")

    out = tmp_path / "raw"
    extract_tar(str(tar_path), str(out))
    assert (out / "Task00_Toy" / "hello.txt").read_text() == "hi"
