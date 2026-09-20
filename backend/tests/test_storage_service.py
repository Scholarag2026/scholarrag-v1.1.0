# backend/tests/test_storage_service.py
import os
import pytest
from app.services.storage import StorageService


@pytest.fixture
def storage(tmp_path):
    return StorageService(base_path=str(tmp_path))


def test_save_file(storage, tmp_path):
    content = b"hello,world\n1,2\n3,4"
    path = storage.save("projects/abc/datasets/test.csv", content)
    assert os.path.exists(path)
    assert open(path, "rb").read() == content


def test_read_file(storage):
    content = b"col1,col2\na,b"
    storage.save("test/read.csv", content)
    result = storage.read("test/read.csv")
    assert result == content


def test_delete_file(storage):
    storage.save("test/delete.csv", b"data")
    storage.delete("test/delete.csv")
    assert not os.path.exists(os.path.join(storage.base_path, "test/delete.csv"))


def test_read_nonexistent_raises(storage):
    with pytest.raises(FileNotFoundError):
        storage.read("nonexistent.csv")


def test_file_exists(storage):
    storage.save("test/exists.csv", b"data")
    assert storage.exists("test/exists.csv")
    assert not storage.exists("test/nope.csv")
