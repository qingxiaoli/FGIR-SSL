from src.data.dataset import FGIRDataset

def test_dataset_length(tmp_path):
    # Create a temporary directory with fake image data
    d = tmp_path / "images"
    d.mkdir()
    (d / "img1.jpg").write_bytes(b"fake image data")
    (d / "img2.jpg").write_bytes(b"fake image data")
    dataset = FGIRDataset(str(d))
    assert len(dataset) == 2
