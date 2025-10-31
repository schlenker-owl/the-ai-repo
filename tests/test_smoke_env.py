def test_env_smoke():
    import torch
    assert isinstance(torch.__version__, str)
    assert hasattr(torch.backends, "mps")  # MPS present on Apple Silicon builds
