def test_importable():
    import airoad

    assert hasattr(airoad, "about")
