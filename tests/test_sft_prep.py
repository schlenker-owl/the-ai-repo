from airoad.sft.lora_sft import build_alpaca_like_examples, format_example


def test_format_example_has_response_tag():
    ex = {"instruction": "Say hi", "input": "", "output": "Hello!"}
    prompt, target = format_example(ex)
    assert "### Response:" in prompt
    assert target.strip() == "Hello!"


def test_build_examples_nonempty_and_strs():
    exs = build_alpaca_like_examples()
    assert len(exs) >= 5
    p, t = format_example(exs[0])
    assert isinstance(p, str) and isinstance(t, str)
