from src.clip_ref import PROMPT_TEMPLATE, build_prompts


def test_prompts_follow_the_lab_10_template():
    prompts = build_prompts()
    assert len(prompts) == 10
    assert prompts[0] == 'A photo of a airplane'
    assert PROMPT_TEMPLATE == 'A photo of a {}'
