from app.services.expertise import get_expertise_prompt


def test_student_prompt_contains_guidance():
    prompt = get_expertise_prompt("student")
    assert "student" in prompt.lower()
    assert "step-by-step" in prompt.lower() or "detailed" in prompt.lower()


def test_researcher_prompt_is_concise():
    prompt = get_expertise_prompt("researcher")
    assert "experienced researcher" in prompt.lower() or "concise" in prompt.lower()


def test_faculty_prompt_is_peer():
    prompt = get_expertise_prompt("faculty")
    assert "peer" in prompt.lower() or "senior" in prompt.lower()


def test_unknown_level_returns_empty():
    prompt = get_expertise_prompt("unknown")
    assert prompt == ""


def test_none_returns_empty():
    prompt = get_expertise_prompt(None)
    assert prompt == ""
