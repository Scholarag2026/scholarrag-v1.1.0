from app.services.writing import (
    LANGUAGE_NAMES,
    build_language_instruction,
    get_section_prompt,
)


def test_english_has_no_language_instruction():
    """English (default) should NOT append any language instruction."""
    prompt = get_section_prompt("literature_review", language="en")
    assert "Write the entire output in" not in prompt


def test_non_english_includes_language_instruction():
    """Non-English languages should include a language instruction in the prompt."""
    prompt = get_section_prompt("literature_review", language="zh")
    assert "Write the entire output in Chinese (Simplified)" in prompt
    assert "academic conventions" in prompt


def test_build_language_instruction_returns_empty_for_english():
    assert build_language_instruction("en") == ""


def test_build_language_instruction_for_all_supported_languages():
    """Every supported non-English language code should produce a valid instruction."""
    non_english = {k: v for k, v in LANGUAGE_NAMES.items() if k != "en"}
    for code, name in non_english.items():
        instruction = build_language_instruction(code)
        assert name in instruction, f"Expected '{name}' in instruction for code '{code}'"
        assert "IMPORTANT" in instruction


def test_all_language_codes_map_to_names():
    """All 8 supported language codes should be present in LANGUAGE_NAMES."""
    expected_codes = {"en", "zh", "ja", "ko", "de", "fr", "es", "pt"}
    assert expected_codes == set(LANGUAGE_NAMES.keys())


def test_unknown_language_falls_back_to_english_name():
    """An unrecognised language code should fall back to 'English' as the name."""
    instruction = build_language_instruction("xx")
    assert "English" in instruction
