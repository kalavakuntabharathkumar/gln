from gradelens.baseline import grade_by_reference

def test_similarity_is_bounded():
    score, confidence = grade_by_reference("Gravity is force between objects with mass.", "Gravity is the attractive force between objects with mass.")
    assert 0 <= score <= 3
    assert 0 <= confidence <= 1

def test_empty_answer():
    assert grade_by_reference("", "A reference answer.") == (0, 0.0)
