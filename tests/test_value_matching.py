from bdikit.value_matching.polyfuzz import (
    TFIDFValueMatcher,
    EditDistanceValueMatcher,
    FastTextValueMatcher,
    EmbeddingValueMatcher,
)


def test_textual_transformation_matching():
    threshold = 0.5
    for value_matcher in [
        TFIDFValueMatcher(threshold=threshold),
        EditDistanceValueMatcher(threshold=threshold),
    ]:
    
        # given
        source_values = ["Red Apple", "Banana", "Oorange", "dragon-fruits"]
        target_values = ["apple", "banana", "orange", "kiwi"]

        # when
        matches = value_matcher.match(source_values, target_values)

        # then
        assert len(matches) == 3

        mapped_matches = {match[0]: (match[1], match[2]) for match in matches}
        assert "dragon-fruits" not in mapped_matches
        assert mapped_matches["Red Apple"][0] == "apple"
        assert mapped_matches["Banana"][0] == "banana"
        assert mapped_matches["Oorange"][0] == "orange"

        scores = [match[2] for match in matches]
        assert all(score > threshold for score in scores)


def test_semantic_matching():
    threshold = 0.4
    value_matcher = FastTextValueMatcher(threshold=threshold)

    # given
    source_values = ["Computer", "Display", "Pencil"]
    target_values = ["PC", "Monitor", "Football field"]

    # when
    matches = value_matcher.match(source_values, target_values)

    # then
    assert len(matches) == 2

    mapped_matches = {match[0]: (match[1], match[2]) for match in matches}
    assert "Pencil" not in mapped_matches
    assert mapped_matches["Computer"][0] == "PC"
    assert mapped_matches["Display"][0] == "Monitor"

    scores = [match[2] for match in matches]
    assert all(score > threshold for score in scores)

    threshold = 0.6
    value_matcher = EmbeddingValueMatcher(threshold=threshold)

    # given
    source_values = ["Computer", "Display", "Pencil"]
    target_values = ["PC", "Monitor", "Football field"]

    # when
    matches = value_matcher.match(source_values, target_values)

    # then
    assert len(matches) == 2

    mapped_matches = {match[0]: (match[1], match[2]) for match in matches}
    assert "Pencil" not in mapped_matches
    assert mapped_matches["Computer"][0] == "PC"
    assert mapped_matches["Display"][0] == "Monitor"

    scores = [match[2] for match in matches]
    assert all(score > threshold for score in scores)