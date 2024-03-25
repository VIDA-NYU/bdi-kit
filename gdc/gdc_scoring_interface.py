import jellyfish


class GDCScoringInterface:
    @staticmethod
    def compute_col_values_score(values, choices):
        """
        This computes the score between the input values and the candidate gdc values.

        :param values: list, the input enum values from actual data column
        :param choices: list, the enum values from GDC schema

        :return: score: float, a score between 0 and 1
        """
        pass

    @staticmethod
    def compute_col_name_score(col_name, candidate_col_name):
        """
        This computes the score between the input column name and the candidate gdc column name.

        :param col_name: str, the input column name
        :param candidate_col_name: str, the candidate gdc column name

        :return: score: float, a score between 0 and 1
        """
        pass


# this is a sample implementation of the GDCScoringInterface using Jaro similarity
class JaroScore(GDCScoringInterface):
    scorer_name = "jaro"  # must have a scorer_name attribute, and should be unique!

    @staticmethod
    def compute_col_values_score(values, choices):
        """
        Compute the Jaro similarity score between the input values and the choices.
        For each value, it will find the maximum Jaro similarity score with the choices
        and then return the average score.

        :param values: list, the input values
        :param choices: list, the choices to compare with from GDC enums
        :return: score: float, the average Jaro similarity score
        """
        score = 0
        for value in values:
            score += max(
                [jellyfish.jaro_similarity(value, choice) for choice in choices]
            )
        return score / len(values)

    @staticmethod
    def compute_col_name_score(col_name, candidate_col_name):
        return jellyfish.jaro_similarity(col_name, candidate_col_name)
