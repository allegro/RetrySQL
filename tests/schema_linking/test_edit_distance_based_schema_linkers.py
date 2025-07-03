import pytest

from text2sql.datasets.domain.model import DatabaseSchema, SchemaColumn, SchemaLink
from text2sql.modules.schema_linking.domain.model import SchemaLinkerHeuristicBasedAlgorithmInputExample, \
    SchemaLinkingOutput
from text2sql.modules.schema_linking.schema_linkers.heuristic_based.schema_linker_edit_distance import (
    SchemaLinkerEditDistance,
)


@pytest.mark.parametrize(
    "input_example, expected_output",
    [
        (
            SchemaLinkerHeuristicBasedAlgorithmInputExample(
                question_id=1,
                question="What is the free meal rate for 2020?",
                database_schema=DatabaseSchema(
                    database_name="school_db",
                    tables={
                        "frpm": [SchemaColumn(column_name="academic_year"), SchemaColumn(column_name="free_meal_rate")]
                    },
                ),
            ),
            SchemaLinkingOutput(
                question_id=1,
                question="What is the free meal rate for 2020?",
                schema_links=[SchemaLink(table_name="frpm", columns=["free_meal_rate"])],
            ),
        ),
        (
            SchemaLinkerHeuristicBasedAlgorithmInputExample(
                question_id=2,
                question="What are the details for 2020 FRPM?",
                database_schema=DatabaseSchema(
                    database_name="school_db",
                    tables={
                        "frpm": [SchemaColumn(column_name="academic_year"), SchemaColumn(column_name="free_meal_rate")]
                    },
                ),
            ),
            SchemaLinkingOutput(
                question_id=2,
                question="What are the details for 2020 FRPM?",
                schema_links=[SchemaLink(table_name="frpm", columns=["academic_year", "free_meal_rate"])],
            ),
        ),
        (
            SchemaLinkerHeuristicBasedAlgorithmInputExample(
                question_id=3,
                question="What is the academic year?",
                database_schema=DatabaseSchema(
                    database_name="school_db",
                    tables={
                        "scores": [SchemaColumn(column_name="academic_year"), SchemaColumn(column_name="average_score")]
                    },
                ),
            ),
            SchemaLinkingOutput(
                question_id=3,
                question="What is the academic year?",
                schema_links=[SchemaLink(table_name="scores", columns=["academic_year"])],
            ),
        ),
        (
            SchemaLinkerHeuristicBasedAlgorithmInputExample(
                question_id=4,
                question="What is the graduation rate?",
                database_schema=DatabaseSchema(
                    database_name="school_db",
                    tables={"scores": [SchemaColumn(column_name="average_score"), SchemaColumn(column_name="years")]},
                ),
            ),
            SchemaLinkingOutput(
                question_id=4,
                question="What is the graduation rate?",
                schema_links=[SchemaLink(table_name="", columns=[])],
            ),
        ),
        (
            SchemaLinkerHeuristicBasedAlgorithmInputExample(
                question_id=5,
                question="What is the free meal rate in the school?",
                database_schema=DatabaseSchema(
                    database_name="school_db",
                    tables={
                        "frpm": [SchemaColumn(column_name="free_meal_rate"), SchemaColumn(column_name="academic_year")]
                    },
                ),
            ),
            SchemaLinkingOutput(
                question_id=5,
                question="What is the free meal rate in the school?",
                schema_links=[SchemaLink(table_name="frpm", columns=["free_meal_rate"])],
            ),
        ),
        (
            SchemaLinkerHeuristicBasedAlgorithmInputExample(
                question_id=6,
                question="What is the FREE MEAL RATE?",
                database_schema=DatabaseSchema(
                    database_name="school_db",
                    tables={
                        "frpm": [SchemaColumn(column_name="free_meal_rate"), SchemaColumn(column_name="academic_year")]
                    },
                ),
            ),
            SchemaLinkingOutput(
                question_id=6,
                question="What is the FREE MEAL RATE?",
                schema_links=[SchemaLink(table_name="frpm", columns=["free_meal_rate"])],
            ),
        ),
        (
            SchemaLinkerHeuristicBasedAlgorithmInputExample(
                question_id=7,
                question="What is the meal rate?",
                external_knowledge="free meals information",
                database_schema=DatabaseSchema(
                    database_name="school_db",
                    tables={
                        "frpm": [SchemaColumn(column_name="free_meal_rate"), SchemaColumn(column_name="academic_year")]
                    },
                ),
            ),
            SchemaLinkingOutput(
                question_id=7,
                question="What is the meal rate?",
                schema_links=[SchemaLink(table_name="frpm", columns=["free_meal_rate"])],
            ),
        ),
    ],
    ids=[
        "Exact table and column match",
        "Table match but no column match",
        "Column match but no table match",
        "No match at all",
        "Stop words ignored",
        "Case insensitivity",
        "External knowledge used",
    ],
)
def test_schema_linker_exact_matching_on_different_corner_cases(input_example, expected_output):
    # GIVEN
    schema_linker = SchemaLinkerEditDistance(
        use_external_knowledge=bool(input_example.external_knowledge),
        edit_distance=0,
    )

    # WHEN
    predictions = schema_linker.forward([input_example])

    # THEN
    assert predictions[0] == expected_output


@pytest.mark.parametrize(
    "edit_distance, input_example, expected_output",
    [
        (
            0,
            SchemaLinkerHeuristicBasedAlgorithmInputExample(
                question_id=1,
                question="What is the best academyx?",
                database_schema=DatabaseSchema(
                    database_name="school_db",
                    tables={"scores": [SchemaColumn(column_name="academy"), SchemaColumn(column_name="average_score")]},
                ),
            ),
            SchemaLinkingOutput(
                question_id=1,
                question="What is the best academyx?",
                schema_links=[SchemaLink(table_name="", columns=[])],
            ),
        ),
        (
            1,
            SchemaLinkerHeuristicBasedAlgorithmInputExample(
                question_id=2,
                question="What is the best academyx?",
                database_schema=DatabaseSchema(
                    database_name="school_db",
                    tables={"scores": [SchemaColumn(column_name="academy"), SchemaColumn(column_name="average_score")]},
                ),
            ),
            SchemaLinkingOutput(
                question_id=2,
                question="What is the best academyx?",
                schema_links=[SchemaLink(table_name="scores", columns=["academy"])],
            ),
        ),
        (
            2,
            SchemaLinkerHeuristicBasedAlgorithmInputExample(
                question_id=3,
                question="What is the best academmc?",
                database_schema=DatabaseSchema(
                    database_name="school_db",
                    tables={"scores": [SchemaColumn(column_name="academy"), SchemaColumn(column_name="average_score")]},
                ),
            ),
            SchemaLinkingOutput(
                question_id=3,
                question="What is the best academmc?",
                schema_links=[SchemaLink(table_name="scores", columns=["academy"])],
            ),
        ),
        (
            3,
            SchemaLinkerHeuristicBasedAlgorithmInputExample(
                question_id=4,
                question="What is the beest academmcc?",
                database_schema=DatabaseSchema(
                    database_name="school_db",
                    tables={"scores": [SchemaColumn(column_name="academy"), SchemaColumn(column_name="average_score")]},
                ),
            ),
            SchemaLinkingOutput(
                question_id=4,
                question="What is the beest academmcc?",
                schema_links=[SchemaLink(table_name="scores", columns=["academy"])],
            ),
        ),
    ],
)
def test_schema_linker_edit_distance_varying(edit_distance, input_example, expected_output):
    # GIVEN
    schema_linker = SchemaLinkerEditDistance(
        use_external_knowledge=bool(input_example.external_knowledge),
        edit_distance=edit_distance,
    )

    # WHEN
    predictions = schema_linker.forward([input_example])

    # THEN
    assert predictions[0] == expected_output
