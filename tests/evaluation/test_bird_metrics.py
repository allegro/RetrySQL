import shutil

import pytest
from upath import UPath

from tests.helpers.builders import DatabaseBuilder
from tests.helpers.domain import GroundTruthTestCase
from text2sql.evaluation.domain.enums import QuestionDifficulty
from text2sql.evaluation.domain.model import EvaluationResult, GenerationResult, GroundTruth
from text2sql.evaluation.metrics.bird_metrics import BirdMetrics


class TestBirdMetrics:
    def setup_method(self):
        (
            DatabaseBuilder("people_db")
            .with_table(
                table_name="person",
                column_names=["name", "profession"],
                column_types=["char(255)", "char(255)"],
            )
            .with_values(
                table_name="person",
                columns=["name", "profession"],
                values=["John Smith", "Plumber"],
            )
            .with_values(
                table_name="person",
                columns=["name", "profession"],
                values=["Mary Potter", "Nurse"],
            )
            .with_values(
                table_name="person",
                columns=["name", "profession"],
                values=["Anna Wilson", "Nurse"],
            )
            .build(storage_directory=UPath("./test_tmp/dev_databases/people_db"))
        )

        (
            DatabaseBuilder("animals_db")
            .with_table(
                table_name="animal",
                column_names=["name", "size"],
                column_types=["char(255)", "int8"],
            )
            .with_values(table_name="animal", columns=["name", "size"], values=["Cat", 10])
            .with_values(table_name="animal", columns=["name", "size"], values=["Dog", 15])
            .with_values(table_name="animal", columns=["name", "size"], values=["Hamster", 2])
            .build(storage_directory=UPath("./test_tmp/dev_databases/animals_db"))
        )

    def teardown_method(self):
        if UPath("./test_tmp").exists():
            shutil.rmtree("./test_tmp")

    @pytest.mark.parametrize(
        ("generated_queries", "ground_truth_test_cases", "expected_accuracy"),
        [
            (
                [
                    "select * from person",
                    'select * from person where profession = "Nurse"',
                    'select * from person where profession = "Nurse" and profession != "Plumber"'
                ],
                [
                    GroundTruthTestCase(
                        database_name="people_db",
                        sql_query="select * from person",
                        question_difficulty=QuestionDifficulty.EASY
                    ),
                    GroundTruthTestCase(
                        database_name="people_db",
                        question_difficulty=QuestionDifficulty.MEDIUM,
                        sql_query='select * from person where profession = "Nurse"',
                    ),
                    GroundTruthTestCase(
                        database_name="people_db",
                        question_difficulty=QuestionDifficulty.HARD,
                        sql_query='select * from person where profession = "Nurse" and profession != "Plumber"',
                    ),
                ],
                EvaluationResult(
                    easy_accuracy=100.0,
                    medium_accuracy=100.0,
                    hard_accuracy=100.0,
                    overall_accuracy=100.0
                )
            ),
            (
                [
                    "select * from person",
                    'select * from person where profession = "Plumber"',
                    'select * from person where profession = "Nurse"',
                    'select * from person where profession = "Nurse" and profession != "Plumber"'
                ],
                [
                    GroundTruthTestCase(
                        database_name="people_db", 
                        question_difficulty=QuestionDifficulty.EASY,
                        sql_query="select * from person"
                    ),
                    GroundTruthTestCase(
                        database_name="people_db",
                        question_difficulty=QuestionDifficulty.MEDIUM,
                        sql_query='select * from person where profession = "Nurse"',
                    ),
                    GroundTruthTestCase(
                        database_name="people_db",
                        question_difficulty=QuestionDifficulty.MEDIUM,
                        sql_query='select * from person where profession = "Plumber"',
                    ),
                    GroundTruthTestCase(
                        database_name="people_db",
                        question_difficulty=QuestionDifficulty.HARD,
                        sql_query='select * from person where profession = "Nurse" and profession != "Plumber"',
                    ),
                ],
                EvaluationResult(
                    easy_accuracy=100.0,
                    medium_accuracy=0.0,
                    hard_accuracy=100.0,
                    overall_accuracy=50.0
                )
            ),
            (
                [
                    "select * from person", 
                    "select * from animal where size = 10",
                    "select * from animal where size >= 10 and size < 20"
                ],
                [
                    GroundTruthTestCase(
                        database_name="people_db", 
                        sql_query="select * from person",
                        question_difficulty=QuestionDifficulty.EASY
                    ),
                    GroundTruthTestCase(
                        database_name="animals_db", 
                        sql_query="select * from animal where size = 10",
                        question_difficulty=QuestionDifficulty.MEDIUM
                    ),
                    GroundTruthTestCase(
                        database_name="animals_db", 
                        sql_query="select * from animal where size >= 10 and size < 20",
                        question_difficulty=QuestionDifficulty.HARD
                    )
                ],
                EvaluationResult(
                    easy_accuracy=100.0,
                    medium_accuracy=100.0,
                    hard_accuracy=100.0,
                    overall_accuracy=100.0
                )
            ),
            (
                [
                    "select * from person", 
                    "select * from animal",
                    "select * from animal where size >= 10 and size < 20",
                    "select * from animal"
                ],
                [
                    GroundTruthTestCase(
                        database_name="people_db", 
                        sql_query="select * from person",
                        question_difficulty=QuestionDifficulty.EASY
                    ),
                    GroundTruthTestCase(
                        database_name="animals_db", 
                        sql_query="select * from animal where size < 10",
                        question_difficulty=QuestionDifficulty.MEDIUM
                    ),
                    GroundTruthTestCase(
                        database_name="animals_db", 
                        sql_query="select * from animal where size >= 10 and size < 20",
                        question_difficulty=QuestionDifficulty.HARD
                    ),
                    GroundTruthTestCase(
                        database_name="animals_db", 
                        sql_query="select * from animal where size > 0 and size < 10",
                        question_difficulty=QuestionDifficulty.HARD
                    )
                ],
                EvaluationResult(
                    easy_accuracy=100.0,
                    medium_accuracy=0.0,
                    hard_accuracy=50.0,
                    overall_accuracy=50.0
                )
            ),
        ],
        ids=[
            "single_db_same_results",
            "single_db_different_results",
            "multiple_db_same_result",
            "multiple_db_different_result",
        ],
    )
    def test_execution_accuracy_calculation(
        self,
        generated_queries: list[str],
        ground_truth_test_cases: list[GroundTruthTestCase],
        expected_accuracy: float,
    ):
        # GIVEN
        generation_results = [
            GenerationResult(question_id=index, sql_query=query) for index, query in enumerate(generated_queries)
        ]
        ground_truth = [
            GroundTruth(
                question_id=index,
                database_name=ground_truth_test_case.database_name,
                question_difficulty=ground_truth_test_case.question_difficulty,
                sql_query=ground_truth_test_case.sql_query,
            )
            for index, ground_truth_test_case in enumerate(ground_truth_test_cases)
        ]

        bird_metrics = BirdMetrics(
            base_path=UPath("./test_tmp"),
            cpu_count=1,
            timeout_in_seconds=5,
        )

        # WHEN
        execution_accuracy = bird_metrics.calculate_execution_accuracy(
            results=generation_results, ground_truth=ground_truth
        )

        # THEN
        assert execution_accuracy == expected_accuracy

    @pytest.mark.parametrize(
        ("generated_queries", "ground_truth_test_cases", "perfect_efficiency_score"),
        [
            (
                [
                    "select a.* from person as a inner join person as b on a.name = b.name",
                    'select * from person where profession = "Nurse"',
                ],
                [
                    GroundTruthTestCase(
                        database_name="people_db", 
                        sql_query="select * from person",
                        question_difficulty=QuestionDifficulty.EASY
                    ),
                    GroundTruthTestCase(
                        database_name="people_db",
                        sql_query='select * from person where profession = "Nurse"',
                        question_difficulty=QuestionDifficulty.MEDIUM
                    ),
                ],
                100.0,
            ),
            (
                [
                    "select a.* from person as a inner join person as b on a.name = b.name",
                    'select * from person where profession = "Plumber"',
                ],
                [
                    GroundTruthTestCase(
                        database_name="people_db", 
                        sql_query="select * from person",
                        question_difficulty=QuestionDifficulty.EASY
                    ),
                    GroundTruthTestCase(
                        database_name="people_db",
                        sql_query='select * from person where profession = "Nurse"',
                        question_difficulty=QuestionDifficulty.MEDIUM
                    ),
                ],
                50.0,
            ),
            (
                [
                    "select a.* from person as a inner join person as b on a.name = b.name",
                    "select * from animal",
                ],
                [
                    GroundTruthTestCase(
                        database_name="people_db", 
                        sql_query="select * from person",
                        question_difficulty=QuestionDifficulty.EASY
                    ),
                    GroundTruthTestCase(
                        database_name="animals_db", 
                        sql_query="select * from animal",
                        question_difficulty=QuestionDifficulty.EASY
                    ),
                ],
                100.0,
            ),
            (
                [
                    "select a.* from person as a inner join person as b on a.name = b.name",
                    "select * from animal where size < 10",
                ],
                [
                    GroundTruthTestCase(
                        database_name="people_db", 
                        sql_query="select * from person",
                        question_difficulty=QuestionDifficulty.EASY
                    ),
                    GroundTruthTestCase(
                        database_name="animals_db", 
                        sql_query="select * from animal",
                        question_difficulty=QuestionDifficulty.EASY
                    ),
                ],
                50.0,
            ),
        ],
        ids=[
            "single_db_same_results",
            "single_db_different_results",
            "multiple_db_same_result",
            "multiple_db_different_result",
        ],
    )
    def test_valid_efficiency_score_calculation(
        self,
        generated_queries: list[str],
        ground_truth_test_cases: list[GroundTruthTestCase],
        perfect_efficiency_score: float,
    ):
        # GIVEN
        generation_results = [
            GenerationResult(question_id=index, sql_query=query) for index, query in enumerate(generated_queries)
        ]
        ground_truth = [
            GroundTruth(
                question_id=index,
                database_name=ground_truth_test_case.database_name,
                question_difficulty=ground_truth_test_case.question_difficulty,
                sql_query=ground_truth_test_case.sql_query,
            )
            for index, ground_truth_test_case in enumerate(ground_truth_test_cases)
        ]

        bird_metrics = BirdMetrics(
            base_path=UPath("./test_tmp"),
            cpu_count=1,
            timeout_in_seconds=5,
        )

        # WHEN
        execution_accuracy = bird_metrics.calculate_valid_efficiency_score(
            results=generation_results, ground_truth=ground_truth
        )

        # THEN
        assert execution_accuracy < perfect_efficiency_score
