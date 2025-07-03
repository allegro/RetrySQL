import pytest

from text2sql.datasets.domain.model import SchemaLink, SchemaLinkingDataSample
from text2sql.evaluation.metrics.schema_linking_metrics import SchemaLinkingMetrics
from text2sql.modules.schema_linking.domain.model import SchemaLinkingOutput


class TestSchemaLinkingMetrics:
    def test_ids_match_and_sorting(self) -> None:
        # GIVEN
        predictions = [
            SchemaLinkingOutput(
                question_id=2,
                question="List all players.",
                schema_links=[SchemaLink(table_name="players", columns=["name"])],
            ),
            SchemaLinkingOutput(
                question_id=1,
                question="What is the number of rares?",
                schema_links=[SchemaLink(table_name="cards", columns=["rarity"])],
            ),
        ]
        ground_truths = [
            SchemaLinkingDataSample(
                question_id=1,
                question="What is the number of rares?",
                schema_links=[SchemaLink(table_name="cards", columns=["rarity"])],
            ),
            SchemaLinkingDataSample(
                question_id=2,
                question="List all players.",
                schema_links=[SchemaLink(table_name="players", columns=["name"])],
            ),
        ]

        # WHEN
        metrics = SchemaLinkingMetrics(predictions, ground_truths)

        # THEN
        assert metrics.predictions[0].question_id == metrics.ground_truths[0].question_id == 1
        assert metrics.predictions[1].question_id == metrics.ground_truths[1].question_id == 2

    @pytest.mark.parametrize(
        "predictions, ground_truths",
        [
            (
                [
                    SchemaLinkingOutput(
                        question_id=1,
                        question="What is the number of rares?",
                        schema_links=[SchemaLink(table_name="cards", columns=["rarity"])],
                    ),
                    SchemaLinkingOutput(
                        question_id=3,
                        question="List all players.",
                        schema_links=[SchemaLink(table_name="players", columns=["name"])],
                    ),
                ],
                [
                    SchemaLinkingOutput(
                        question_id=1,
                        question="What is the number of rares?",
                        schema_links=[SchemaLink(table_name="cards", columns=["rarity"])],
                    ),
                    SchemaLinkingOutput(
                        question_id=2,
                        question="List all players.",
                        schema_links=[SchemaLink(table_name="players", columns=["name"])],
                    ),
                ],
            ),
            (
                [
                    SchemaLinkingDataSample(
                        question_id=1,
                        question="What is the number of rares?",
                        schema_links=[SchemaLink(table_name="cards", columns=["rarity"])],
                    ),
                ],
                [
                    SchemaLinkingDataSample(
                        question_id=1,
                        question="What is the number of rares?",
                        schema_links=[SchemaLink(table_name="cards", columns=["rarity"])],
                    ),
                    SchemaLinkingDataSample(
                        question_id=2,
                        question="List all players.",
                        schema_links=[SchemaLink(table_name="players", columns=["name"])],
                    ),
                ],
            ),
        ],
    )
    def test_consistency_checker(self, predictions, ground_truths) -> None:
        # WHEN & THEN
        with pytest.raises(AssertionError):
            SchemaLinkingMetrics(predictions, ground_truths)

    @pytest.mark.parametrize(
        "predictions, ground_truths, expected_fp_rate",
        [
            (
                [
                    SchemaLinkingOutput(
                        question_id=1,
                        question="What is the number of rares?",
                        schema_links=[SchemaLink(table_name="cards", columns=["rarity"])],
                    ),
                    SchemaLinkingOutput(
                        question_id=2,
                        question="List all players.",
                        schema_links=[SchemaLink(table_name="players", columns=["name"])],
                    ),
                ],
                [
                    SchemaLinkingDataSample(
                        question_id=1,
                        question="What is the number of rares?",
                        schema_links=[SchemaLink(table_name="cards", columns=["rarity"])],
                    ),
                    SchemaLinkingDataSample(
                        question_id=2,
                        question="List all players.",
                        schema_links=[SchemaLink(table_name="players", columns=["name"])],
                    ),
                ],
                0.0,
            ),
            (
                [
                    SchemaLinkingOutput(
                        question_id=1,
                        question="What is the number of rares?",
                        schema_links=[SchemaLink(table_name="cards", columns=["playability"])],
                    ),
                    SchemaLinkingOutput(
                        question_id=2,
                        question="List all players.",
                        schema_links=[SchemaLink(table_name="players", columns=["nationality"])],
                    ),
                ],
                [
                    SchemaLinkingDataSample(
                        question_id=1,
                        question="What is the number of rares?",
                        schema_links=[SchemaLink(table_name="cards", columns=["rarity"])],
                    ),
                    SchemaLinkingDataSample(
                        question_id=2,
                        question="List all players.",
                        schema_links=[SchemaLink(table_name="players", columns=["name"])],
                    ),
                ],
                1.0,
            ),
            (
                [
                    SchemaLinkingOutput(
                        question_id=1,
                        question="What is the number of rares?",
                        schema_links=[SchemaLink(table_name="cards", columns=["rarity"])],
                    ),
                    SchemaLinkingOutput(
                        question_id=2,
                        question="List all players.",
                        schema_links=[SchemaLink(table_name="players", columns=["nationality"])],
                    ),
                ],
                [
                    SchemaLinkingDataSample(
                        question_id=1,
                        question="What is the number of rares?",
                        schema_links=[SchemaLink(table_name="cards", columns=["rarity"])],
                    ),
                    SchemaLinkingDataSample(
                        question_id=2,
                        question="List all players.",
                        schema_links=[SchemaLink(table_name="players", columns=["name"])],
                    ),
                ],
                0.5,
            ),
            (
                [
                    SchemaLinkingOutput(
                        question_id=1,
                        question="What is the number of rares?",
                        schema_links=[SchemaLink(table_name="cards", columns=["rarity", "playability"])],
                    ),
                    SchemaLinkingOutput(
                        question_id=2,
                        question="List all players.",
                        schema_links=[SchemaLink(table_name="players", columns=["name", "nationality"])],
                    ),
                ],
                [
                    SchemaLinkingDataSample(
                        question_id=1,
                        question="What is the number of rares?",
                        schema_links=[SchemaLink(table_name="cards", columns=["rarity"])],
                    ),
                    SchemaLinkingDataSample(
                        question_id=2,
                        question="List all players.",
                        schema_links=[SchemaLink(table_name="players", columns=["name"])],
                    ),
                ],
                0.5,
            ),
            (
                [
                    SchemaLinkingOutput(
                        question_id=1,
                        question="What is the number of rares?",
                        schema_links=[SchemaLink(table_name="cards", columns=["rarity", "playability"])],
                    ),
                    SchemaLinkingOutput(
                        question_id=2,
                        question="List all players.",
                        schema_links=[SchemaLink(table_name="players", columns=["name", "nationality"])],
                    ),
                ],
                [
                    SchemaLinkingDataSample(
                        question_id=1,
                        question="What is the number of rares?",
                        schema_links=[
                            SchemaLink(table_name="cards", columns=["rarity", "playability"]),
                            SchemaLink(table_name="game", columns=["name"]),
                        ],
                    ),
                    SchemaLinkingDataSample(
                        question_id=2,
                        question="List all players.",
                        schema_links=[SchemaLink(table_name="players", columns=["name"])],
                    ),
                ],
                0.25,
            ),
        ],
    )
    def test_false_positive_rate(self, predictions, ground_truths, expected_fp_rate) -> None:
        # GIVEN
        metrics = SchemaLinkingMetrics(predictions, ground_truths)

        # WHEN
        fpr = metrics.false_positive_rate()

        # THEN
        assert fpr == expected_fp_rate

    @pytest.mark.parametrize(
        "predictions, ground_truths, expected_recall",
        [
            (
                [
                    SchemaLinkingOutput(
                        question_id=1,
                        question="What is the number of rares?",
                        schema_links=[SchemaLink(table_name="cards", columns=["rarity"])],
                    ),
                    SchemaLinkingOutput(
                        question_id=2,
                        question="List all players.",
                        schema_links=[SchemaLink(table_name="players", columns=["name"])],
                    ),
                ],
                [
                    SchemaLinkingDataSample(
                        question_id=1,
                        question="What is the number of rares?",
                        schema_links=[SchemaLink(table_name="cards", columns=["rarity"])],
                    ),
                    SchemaLinkingDataSample(
                        question_id=2,
                        question="List all players.",
                        schema_links=[SchemaLink(table_name="players", columns=["name"])],
                    ),
                ],
                1.0,
            ),
            (
                [
                    SchemaLinkingOutput(
                        question_id=1,
                        question="What is the number of rares?",
                        schema_links=[SchemaLink(table_name="cards", columns=["playability"])],
                    ),
                    SchemaLinkingOutput(
                        question_id=2,
                        question="List all players.",
                        schema_links=[SchemaLink(table_name="players", columns=["nationality"])],
                    ),
                ],
                [
                    SchemaLinkingDataSample(
                        question_id=1,
                        question="What is the number of rares?",
                        schema_links=[SchemaLink(table_name="cards", columns=["rarity"])],
                    ),
                    SchemaLinkingDataSample(
                        question_id=2,
                        question="List all players.",
                        schema_links=[SchemaLink(table_name="players", columns=["name"])],
                    ),
                ],
                0.0,
            ),
            (
                [
                    SchemaLinkingOutput(
                        question_id=1,
                        question="What is the number of rares?",
                        schema_links=[SchemaLink(table_name="cards", columns=["rarity"])],
                    ),
                    SchemaLinkingOutput(
                        question_id=2,
                        question="List all players.",
                        schema_links=[SchemaLink(table_name="players", columns=["nationality"])],
                    ),
                ],
                [
                    SchemaLinkingDataSample(
                        question_id=1,
                        question="What is the number of rares?",
                        schema_links=[SchemaLink(table_name="cards", columns=["rarity"])],
                    ),
                    SchemaLinkingDataSample(
                        question_id=2,
                        question="List all players.",
                        schema_links=[SchemaLink(table_name="players", columns=["name"])],
                    ),
                ],
                0.5,
            ),
            (
                [
                    SchemaLinkingOutput(
                        question_id=1,
                        question="What is the number of rares?",
                        schema_links=[SchemaLink(table_name="cards", columns=["rarity", "playability"])],
                    ),
                    SchemaLinkingOutput(
                        question_id=2,
                        question="List all players.",
                        schema_links=[SchemaLink(table_name="players", columns=["name", "nationality"])],
                    ),
                ],
                [
                    SchemaLinkingDataSample(
                        question_id=1,
                        question="What is the number of rares?",
                        schema_links=[SchemaLink(table_name="cards", columns=["rarity"])],
                    ),
                    SchemaLinkingDataSample(
                        question_id=2,
                        question="List all players.",
                        schema_links=[SchemaLink(table_name="players", columns=["name"])],
                    ),
                ],
                1.0,
            ),
            (
                [
                    SchemaLinkingOutput(
                        question_id=1,
                        question="What is the number of rares?",
                        schema_links=[SchemaLink(table_name="cards", columns=["rarity", "playability"])],
                    ),
                    SchemaLinkingOutput(
                        question_id=2,
                        question="List all players.",
                        schema_links=[SchemaLink(table_name="players", columns=["name", "nationality"])],
                    ),
                ],
                [
                    SchemaLinkingDataSample(
                        question_id=1,
                        question="What is the number of rares?",
                        schema_links=[
                            SchemaLink(table_name="cards", columns=["rarity", "playability"]),
                            SchemaLink(table_name="game", columns=["name"]),
                        ],
                    ),
                    SchemaLinkingDataSample(
                        question_id=2,
                        question="List all players.",
                        schema_links=[SchemaLink(table_name="players", columns=["name"])],
                    ),
                ],
                0.5,
            ),
        ],
    )
    def test_schema_linking_recall(self, predictions, ground_truths, expected_recall) -> None:
        # GIVEN
        metrics = SchemaLinkingMetrics(predictions, ground_truths)

        # WHEN
        schema_linking_recall = metrics.schema_linking_recall()

        # THEN
        assert schema_linking_recall == expected_recall

    @pytest.mark.parametrize(
        "predictions, ground_truths, expected_recall",
        [
            (
                [
                    SchemaLinkingOutput(
                        question_id=1,
                        question="What is the number of rares?",
                        schema_links=[SchemaLink(table_name="cards", columns=["rarity"])],
                    ),
                    SchemaLinkingOutput(
                        question_id=2,
                        question="List all players.",
                        schema_links=[SchemaLink(table_name="players", columns=["name"])],
                    ),
                ],
                [
                    SchemaLinkingDataSample(
                        question_id=1,
                        question="What is the number of rares?",
                        schema_links=[SchemaLink(table_name="cards", columns=["rarity"])],
                    ),
                    SchemaLinkingDataSample(
                        question_id=2,
                        question="List all players.",
                        schema_links=[SchemaLink(table_name="players", columns=["name"])],
                    ),
                ],
                1.0,
            ),
            (
                [
                    SchemaLinkingOutput(
                        question_id=1,
                        question="What is the number of rares?",
                        schema_links=[SchemaLink(table_name="cards", columns=["playability"])],
                    ),
                    SchemaLinkingOutput(
                        question_id=2,
                        question="List all players.",
                        schema_links=[SchemaLink(table_name="players", columns=["nationality"])],
                    ),
                ],
                [
                    SchemaLinkingDataSample(
                        question_id=1,
                        question="What is the number of rares?",
                        schema_links=[SchemaLink(table_name="cards", columns=["rarity"])],
                    ),
                    SchemaLinkingDataSample(
                        question_id=2,
                        question="List all players.",
                        schema_links=[SchemaLink(table_name="players", columns=["name"])],
                    ),
                ],
                0.0,
            ),
            (
                [
                    SchemaLinkingOutput(
                        question_id=1,
                        question="What is the number of rares?",
                        schema_links=[SchemaLink(table_name="cards", columns=["rarity"])],
                    ),
                    SchemaLinkingOutput(
                        question_id=2,
                        question="List all players.",
                        schema_links=[SchemaLink(table_name="players", columns=["nationality"])],
                    ),
                ],
                [
                    SchemaLinkingDataSample(
                        question_id=1,
                        question="What is the number of rares?",
                        schema_links=[SchemaLink(table_name="cards", columns=["rarity"])],
                    ),
                    SchemaLinkingDataSample(
                        question_id=2,
                        question="List all players.",
                        schema_links=[SchemaLink(table_name="players", columns=["name"])],
                    ),
                ],
                0.5,
            ),
            (
                [
                    SchemaLinkingOutput(
                        question_id=1,
                        question="What is the number of rares?",
                        schema_links=[SchemaLink(table_name="cards", columns=["rarity", "playability"])],
                    ),
                    SchemaLinkingOutput(
                        question_id=2,
                        question="List all players.",
                        schema_links=[SchemaLink(table_name="players", columns=["name", "nationality"])],
                    ),
                ],
                [
                    SchemaLinkingDataSample(
                        question_id=1,
                        question="What is the number of rares?",
                        schema_links=[SchemaLink(table_name="cards", columns=["rarity"])],
                    ),
                    SchemaLinkingDataSample(
                        question_id=2,
                        question="List all players.",
                        schema_links=[SchemaLink(table_name="players", columns=["name"])],
                    ),
                ],
                1.0,
            ),
            (
                [
                    SchemaLinkingOutput(
                        question_id=1,
                        question="What is the number of rares?",
                        schema_links=[SchemaLink(table_name="cards", columns=["rarity", "playability"])],
                    ),
                    SchemaLinkingOutput(
                        question_id=2,
                        question="List all players.",
                        schema_links=[SchemaLink(table_name="players", columns=["name", "nationality"])],
                    ),
                ],
                [
                    SchemaLinkingDataSample(
                        question_id=1,
                        question="What is the number of rares?",
                        schema_links=[
                            SchemaLink(table_name="cards", columns=["rarity", "playability"]),
                            SchemaLink(table_name="game", columns=["name"]),
                        ],
                    ),
                    SchemaLinkingDataSample(
                        question_id=2,
                        question="List all players.",
                        schema_links=[
                            SchemaLink(
                                table_name="players",
                                columns=["name", "nationality", "popularity"],
                            )
                        ],
                    ),
                ],
                2 / 3,
            ),
        ],
    )
    def test_column_recall(self, predictions, ground_truths, expected_recall) -> None:
        # GIVEN
        metrics = SchemaLinkingMetrics(predictions, ground_truths)

        # WHEN
        recall = metrics.column_recall()

        # THEN
        assert recall == expected_recall
