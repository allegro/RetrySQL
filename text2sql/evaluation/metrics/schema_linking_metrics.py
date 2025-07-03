from text2sql.commons.logging_utils import get_logger
from text2sql.datasets.domain.model import SchemaLink, SchemaLinkingDataSample
from text2sql.modules.schema_linking.domain.model import SchemaLinkingOutput

logger = get_logger(__name__)


class SchemaLinkingMetrics:
    def __init__(
        self,
        predictions: list[SchemaLinkingOutput],
        ground_truths: list[SchemaLinkingDataSample],
    ) -> None:
        self.predictions = predictions
        self.ground_truths = ground_truths

        self._sort_predictions_and_ground_truths_by_question_id()
        self._run_consistency_check()

    def _run_consistency_check(self) -> None:
        if len(self.predictions) != len(self.ground_truths):
            raise AssertionError("Something went wrong - predictions and ground truths have different lengths.")

        for prediction, ground_truth in zip(self.predictions, self.ground_truths):
            if prediction.question_id != ground_truth.question_id:
                raise AssertionError(
                    f"Mismatch in question IDs between predictions and ground truths at index"
                    f" {self.predictions.index(prediction)}.\n"
                )

    def _sort_predictions_and_ground_truths_by_question_id(self) -> None:
        self.predictions = sorted(self.predictions, key=lambda x: x.question_id)
        self.ground_truths = sorted(self.ground_truths, key=lambda x: x.question_id)

    @staticmethod
    def _unroll_table_and_column_schema_links(
        schema_links: list[SchemaLink],
    ) -> set[tuple[str, str]]:
        return {(schema_link.table_name, column) for schema_link in schema_links for column in schema_link.columns}

    def false_positive_rate(self) -> float:
        """
        False Positive Rate (FPR):
        The proportion of irrelevant columns retrieved over the total number of retrieved columns.
        source: https://arxiv.org/abs/2408.07702
        """
        total_false_positives = 0
        total_retrieved_columns = 0

        for prediction, ground_truth in zip(self.predictions, self.ground_truths):
            pred_schema_links = SchemaLinkingMetrics._unroll_table_and_column_schema_links(prediction.schema_links)
            true_schema_links = SchemaLinkingMetrics._unroll_table_and_column_schema_links(ground_truth.schema_links)

            false_positives = pred_schema_links - true_schema_links
            total_false_positives += len(false_positives)
            total_retrieved_columns += len(pred_schema_links)

        return total_false_positives / total_retrieved_columns if total_retrieved_columns > 0 else 0

    def schema_linking_recall(self) -> float:
        """
        Schema Linking Recall (SLR):
        The proportion of queries for which all required columns are retrieved.
        source: https://arxiv.org/abs/2408.07702
        """
        total_queries = len(self.ground_truths)
        total_correct_queries = 0

        for prediction, ground_truth in zip(self.predictions, self.ground_truths):
            pred_schema_links = SchemaLinkingMetrics._unroll_table_and_column_schema_links(prediction.schema_links)
            true_schema_links = SchemaLinkingMetrics._unroll_table_and_column_schema_links(ground_truth.schema_links)

            if true_schema_links.issubset(pred_schema_links):
                total_correct_queries += 1

        return total_correct_queries / total_queries if total_queries > 0 else 0

    def column_recall(self) -> float:
        """
        Column Recall (CR):
        The fraction between actually retrieved columns over the total (ground truth) columns,
        averaged over all the test examples.
        """
        total_queries = len(self.ground_truths)
        recalls = []

        for prediction, ground_truth in zip(self.predictions, self.ground_truths):
            pred_schema_links = SchemaLinkingMetrics._unroll_table_and_column_schema_links(prediction.schema_links)
            true_schema_links = SchemaLinkingMetrics._unroll_table_and_column_schema_links(ground_truth.schema_links)

            correctly_retrieved_schema_links = true_schema_links.intersection(pred_schema_links)
            recalls.append(len(correctly_retrieved_schema_links) / len(true_schema_links))

        return sum(recalls) / total_queries if total_queries > 0 else 0

    def compute_metrics(self) -> None:
        fp_rate = self.false_positive_rate()
        logger.info(f"False Positive Rate: {fp_rate:.2f}")
        column_recall = self.column_recall()
        logger.info(f"Column Recall: {column_recall:.2f}")
        schema_linking_recall = self.schema_linking_recall()
        logger.info(f"Schema Linking Recall: {schema_linking_recall:.2f}")
