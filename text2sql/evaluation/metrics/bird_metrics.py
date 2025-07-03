import math
import multiprocessing as mp
import re
import sqlite3
import time
from typing import Any, Callable

import numpy as np
from func_timeout import FunctionTimedOut, func_timeout
from pydash import chain
from upath import UPath

from text2sql.commons.logging_utils import get_logger
from text2sql.evaluation.domain.enums import QuestionDifficulty
from text2sql.evaluation.domain.model import EvaluationResult, GenerationResult, GroundTruth, QueryExecutionResult, ResultPair
from text2sql.settings import DATASETS_DIR

logger = get_logger(__name__)


class BirdMetrics:
    def __init__(
        self,
        cpu_count: int,
        timeout_in_seconds: float,
        base_path: UPath = DATASETS_DIR.joinpath("BIRD_dev"),
    ):
        self._base_path = base_path
        self._cpu_count = cpu_count
        self._timeout_in_seconds = timeout_in_seconds

    def calculate_execution_accuracy(
        self,
        results: list[GenerationResult],
        ground_truth: list[GroundTruth],
    ) -> EvaluationResult:
        generation_results_lookup_dict = {result.question_id: result for result in results}

        pairs_to_check = [
            ResultPair(
                question_id=example.question_id,
                question_difficulty=example.question_difficulty,
                database_name=example.database_name,
                ground_truth_query=example.sql_query,
                generated_query=generation_results_lookup_dict[example.question_id].sql_query,
            )
            for example in ground_truth
        ]

        easy_results = self._run_parallel_execution_accuracy_queries(
            bird_result_pairs=[pair for pair in pairs_to_check if pair.question_difficulty == QuestionDifficulty.EASY],
            cpu_count=self._cpu_count
        )
        medium_results = self._run_parallel_execution_accuracy_queries(
            bird_result_pairs=[pair for pair in pairs_to_check if pair.question_difficulty == QuestionDifficulty.MEDIUM],
            cpu_count=self._cpu_count
        )
        hard_results = self._run_parallel_execution_accuracy_queries(
            bird_result_pairs=[pair for pair in pairs_to_check if pair.question_difficulty == QuestionDifficulty.HARD],
            cpu_count=self._cpu_count
        )

        ground_truth_difficulty_counts = self._count_ground_truth_by_difficulty(ground_truth)

        return EvaluationResult(
            easy_accuracy=self._compute_accuracy_percentage(
                execution_match_results=easy_results, 
                dataset_size=ground_truth_difficulty_counts[QuestionDifficulty.EASY]
            ),
            medium_accuracy=self._compute_accuracy_percentage(
                execution_match_results=medium_results, 
                dataset_size=ground_truth_difficulty_counts[QuestionDifficulty.MEDIUM]
            ),
            hard_accuracy=self._compute_accuracy_percentage(
                execution_match_results=hard_results, 
                dataset_size=ground_truth_difficulty_counts[QuestionDifficulty.HARD]
            ),
            overall_accuracy=self._compute_accuracy_percentage(
                execution_match_results=easy_results + medium_results + hard_results, 
                dataset_size=len(ground_truth)
            )
        )

    def calculate_valid_efficiency_score(
        self,
        results: list[GenerationResult],
        ground_truth: list[GroundTruth],
    ):
        generation_results_lookup_dict = {result.question_id: result for result in results}

        pairs_to_check = [
            ResultPair(
                question_id=example.question_id,
                database_name=example.database_name,
                question_difficulty=example.question_difficulty,
                ground_truth_query=example.sql_query,
                generated_query=generation_results_lookup_dict[example.question_id].sql_query,
            )
            for example in ground_truth
        ]

        results = self._run_parallel_valid_efficiency_score_queries(
            bird_result_pairs=pairs_to_check, cpu_count=self._cpu_count
        )

        return self._compute_accuracy_percentage(
            execution_match_results=results, 
            dataset_size=len(ground_truth)
        )
    
    def _compute_accuracy_percentage(self, execution_match_results: list[int], dataset_size: int) -> float:
        return 100 * sum(execution_match_results) / dataset_size
    
    def _count_ground_truth_by_difficulty(self, ground_truth: list[GroundTruth]) -> dict[QuestionDifficulty, int]:
        return (
            chain(ground_truth)
            .count_by(lambda example: example.question_difficulty)
            .value()
        )

    @staticmethod
    def _remove_outliers(input_list: list[float]) -> list[float]:
        mean = np.mean(input_list, axis=0)
        standard_deviation = np.std(input_list, axis=0)

        filtered_input = []

        for number in input_list:
            if number < mean + 3 * standard_deviation and number > mean - 3 * standard_deviation:
                filtered_input.append(number)

        return filtered_input

    @staticmethod
    def get_database_path(base_path: UPath, database_name: str) -> UPath:
        return base_path.joinpath("dev_databases", database_name, f"{database_name}.sqlite")

    @staticmethod
    def _execute_with_timeout(
        timeout_in_seconds: float,
        function_to_execute: Callable[..., bool],
        function_arguments: tuple[Any],
    ):
        try:
            return func_timeout(
                timeout_in_seconds,
                function_to_execute,
                args=function_arguments,
            )
        except FunctionTimedOut:
            logger.info(
                f"Execution took more than {timeout_in_seconds} seconds, timeout for question {function_arguments[0]}"
            )
            return 0

    def _run_parallel_execution_accuracy_queries(self, bird_result_pairs: list[ResultPair], cpu_count: int):
        logger.info(f"Starting parallel execution accuracy queries with {cpu_count} CPU cores.")
        process_pool = mp.Pool(processes=cpu_count)

        results = [
            process_pool.apply_async(
                BirdMetrics._execute_with_timeout,
                args=(
                    self._timeout_in_seconds,
                    BirdMetrics._execute_and_compare,
                    (
                        result_pair.question_id,
                        result_pair.generated_query,
                        result_pair.ground_truth_query,
                        BirdMetrics.get_database_path(
                            base_path=self._base_path,
                            database_name=result_pair.database_name,
                        ),
                    ),
                ),
            )
            for result_pair in bird_result_pairs
        ]

        process_pool.close()
        process_pool.join()
        logger.info("Completed all parallel execution accuracy queries.")

        return [async_result.get() for async_result in results]

    def _run_parallel_valid_efficiency_score_queries(
        self,
        bird_result_pairs: list[ResultPair],
        cpu_count: int,
        number_of_iterations: int = 100,
    ):
        logger.info(f"Starting parallel valid efficiency score queries with {cpu_count} CPU cores.")
        process_pool = mp.Pool(processes=cpu_count)

        results = [
            process_pool.apply_async(
                BirdMetrics._execute_with_timeout,
                args=(
                    self._timeout_in_seconds * number_of_iterations,
                    BirdMetrics._execute_query_multiple_iterations,
                    (
                        result_pair.question_id,
                        result_pair.generated_query,
                        result_pair.ground_truth_query,
                        BirdMetrics.get_database_path(
                            base_path=self._base_path,
                            database_name=result_pair.database_name,
                        ),
                        number_of_iterations,
                    ),
                ),
            )
            for result_pair in bird_result_pairs
        ]

        process_pool.close()
        process_pool.join()
        logger.info("Completed all parallel valid efficiency score queries.")

        return [async_result.get() for async_result in results]

    @staticmethod
    def _execute_and_compare(
        question_id: str,
        generated_query: str,
        ground_truth_query: str,
        database_path: str,
    ) -> int:
        generated_query_results = BirdMetrics._execute_single_query(
            question_id=question_id,
            query=generated_query,
            database_path=database_path,
            should_fetch_results=True,
        ).result_set

        ground_truth_results = BirdMetrics._execute_single_query(
            question_id=question_id,
            query=ground_truth_query,
            database_path=database_path,
            should_fetch_results=True,
        ).result_set

        return 1 if set(generated_query_results) == set(ground_truth_results) else 0

    @staticmethod
    def _calculate_reward(time_ratio: float) -> float:
        if time_ratio == 0:
            return 0
        elif time_ratio >= 2:
            return 1.25
        elif time_ratio >= 1 and time_ratio < 2:
            return 1
        elif time_ratio >= 0.5 and time_ratio < 1:
            return 0.75
        elif time_ratio >= 0.25 and time_ratio < 0.5:
            return 0.5
        else:
            return 0.25

    @staticmethod
    def _execute_query_multiple_iterations(
        question_id: str,
        generated_query: str,
        ground_truth_query: str,
        database_path: str,
        number_of_iterations: int,
    ):
        generated_query_results = BirdMetrics._execute_single_query(
            question_id=question_id,
            query=generated_query,
            database_path=database_path,
            should_fetch_results=True,
        ).result_set

        ground_truth_results = BirdMetrics._execute_single_query(
            question_id=question_id,
            query=ground_truth_query,
            database_path=database_path,
            should_fetch_results=True,
        ).result_set

        if set(generated_query_results) != set(ground_truth_results):
            return 0

        relative_times = []

        for _ in range(number_of_iterations):
            generated_query_time = BirdMetrics._execute_single_query(
                question_id=question_id,
                query=generated_query,
                database_path=database_path,
                should_fetch_results=False,
            ).execution_time

            ground_truth_time = BirdMetrics._execute_single_query(
                question_id=question_id,
                query=ground_truth_query,
                database_path=database_path,
                should_fetch_results=False,
            ).execution_time

            relative_times.append(ground_truth_time / generated_query_time)

        relative_times_no_outliers = BirdMetrics._remove_outliers(relative_times)
        time_ratio = sum(relative_times_no_outliers) / len(relative_times_no_outliers)

        return math.sqrt(BirdMetrics._calculate_reward(time_ratio))

    @staticmethod
    def _execute_single_query(
        question_id: str,
        query: str,
        database_path: str,
        should_fetch_results: bool,
    ) -> QueryExecutionResult:
        logger.info(f"Resolving question ID {question_id}")
        logger.info(f"Executing query on database: {database_path}")
        database_connection = sqlite3.connect(database_path)
        cursor = database_connection.cursor()

        start_time = time.time()

        try:
            cursor.execute(query)
        except Exception as e:
            logger.error(f"Error executing query: {e}")

        execution_time = time.time() - start_time
        logger.info(f"Query executed in {execution_time:.4f} seconds.")

        if should_fetch_results:
            return QueryExecutionResult(
                result_set=cursor.fetchall(),
                execution_time=execution_time,
            )

        return QueryExecutionResult(result_set=[], execution_time=execution_time)
