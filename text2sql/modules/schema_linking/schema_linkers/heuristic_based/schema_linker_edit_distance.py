import re

import Levenshtein
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pydash import chain

from text2sql.datasets.domain.model import SchemaLink
from text2sql.modules.schema_linking.domain.model import (
    SchemaLinkerHeuristicBasedAlgorithmInputExample,
    SchemaLinkingOutput,
)
from text2sql.modules.schema_linking.schema_linkers.schema_linker_base import SchemaLinkerBase


class SchemaLinkerEditDistance(SchemaLinkerBase):
    def __init__(self, use_external_knowledge: bool, edit_distance: int = 0, min_token_length: int = 3) -> None:
        super().__init__(use_external_knowledge=use_external_knowledge)
        nltk.download("stopwords")
        nltk.download("punkt_tab")

        self._edit_distance = edit_distance
        self._stop_words = set(stopwords.words("english"))
        self._min_token_length = min_token_length

    def _is_match(self, word1: str, word2: str) -> bool:
        return Levenshtein.distance(word1, word2) <= self._edit_distance

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.lower().replace("_", " ").replace("-", " ")
        return re.sub(r"[^a-z0-9\s]", "", text)

    def _tokenize_and_filter_input_string(self, input_string: str) -> list[str]:
        tokens = word_tokenize(input_string)
        return [token for token in tokens if token not in self._stop_words and len(token) >= self._min_token_length]

    def _are_any_matching_tokens(self, query_tokens: list[str], entity_tokens: list[str]) -> bool:
        for entity_token in entity_tokens:
            if any(self._is_match(entity_token, token) for token in query_tokens):
                return True
        return False

    def _match_table_or_columns(
        self,
        query_tokens: list[str],
        table_name: str,
        columns: list[str],
    ) -> SchemaLink | None:
        normalized_table_name = self._normalize_text(table_name)
        table_name_tokens = self._tokenize_and_filter_input_string(normalized_table_name)
        found_matching_table_name = self._are_any_matching_tokens(query_tokens, table_name_tokens)
        matched_columns = self._find_matching_columns(query_tokens, columns)

        if len(matched_columns) > 0:
            return SchemaLink(table_name=table_name, columns=[column.column_name for column in matched_columns])
        elif len(matched_columns) == 0 and found_matching_table_name:
            return SchemaLink(table_name=table_name, columns=[column.column_name for column in columns])

        return None

    def _find_matching_columns(self, query_tokens: list[str], columns: list[str]) -> list[str]:
        return (
            chain(columns)
            .map(lambda column: (column, self._normalize_text(column.column_name)))
            .map(
                lambda original_and_normalized_col_names_tuple: (
                    original_and_normalized_col_names_tuple[0],
                    self._tokenize_and_filter_input_string(original_and_normalized_col_names_tuple[1]),
                )
            )
            .filter(
                lambda original_and_normalized_col_names_tuple: self._are_any_matching_tokens(
                    query_tokens, original_and_normalized_col_names_tuple[1]
                )
            )
            .map(lambda original_and_normalized_col_names_tuple: original_and_normalized_col_names_tuple[0])
            .value()
        )

    def _process_example(self, example: SchemaLinkerHeuristicBasedAlgorithmInputExample) -> SchemaLinkingOutput:
        query_string = example.question
        if self.use_external_knowledge:
            query_string += " " + example.external_knowledge

        query_tokens = self._tokenize_and_filter_input_string(self._normalize_text(query_string))

        schema_links = (
            chain(example.database_schema.tables.items())
            .map(
                lambda table_name_and_columns: self._match_table_or_columns(
                    query_tokens, table_name_and_columns[0], table_name_and_columns[1]
                )
            )
            .filter(lambda schema_link: schema_link is not None)
            .value()
        )

        if len(schema_links) == 0:
            schema_links = [SchemaLink(table_name="", columns=[])]

        return SchemaLinkingOutput(
            question_id=example.question_id,
            question=example.question,
            schema_links=schema_links,
        )

    def forward(
        self, input_examples: list[SchemaLinkerHeuristicBasedAlgorithmInputExample]
    ) -> list[SchemaLinkingOutput]:
        return [self._process_example(example) for example in input_examples]
