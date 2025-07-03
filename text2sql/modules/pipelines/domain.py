from dataclasses import dataclass

from allms.domain.response import ResponseData

from text2sql.modules.schema_linking.domain.model import SchemaLinkingOutput


@dataclass
class PipelineResults:
    schema_linking_responses: list[SchemaLinkingOutput] | None = None
    sql_generation_responses: list[ResponseData] = None