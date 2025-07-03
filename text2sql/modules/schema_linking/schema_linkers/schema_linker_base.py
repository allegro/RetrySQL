from text2sql.modules.schema_linking.domain.model import SchemaLinkingOutput


class SchemaLinkerBase:
    def __init__(self, use_external_knowledge: bool) -> None:
        self.use_external_knowledge = use_external_knowledge

    def __call__(self, **kwargs) -> list[SchemaLinkingOutput]:
        return self.forward(**kwargs)

    def forward(self, **kwargs) -> list[SchemaLinkingOutput]:
        pass
