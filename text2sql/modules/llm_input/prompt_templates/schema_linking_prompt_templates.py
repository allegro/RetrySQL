from text2sql.modules.llm_input.prompt_templates.prompt_template import PromptTemplate


class TableAndColumnSchemaLinkingPromptTemplate(PromptTemplate):
    def __init__(
        self,
        db_schema_prompt: str = "Database Schema: \n{database_schema}",
        question_prompt: str = "-- Question: {question}",
        knowledge_prompt: str = "-- External Knowledge: {knowledge}",
        instruct_with_knowledge: str = (
            "-- Based on Database Schema provided above and understanding External Knowledge, "
            "your task is to select table-column pairs (called `schema links`) most relevant to the given Question."
        ),
        instruct_no_knowledge: str = (
            "-- Based on Database Schema provided above, "
            "your task is to select table-column pairs most relevant to the given Question."
        ),
        cot_prompt: str = "Choose the relevant table-column pairs after thinking step by step: ",
    ) -> None:
        super().__init__(
            instruct_with_knowledge=instruct_with_knowledge,
            instruct_no_knowledge=instruct_no_knowledge,
            db_schema_prompt=db_schema_prompt,
            question_prompt=question_prompt,
            knowledge_prompt=knowledge_prompt,
            cot_prompt=cot_prompt,
        )
