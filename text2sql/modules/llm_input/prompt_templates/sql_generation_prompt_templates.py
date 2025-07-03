from text2sql.modules.llm_input.prompt_templates.prompt_template import PromptTemplate


class BaselineSqlPromptTemplate(PromptTemplate):
    def __init__(
        self,
        db_schema_prompt: str = "{database_schema}",
        question_prompt: str = "-- {question}",
        knowledge_prompt: str = "-- External Knowledge: {knowledge}",
        instruct_with_knowledge: str = (
            "-- Using valid SQLite and understanding External Knowledge, "
            "answer the following questions for the tables provided above."
        ),
        instruct_no_knowledge: str = (
            "-- Using valid SQLite, answer the following questions for the tables provided above."
        ),
        cot_prompt: str = "Generate the SQL after thinking step by step: ",
    ) -> None:
        super().__init__(
            instruct_with_knowledge=instruct_with_knowledge,
            instruct_no_knowledge=instruct_no_knowledge,
            db_schema_prompt=db_schema_prompt,
            question_prompt=question_prompt,
            knowledge_prompt=knowledge_prompt,
            cot_prompt=cot_prompt,
        )

    def create(
            self,
            use_cot: bool,
            use_knowledge: bool,
            knowledge: str | None = None,
            with_question: bool = True
    ) -> str:
        prompt = super().create(
            use_cot=use_cot,
            use_knowledge=use_knowledge,
            knowledge=knowledge,
            with_question=with_question
        )
        prompt += "\nSELECT "

        return prompt
    
    def create_without_select(
            self, use_cot: bool,
            use_knowledge: bool,
            knowledge: str | None = None,
            with_question: bool = True
    ) -> str:
        prompt = super().create(
            use_cot=use_cot,
            use_knowledge=use_knowledge,
            knowledge=knowledge,
            with_question=with_question
        )

        return prompt
