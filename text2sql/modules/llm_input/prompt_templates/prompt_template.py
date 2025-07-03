class PromptTemplate:
    def __init__(
        self,
        instruct_with_knowledge: str,
        instruct_no_knowledge: str,
        db_schema_prompt: str = "{database_schema}",
        question_prompt: str = "-- {question}",
        knowledge_prompt: str = "-- External Knowledge: {knowledge}",
        cot_prompt: str | None = "Answer after thinking step by step: ",
    ) -> None:
        self.db_schema_prompt = db_schema_prompt
        self.question_prompt = question_prompt
        self.knowledge_prompt = knowledge_prompt
        self.cot_prompt = cot_prompt
        self.instruct_with_knowledge = instruct_with_knowledge
        self.instruct_no_knowledge = instruct_no_knowledge

    def create(
            self,
            use_cot: bool,
            use_knowledge: bool,
            knowledge: str | None = None,
            with_question: bool = True
    ) -> str:
        prompt = self.db_schema_prompt + "\n\n"

        if use_knowledge and knowledge and knowledge != "":
            prompt += self.knowledge_prompt + "\n"
            prompt += self.instruct_with_knowledge + "\n"
        else:
            prompt += self.instruct_no_knowledge + "\n"

        if with_question:
            prompt += self.question_prompt + "\n"

        if use_cot:
            prompt += self.cot_prompt

        return prompt
