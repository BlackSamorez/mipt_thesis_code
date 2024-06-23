from typing import List

from langchain_core.pydantic_v1 import BaseModel, Field


class MCQ(BaseModel):
    question: str = Field(description="question")
    answer_options: List[str] = Field(description="answer options")
    correct_answer: int = Field(description="correct answer")

    def __repr__(self):
        return (
            f"Question: {self.question}\nAnswer options:\n"
            + "".join(
                [f" - {option}\n" for i, option in enumerate(self.answer_options)]
            )
            + f"Correct answer: {self.correct_answer}"
        )


class MCQOutput(BaseModel):
    mcq: MCQ = Field(description="Question")
    valid: bool = Field(description="Is the question valid")
    reasoning: str = Field(description="Model reasoning")
    sources: List[str] = Field(description="Relevant document passages")


class FFQ(BaseModel):
    question: str = Field(description="question")
    answer: str = Field(description="answer")

    def __repr__(self):
        return f"Question: {self.question}\nAnswer: {self.answer}"


class FFQOutput(BaseModel):
    ffq: FFQ = Field(description="Question")
    reasoning: str = Field(description="Model reasoning")
    sources: List[str] = Field(description="Relevant document passages")
