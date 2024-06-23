from classes import FFQOutput, MCQOutput
from md2pdf.core import md2pdf


def print_question(question):
    if isinstance(question, MCQOutput):
        return question.mcq.__repr__()
    elif isinstance(question, FFQOutput):
        return question.ffq.__repr__()
    else:
        raise NotImplementedError(f"Unknown question type {type(question)}")


def compile(path, questions):
    with open(path, "w") as file:
        file.write(
            "\n\n".join(
                [
                    f"## Question {i}\n{print_question(question)}"
                    for i, question in enumerate(questions)
                ]
            )
        )
