import os

import regex as re
import yaml
from classes import FFQ, MCQ, FFQOutput, MCQOutput
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chat_models.gigachat import GigaChat
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatYandexGPT
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.llms.vllm import VLLM
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ocr import parse_pdf
from prompts import MULTIPLE_CHOICE_PROMPT
from pydantic import ValidationError


def parse_mcq(output):
    formulation_pattern = re.compile(
        r"```yaml\n(.*)\n```", flags=re.MULTILINE | re.DOTALL
    )
    mcq_str_candidates = formulation_pattern.findall(output["answer"])
    try:
        if len(mcq_str_candidates) == 0:
            raise ValidationError("No MCQ found")
        mcq_yaml = yaml.safe_load(mcq_str_candidates[-1])
        return MCQOutput(
            mcq=MCQ.parse_obj(mcq_yaml),
            valid=True,
            reasoning=output["answer"],
            sources=[ctx.page_content for ctx in output["context"]],
        )
    except Exception as e:
        return MCQOutput(
            mcq=MCQ(
                question=str(e),
                answer_options=[],
                correct_answer=-1,
            ),
            valid=False,
            reasoning=output["answer"],
            sources=[ctx.page_content for ctx in output["context"]],
        )


def parse_ffq(output):
    formulation_pattern = re.compile(
        r"```yaml\n(.*)\n```", flags=re.MULTILINE | re.DOTALL
    )
    mcq_str_candidates = formulation_pattern.findall(output["answer"])
    if len(mcq_str_candidates) == 0:
        raise Exception("AAA")
    ffq_yaml = yaml.safe_load(mcq_str_candidates[-1])

    return FFQOutput(
        ffq=FFQ.parse_obj(ffq_yaml),
        reasoning=output["answer"],
        sources=[ctx.page_content for ctx in output["context"]],
    )


EMBEDDINGS_MODEL = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)


# LLM_NAME = "ISTA-DASLab/Mixtral-8x7B-Instruct-v0_1-AQLM-2Bit-1x16-hf" # "mistralai/Mistral-7B-Instruct-v0.2"
LLM_NAME = os.environ["LLM_NAME"]

if "yandex" in LLM_NAME:
    LLM = ChatYandexGPT(
        model_uri=f"gpt://b1gf3alsif54b30luff2/{LLM_NAME}",
        model_name=LLM_NAME,
        api_key=os.environ["YANDEX_API_KEY"],
    )
elif "gpt" in LLM_NAME:
    LLM = ChatOpenAI(
        model=LLM_NAME,
    )
elif "GigaChat" in LLM_NAME:
    LLM = GigaChat(
        model=LLM_NAME,
        verify_ssl_certs=False,
        max_tokens=2048,
        credentials=os.environ["GIGACHAT_KEY"],
        timeout=90,
    )
else:
    LLM = VLLM(
        model=LLM_NAME,
        trust_remote_code=True,
        max_new_tokens=2048,
        top_k=10,
        top_p=0.95,
        temperature=0.8,
        verbose=False,
        vllm_kwargs={
            "enforce_eager": True,
            "max_model_len": 8192,
            "gpu_memory_utilization": 0.8,
        },
        stop=["<|eot_id|>"],
        # tensor_parallel_size=4,
    )


async def get_question_gen_chains(file_path):
    text = parse_pdf(file_path)

    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=1024,
        chunk_overlap=128,
        length_function=len,
        is_separator_regex="##",
    )
    docs = text_splitter.create_documents([text])

    vector = FAISS.from_documents(docs, EMBEDDINGS_MODEL)
    retriever = vector.as_retriever(search_kwargs={"k": 3})

    mcq_generation_chain = create_stuff_documents_chain(LLM, MULTIPLE_CHOICE_PROMPT)
    mcq_chain = create_retrieval_chain(retriever, mcq_generation_chain) | parse_mcq

    return mcq_chain, None
