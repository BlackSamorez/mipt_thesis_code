import asyncio
import json
import logging
import os
# Add parent folder to sys path
import sys
from pathlib import Path

from tqdm import tqdm, trange

sys.path.append(str(Path(__file__).resolve().parents[1]) + "/bot")

# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main(args):
    logger.info(f"Setting up question generation chains for {args.file_path}...")
    mcq_chain, _ = await get_question_gen_chains(args.file_path)
    logger.info("Question generation chains set up.")

    # Load the question topics list
    with open(args.file_path[:-3] + "json", "r") as f:
        question_topics = json.load(f)

    # Envoke chain for each topic
    results = []
    for topic in tqdm(question_topics):
        for _ in trange(args.reps, leave=False):
            mcq = json.loads((await mcq_chain.ainvoke({"input": topic})).json())
            mcq["topic"] = topic
            results.append(mcq)

    # extract file path folder with pathlib
    folder = Path(args.file_path).parent
    # add model name to folder path and create if not exists
    folder = folder / args.model.split("/")[-1]
    folder.mkdir(parents=True, exist_ok=True)
    # add file_path name to folder path
    folder = folder / Path(args.file_path).stem
    # add json extension
    output_path = folder.with_suffix(".json")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        "model",
        type=str,
        help="path to model",
    )

    parser.add_argument(
        "file_path",
        type=str,
        help="path to pdf file",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=1,
        help="Number of reps for each question",
    )

    args = parser.parse_args()
    os.environ["LLM_NAME"] = args.model
    from chains import get_question_gen_chains

    asyncio.run(main(args=args))
