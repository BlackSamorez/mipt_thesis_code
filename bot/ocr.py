from marker.convert import convert_single_pdf
from marker.logger import configure_logging
from marker.models import load_all_models

configure_logging()


def parse_pdf(file_path):
    model_lst = load_all_models()
    full_text, out_meta = convert_single_pdf(file_path, model_lst)
    return full_text
