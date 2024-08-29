import re
import pandas as pd
from pdfminer.high_level import extract_text
from pydantic import BaseModel
from utils.fetch_pdf import fetch_pdf_content


def available_paragraphs(text: str):
    i = 1
    while f"{i}." in text:
        i += 1
    return i - 1


def clean_paragraph(paragraph: str):
    pattern = r"\n\s*([IVXLCDM]+|[A-Z]|[a-u]|\d+)\.\s+"
    all_matches = re.findall(pattern, paragraph)

    if len(all_matches) != 0:
        split_paragraph = re.split(pattern, paragraph, 1)
        paragraph = split_paragraph[0]

    paragraph = re.sub(r"\s+", " ", paragraph)  # remove double spaces
    paragraph = paragraph.strip()
    return paragraph


def extract_paragraph(i: int, text: str):
    split_on_par = text.split(f"\n{i+1}.  ", maxsplit=1)

    if len(split_on_par) != 2:
        split_on_par = text.split(f"\n{i+1}.", maxsplit=1)
        if len(split_on_par) != 2:
            print("ERROR!")
            print(text)
            raise Exception("Failed to split on", i + 1)

    paragraph = clean_paragraph(split_on_par[0])
    continued_text = split_on_par[1]

    return paragraph, continued_text


def get_text(url):
    content = fetch_pdf_content(url)
    text = extract_text(content)
    return text


class GuideParsingMeta(BaseModel):
    guide_id: str
    starting_string: str | None = None
    url: str = "https://ks.echr.coe.int/documents/d/echr-ks/"


class GuideParser:
    def __init__(self, guide_id: str, starting_string: str | None = None, url: str = "https://ks.echr.coe.int/documents/d/echr-ks/", remove_patterns: list[str] = []):
        self.guide_id = guide_id
        self.url = url + guide_id
        self.starting_string = starting_string
        self.remove_patterns = remove_patterns

    def __extract_paragraphs(self, text: str):
        if self.starting_string:
            text = self.starting_string.replace("1. ", "") + text.split(self.starting_string, maxsplit=1)[1]
        else:
            text = text.split("HUDOC keywords", maxsplit=1)[1]
            text = text.split("\n1.  ", maxsplit=1)[1]
        
        text = text.split("List of cited cases")[0]
        return text

    def __clean_paragraphs(self, text: str):
        self.remove_patterns.append(
            r"European Court of Human Rights\s+[\s\n]*\d+\/\d+[\s\n]+Last update: \d{2}\.\d{2}\.\d{4}[\s\n]*(.*?)\n"
        )
        for pattern in self.remove_patterns:
            text = re.sub(pattern, " ", text, flags=re.DOTALL)
        return text

    def __get_text(self):
        return get_text(self.url)

    def __extract_all_paragraphs(self):
        text = self.__get_text()
        text = self.__extract_paragraphs(text)
        text = self.__clean_paragraphs(text)
        return text

    def parse(self):
        text = self.__extract_all_paragraphs()
        n = available_paragraphs(text)
        paragraphs = []
        for i in range(1, n):
            paragraph, text = extract_paragraph(i, text)
            paragraphs.append(paragraph)
        paragraphs.append(clean_paragraph(text))
        return paragraphs

    def to_csv(self):
        paragraphs = self.parse()
        df = pd.DataFrame(paragraphs, columns=['paragraph'], index=range(1, len(paragraphs)+1))
        guide_ids = [self.guide_id] * len(paragraphs)
        df['guide_id'] = guide_ids
        df['paragraph_id'] = df.index
        return df