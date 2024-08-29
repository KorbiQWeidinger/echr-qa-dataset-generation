import re
import json
import regex
import spacy
import pandas as pd
from pydantic import BaseModel
from fuzzywuzzy import process, fuzz

nlp = spacy.load("en_core_web_trf")

UNIDENTIFIABLE_CASE_NAME = "UNIDENTIFIABLE_CASE_NAME"
UNIDENTIFIABLE_CASE_ID = "UNIDENTIFIABLE_CASE_ID"

last_cited_case_name = UNIDENTIFIABLE_CASE_NAME
last_cited_case_id = UNIDENTIFIABLE_CASE_ID


class Citation(BaseModel):
    case_name: str
    case_id: str
    paragraph_numbers: list[int]


def get_sentences_spacy(text: str):
    doc = nlp(text)
    return [sentence.text for sentence in doc.sents]


def normalize_case_name(case_name: str) -> str:
    case_name = (
        case_name.lower().replace("case of", "").replace("against", "v.").strip()
    )
    case_name = regex.sub(r"\s+", " ", case_name)
    return case_name


def find_citation_paragraphs(text: str):
    # Simple heuristic to clean text from irrelevant paragraphs
    pattern = r"Article(?:(?![()]).){0,30}?§{1,2} \d+[\s.,(;]+"
    text = re.sub(pattern, "", text)

    # Remove spaces around hyphens and normalize spacing around commas and 'and'
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"\s*and\s*", " and ", text)

    pattern = (
        r"§{1,2}\s*((?:\d+-\d+|\d+)(?:, (?:\d+-\d+|\d+))*(?: and (?:\d+-\d+|\d+))*)"
    )
    matches = re.finditer(pattern, text)
    result = []

    for match in matches:
        parts = match.group(1).replace(",", "").split(" and ")
        for part in parts:
            individual_numbers = part.split()
            for number in individual_numbers:
                if "-" in number:
                    start, end = map(int, number.split("-"))
                    result.extend(range(start, end + 1))
                else:
                    result.append(int(number))
    return result


def split_at_case_name(sentence: str) -> list[str]:
    """
    Splits the text at occurrences of ' v. ' and recombines each split
    to ensure each element in the returned list contains exactly one ' v. '.
    This is used to identify segments of text containing case names.
    """
    # Split the sentence using ' v. ' as the delimiter
    splits = sentence.split(" v. ")

    # If there's no ' v. ' there's no need to process further
    if len(splits) == 1:
        return None

    # Combine the splits such that each part contains one ' v. '
    return [splits[i] + " v. " + splits[i + 1] for i in range(len(splits) - 1)]


def find_case_name(text: str):
    # we find the area with the case name
    i = text.find("v.")
    # before the "v." we look for a "(" and remove everything before it
    j = text.rfind("(", 0, i)
    # we find a ; and remove everything after it
    k = text.find(";", i)
    if k == -1:
        k = len(text)

    text = text[j + 1 : k]
    return text


def is_valid_find(snippet: str, closest_case_name: str):
    def fix_common_issues(text: str):
        # remove "the"
        text = re.sub(r"\bthe\b", "", text)
        text = regex.sub(r"\s+", " ", text)
        return text

    snippet = fix_common_issues(snippet)
    closest_case_name = fix_common_issues(closest_case_name)

    # remove all dots from both strings
    snippet = snippet.replace(".", "")
    closest_case_name = closest_case_name.replace(".", "")

    # extract the case name from the snippet
    svi = snippet.find(" v ")
    cvi = closest_case_name.find(" v ")
    l_len = cvi
    r_len = len(closest_case_name) - cvi
    case_name_snippet = snippet[svi - l_len : svi + r_len]

    # we compare the case name from the snippet with the closest case name
    similarity_score = fuzz.ratio(closest_case_name, case_name_snippet)
    num_differences = len(closest_case_name) * (100 - similarity_score) // 100
    # if num_differences > 5:
    # print(f"Snippet: {case_name_snippet}; Closest case name: {closest_case_name}")
    # print(f"SimScore: {similarity_score}, #diffs: {num_differences}")
    return num_differences <= 5, case_name_snippet


def extract_case_name_from_snippet_with_citation(
    snippet: str, possible_citations: dict[str, str]
):
    global last_cited_case_name
    global last_cited_case_id

    original_snippet = snippet

    snippet = find_case_name(snippet)
    snippet = snippet.lower()

    match = process.extractOne(
        snippet, possible_citations.keys(), scorer=fuzz.partial_ratio
    )
    if match:
        closest_match = match[0]
        case_id = possible_citations[closest_match]
    else:
        closest_match = UNIDENTIFIABLE_CASE_NAME
        case_id = UNIDENTIFIABLE_CASE_ID

    valid, _ = is_valid_find(snippet, closest_match)

    if not valid:
        global manual_mappings
        # for manual mappings we look for a direct match
        for k, v in manual_mappings.items():
            if original_snippet.count(k) > 0:
                last_cited_case_name = k
                last_cited_case_id = v
                # print(f'Manual mapping: "{k}" in "{original_snippet}"')
                return k, v

        # we could not identify the citation
        last_cited_case_name = UNIDENTIFIABLE_CASE_NAME
        last_cited_case_id = UNIDENTIFIABLE_CASE_ID

        iv = original_snippet.find(" v. ")
        citation_area = original_snippet[
            max(0, iv - 50) : min(iv + 50, len(original_snippet) - 1)
        ]
        print(
            f'Could not identify: "{citation_area}" <- "{closest_match}" -> "{case_id}"'
        )

        return None, None

    # print(f'Identified citation: "{closest_match}" == "{area}"')
    last_cited_case_name = closest_match
    last_cited_case_id = case_id
    return closest_match, case_id


manual_mappings = {}
# open json file with manual mappings
with open("data/manual_mappings.json", "r") as f:
    manual_mappings = json.load(f)
    manual_mappings = {v: k for k, v in manual_mappings.items()}

all_sentences = []
all_citations = []
all_paragraphs = []
all_guide_ids = []

df = pd.read_csv("data/echr_case_law_guides_with_possible_eng_citations.csv")

current_guide_id = None

for i, row in df.iterrows():
    possible_citations = row["possible_eng_citations"]
    # we handle the possible citations as a dictionary with the case name as the key
    possible_citations = json.loads(possible_citations)
    possible_citations = {
        normalize_case_name(v): k
        for k, v in possible_citations.items()
        if normalize_case_name(v).count("v.") > 0
    }

    paragraph_text = row["paragraph"]
    guide_id = row["guide_id"]

    if guide_id != current_guide_id:
        current_guide_id = guide_id
        last_cited_case_name = UNIDENTIFIABLE_CASE_NAME
        last_cited_case_id = UNIDENTIFIABLE_CASE_ID
        print(f"\n\nProcessing guide {guide_id}")

    paragraph = row["paragraph_id"]

    sentences = get_sentences_spacy(paragraph_text)

    for sentence in sentences:
        split_sentence = split_at_case_name(sentence)
        citations = []

        if not split_sentence:
            # sentence has no case, but might have a citation in the form of a paragraph number
            paragraph_numbers = find_citation_paragraphs(sentence)
            if not paragraph_numbers:
                # sentence has no citation
                all_sentences.append(sentence)
                all_paragraphs.append(paragraph)
                all_guide_ids.append(guide_id)
                all_citations.append([])
                continue

            # we have paragraph numbers so they are part of the previous citation
            citation = Citation(
                case_name=last_cited_case_name,
                case_id=last_cited_case_id,
                paragraph_numbers=paragraph_numbers,
            )

            all_paragraphs.append(paragraph)
            all_sentences.append(sentence)
            all_citations.append(json.dumps([citation.model_dump()]))
            all_guide_ids.append(guide_id)
            continue

        # if we have paragraphs before the first case reference, they are part of the last cited case
        paragraph_numbers = find_citation_paragraphs(split_sentence[0].split("v.")[0])
        if paragraph_numbers:
            citation = Citation(
                case_name=last_cited_case_name,
                case_id=last_cited_case_id,
                paragraph_numbers=paragraph_numbers,
            )
            citations.append(citation)

        # we have case names, one in each split
        for cs in split_sentence:
            case_name, case_id = extract_case_name_from_snippet_with_citation(
                cs, possible_citations if possible_citations else {}
            )
            paragraph_numbers = find_citation_paragraphs(cs.split(" v. ")[1])

            if not case_name:
                # we were not able to identify the citation
                citation = Citation(
                    case_name=UNIDENTIFIABLE_CASE_NAME,
                    case_id=UNIDENTIFIABLE_CASE_ID,
                    paragraph_numbers=paragraph_numbers,
                )
                citations.append(citation)
                continue

            citation = Citation(
                case_name=case_name,
                case_id=case_id,
                paragraph_numbers=paragraph_numbers,
            )
            citations.append(citation)

        all_paragraphs.append(paragraph)
        all_sentences.append(sentence)
        all_citations.append(json.dumps([c.model_dump() for c in citations]))
        all_guide_ids.append(guide_id)

df = pd.DataFrame(
    {
        "guide_id": all_guide_ids,
        "paragraph": all_paragraphs,
        "sentence": all_sentences,
        "citations": all_citations,
    }
)
df.to_csv("data/sentences_with_citations.csv", index=False)
