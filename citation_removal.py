"""
For evaluation purposes we aim to remove the citations from the text. 
This script is used to remove the citations from the text.
"""

import spacy
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
import re

nlp = spacy.load("en_core_web_trf")


def split_sentences(text: str) -> list[str]:
    doc = nlp(text)
    return [sentence.text for sentence in doc.sents]


llm = ChatOpenAI(model="gpt-3.5-turbo")

PROMPT = """
Please remove all citations from the following sentence. 
We will verify that the sentence without citations matches the original sentence except for removal.
ONLY remove citations! Do NOT change the sentence otherwise! 
Your response should ONLY contain the sentence without citations.

Original Sentence: {sentence}
Sentence without citations:
"""

df = pd.read_csv("data/echr_qa_dataset.csv")

ANSWER_NO_CITATIONS = "answer_no_citations"


def verify_sentence_without_citations(sentence: str, cleaned_sentence: str):
    # replace all punctuation with spaces
    sentence = re.sub(r"[^\w\s]", " ", sentence)
    cleaned_sentence = re.sub(r"[^\w\s]", " ", cleaned_sentence)

    # split sentences into words on space and lowercase them
    base_words = [word.lower() for word in sentence.split()]
    check_words = [word.lower() for word in cleaned_sentence.split()]

    # Use an iterator to check ordered inclusion
    iter_base = iter(base_words)

    # Check each word in check_words if it can be found in base_words in order
    for word in check_words:
        if word not in iter_base:
            return False, f"Word {word} not found in sentence"

    return True, ""


for i, row in df.iterrows():
    if ANSWER_NO_CITATIONS in row and pd.notnull(row[ANSWER_NO_CITATIONS]):
        print("Skipping row", i, "as it already has been processed")
        continue

    print("Processing row", i)

    answer = row["answer"]
    sentences = split_sentences(answer)
    sentences_no_citations = []
    error = False

    for sentence in sentences:
        if "v." not in sentence and "§" not in sentence:
            print("No citation in sentence:", sentence)
            sentences_no_citations.append(sentence)
            continue

        retries = 0
        prompt = PromptTemplate.from_template(PROMPT).format(sentence=sentence)
        error = None
        while retries < 3:
            response = llm.invoke(prompt)
            response = response.content

            valid, _error = verify_sentence_without_citations(sentence, response)

            if valid:
                print()
                print("Original:", sentence)
                print("Success:", response)
                print()
                sentences_no_citations.append(response)
                break

            prompt = (
                prompt
                + response
                + f"\nError: {_error} \nFixed sentence without citations:"
            )

            retries += 1

            print()
            print("Tries:", retries)
            print("Original:", sentence)
            print("Failure:", response)
            print()

            if retries == 3:
                error = _error

    if error:
        print("Error in row", i)
        df.at[i, ANSWER_NO_CITATIONS] = response
        df.to_csv("data/echr_qa_dataset_no_citations.csv", index=False)
        continue

    print("Success in row", i)
    df.at[i, ANSWER_NO_CITATIONS] = " ".join(sentences_no_citations)
    df.to_csv("data/echr_qa_dataset_no_citations.csv", index=False)
