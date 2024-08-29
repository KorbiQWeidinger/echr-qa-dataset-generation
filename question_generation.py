import json
import re
import spacy
import pandas as pd
from enum import Enum
from langchain_core.messages import AIMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from scipy.spatial.distance import cosine
from fuzzywuzzy import process, fuzz
import concurrent.futures
import time

nlp = spacy.load("en_core_web_trf")


class Guide(Enum):
    GUIDE_ART_1_ENG = "guide_art_1_eng"
    GUIDE_ART_2_ENG = "guide_art_2_eng"
    GUIDE_ART_3_ENG = "guide_art_3_eng"
    GUIDE_ART_4_ENG = "guide_art_4_eng"
    GUIDE_ART_5_ENG = "guide_art_5_eng"

    GUIDE_ART_6_CIVIL_ENG = "guide_art_6_civil_eng"
    GUIDE_ART_6_CRIMINAL_ENG = "guide_art_6_criminal_eng"
    GUIDE_ART_7_ENG = "guide_art_7_eng"
    GUIDE_ART_8_ENG = "guide_art_8_eng"
    GUIDE_ART_9_ENG = "guide_art_9_eng"

    GUIDE_ART_10_ENG = "guide_art_10_eng"
    GUIDE_ART_11_ENG = "guide_art_11_eng"
    GUIDE_ART_12_ENG = "guide_art_12_eng"
    GUIDE_ART_13_ENG = "guide_art_13_eng"
    GUIDE_ART_14_ART_1_PROTOCOL_12_ENG = "guide_art_14_art_1_protocol_12_eng"

    GUIDE_ART_15_ENG = "guide_art_15_eng"
    GUIDE_ART_17_ENG = "guide_art_17_eng"
    GUIDE_ART_18_ENG = "guide_art_18_eng"
    ADMISSIBILITY_GUIDE_ENG = "Admissibility_guide_ENG"
    GUIDE_ART_46_ENG = "guide_art_46_eng"

    GUIDE_ART_1_PROTOCOL_1_ENG = "guide_art_1_protocol_1_eng"
    GUIDE_ART_2_PROTOCOL_1_ENG = "guide_art_2_protocol_1_eng"
    GUIDE_ART_3_PROTOCOL_1_ENG = "guide_art_3_protocol_1_eng"
    GUIDE_ART_2_PROTOCOL_4_ENG = "guide_art_2_protocol_4_eng"
    GUIDE_ART_3_PROTOCOL_4_ENG = "guide_art_3_protocol_4_eng"

    GUIDE_ART_4_PROTOCOL_4_ENG = "guide_art_4_protocol_4_eng"
    GUIDE_ART_1_PROTOCOL_7_ENG = "guide_art_1_protocol_7_eng"
    GUIDE_ART_2_PROTOCOL_7_ENG = "guide_art_2_protocol_7_eng"
    GUIDE_ART_4_PROTOCOL_7_ENG = "guide_art_4_protocol_7_eng"

    GUIDE_DATA_PROTECTION_ENG = "guide_data_protection_eng"
    GUIDE_ENVIRONMENT_ENG = "guide_environment_eng"
    GUIDE_IMMIGRATION_ENG = "guide_immigration_eng"
    GUIDE_MASS_PROTESTS_ENG = "guide_mass_protests_eng"
    GUIDE_PRISONERS_RIGHTS_ENG = "guide_prisoners_rights_eng"
    GUIDE_LGBTI_RIGHTS_ENG = "guide_lgbti_rights_eng"
    GUIDE_SOCIAL_RIGHTS_ENG = "guide_social_rights_eng"
    GUIDE_TERRORISM_ENG = "guide_terrorism_eng"


citations_df = pd.read_csv("data/sentences_with_citations_usable.csv")


def is_all_usable_paragraph(i: int, guide: Guide):
    paragraph_df = citations_df[
        (citations_df["paragraph"] == i) & (citations_df["guide_id"] == guide.value)
    ]
    return paragraph_df["usable"].all()


def has_only_valid_citations(sentences: list[str]):
    for sentence in sentences:
        match = process.extractOne(
            sentence, citations_df["sentence"], scorer=fuzz.partial_ratio
        )
        find_sentence_df = citations_df[citations_df["sentence"] == match[0]]
        if find_sentence_df.empty:
            print(f"Sentence not found in the citations_df: {sentence}")
            return False
        usable = find_sentence_df["usable"].values[0]
        if not usable:
            return False
    return True


def get_sentences(guide: Guide, paragraphs: list[int]):
    sentences = []
    for pn in paragraphs:
        paragraph_df = citations_df[
            (citations_df["paragraph"] == pn)
            & (citations_df["guide_id"] == guide.value)
        ]
        sentences.extend(paragraph_df["sentence"].tolist())
    return sentences


QG_WITH_SEARCH_1 = """
Case law paragraphs: 
{paragraphs}

Task:
Define a single challenging legal question that can be answered with the given case law paragraphs.
Reuse the language from the case law in the question. 
Make sure the question is general and does NOT mention specific cases. 

Question: 
"""

QG_WITH_SEARCH_2 = """
Case law sentences: 
{sentences}

Task: Answer the following question with the provided sentences.
At the end of each sentence in your answer cite the used sentences in square brackets.

Question: 
{question}
Answer: 
"""

QG_LEGAL_1_1 = """
Your objective is to develop educational and challenging questions for lawyers working with ECHR case law and for judges who want to draft judgments based on ECHR case law.
Each question should be based on the provided paragraphs from the ECHR case law guides.
When formulating a question reuse the language from the ECHR case law and match legal doctrines to specific facts. 
Make sure the question is general and does NOT mention specific cases. 
Emphasize the patterns that link facts to specific legal doctrines.

Doctrines and facts: 
{paragraphs}

Steps:
1. Identify how the margin of appreciation and positive obligations apply in relation to the State's discretion
2. Identify the reasons that justify necessity and pressing social needs
3. Identify the reasons that command that rights be effective in their application
4. Identify how reasonable measures apply in relation to the State's discretion and to restrictions imposed by States or private individuals
5. Identify the reasons set forth by the Court to defer to domestic reasons provided by domestic authorities
6. Define a question that can be answered exactly by the given legal doctrines and applicable facts to those doctrines

Answer Template:
Margin of appreciation: {{how do the margin of appreciation and positive obligations apply in relation to the State's discretion}}
Necessities: {{reasons that justify necessity and pressing social needs}}
Effectivity: {{reasons that command that rights be effective in their application}}
Reasonable Measures: {{how do reasonable measures apply in relation to the State's discretion and to restrictions imposed by States or private individuals?}}
Domestic Reasons: {{the reasons set forth by the Court to defer to domestic reasons provided by domestic authorities}}
Question: {{define a single question that can be answered exactly by the given legal doctrines and applicable facts reusing the language from the ECHR case law}}
"""

QG_LEGAL_1_2 = """
Your objective is to develop educational and challenging question-answer pairs for lawyers working with ECHR case law and for judges who want to draft judgments based on ECHR case law.

Doctrines and facts: 
{sentences}

Task: Answer the following question based on the provided doctrines and facts from the ECHR case law guides.

Use the provided doctrines and facts to answer the question.
Use citations! At the end of each sentence in your answer add all the numbers of the used facts and doctrines in square brackets.

Question: 
{question}
Answer: 
"""

QG_LEGAL_2_1 = """
Your objective is to develop educational and challenging questions for lawyers working with ECHR case law and for judges who want to draft judgments based on ECHR case law.
Each question should be based on the provided paragraphs from the ECHR case law guides.
When formulating a question reuse the language from the ECHR case law and match legal doctrines to specific facts.
Make sure the question is general and does NOT mention specific cases. 
Emphasize the patterns that link facts to specific legal doctrines.

Doctrines and facts: 
{paragraphs}

Steps:
1. Identify what are the criteria under the Convention for applying the rights enshrined therein.
2. Identify the conditions that the Court sets forth with view to analyse the legality of domestic measures and restrictions
3. Identify the reasons provided by the Court to protect applicants and victims and to differentiate between them.
4. Identify the reasons set forth by the Court to distinguish between legal doctrines and contextual application of those doctrines.
5. Assign a set of facts to its corresponding Article and identify a sequence of reasons that justify the application of the Article to those facts.
6. Explain why analogies and comparisons between Article-fact pair are pertinent
7. Identify separately reasons that are linked to margin of appreciation of the State from those linked to the Court's appreciation of facts
8. Define a question that can be answered exactly by the given Article-facts correspondence

Answer Template:
Criteria for rights: {{how does the Court define the criteria for applying rights}}
Legality of domestic measures and restrictions: {{conditions that determine that domestic measures are compliant with the Convention}}
Protection and differentiation of applicants and victims: {{circumstances and conditions that limit or allow applicants and victims to present their case}}
Distinction between legal doctrines and contextual application: {{how and why and in what circumstances legal doctrines apply to specific facts}}
Article-facts correspondence: {{the reasons set forth by the Court to justify the application of the Article/Articles to those facts}}
Analogies and comparisons between Article-fact pair {{the reasons set forth by the Court to justify why articles and facts differ from one another}}
margin of appreciation of States and Court's appreciation {{the reasons, circumstances and conditions set forth by the Court to explain margin of appreciation of States and its own appreciation}}
Question: {{define a single question that can be answered exactly by the given legal doctrines and applicable facts and by the Article-fact pair, reusing the language from the ECHR case law}}
"""

QG_LEGAL_2_2 = """
Your objective is to develop question-answer pairs that could help lawyers working with ECHR case law 
and judges who want to draft judgments based on ECHR find the best arguments to justify violations or non-violations of the ECHR.

Doctrines, articles, and facts:
{sentences}

Answer the following question following the sequence: characterization of facts according to ECHR doctrines and justification of those facts in relation to specific articles of the ECHR.
Use the provided doctrines and facts to answer the question.
Use citations! At the end of each sentence in your answer add the numbers of the used facts and doctrines in square brackets.

Question: 
{question}
Answer: 
"""


def extract_question(message: AIMessage):
    match = re.search(r"Question: (.*)?", message.content)
    if match:
        return match.group(1)
    raise ValueError("No question found in the message", message.content)


def extract_question_simple(message: AIMessage):
    match = re.search(r"Question: (.*)?", message.content)
    if match:
        message.content = match.group(1)
    question = message.content.split("?")[0] + "?"
    question = question.replace("Question: ", "")
    return question


def extract_chosen_paragraphs(message: AIMessage):
    pattern = r"Paragraphs:.*?(\[.*\])"
    numbers = []
    matches = re.findall(pattern, message.content)
    for match in matches:
        numbers.extend(re.findall(r"\d+", match))
    chosen_paragraphs = [int(number) for number in numbers]
    if not chosen_paragraphs:
        raise ValueError("No chosen paragraphs found in the message", message.content)
    return chosen_paragraphs


def extract_chosen_sentences(message: AIMessage):
    pattern = r"Chosen Sentences:.*?(\[.*\])"
    numbers = []
    matches = re.findall(pattern, message.content)
    for match in matches:
        numbers.extend(re.findall(r"\d+", match))
    chosen_sentences = [int(number) for number in numbers]
    if not chosen_sentences:
        raise ValueError("No chosen sentences found in the message", message.content)
    return chosen_sentences


def extract_citations(text: str, allowed: set[int] = set()):
    pattern = r"\[\d+(?:-\d+)?(?:, \d+(?:-\d+)?)*\]"
    all_ints = set()
    matches = re.findall(pattern, text)
    for match in matches:
        items = re.findall(r"\d+(?:-\d+)?", match)
        for item in items:
            if "-" in item:
                start, end = map(int, item.split("-"))
                nums = range(start, end + 1)
            else:
                nums = [int(item)]
            for num in nums:
                if not allowed or num in allowed:
                    all_ints.add(num)
    return all_ints


def get_sentences_spacy(text: str):
    doc = nlp(text)
    return [sentence.text for sentence in doc.sents]


def numbered_string(strings: list[str]):
    return "\n".join(f"[{i+1}]: {s}" for i, s in enumerate(strings))


guides_df = pd.read_csv("data/echr_case_law_guides_with_openai_embeddings.csv")


def get_guide(guide: Guide):
    df_copy = guides_df.copy()
    df_copy = df_copy[df_copy["guide_id"].isin([guide.value])]
    df_copy = df_copy.reset_index(drop=True)
    df_copy.index = df_copy.index + 1
    return df_copy


def get_paragraphs(guide: Guide, paragraphs: list[int]):
    df = get_guide(guide)
    paragraphs_df = df.loc[paragraphs]
    paragraphs_str = " ".join(paragraphs_df["paragraph"])
    return paragraphs_str


def numbered_paragraphs_string(guide: Guide, paragraphs: list[int]):
    df = get_guide(guide)
    paragraphs_df = df.loc[paragraphs]
    paragraphs_list = paragraphs_df["paragraph"].tolist()
    paragraphs_numbered_str = "\n".join(
        f"{i+1}. {paragraph}" for i, paragraph in enumerate(paragraphs_list)
    )
    return paragraphs_numbered_str


def numbered_sentence_string(guide: Guide, paragraphs: list[int]):
    sentences = get_sentences(guide, paragraphs)
    sentences_with_numbers = "\n".join(f"[{i+1}]: {s}" for i, s in enumerate(sentences))
    return sentences_with_numbers


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

OPENAI_EMBEDDINGS = "openai_embeddings"
guides_df[OPENAI_EMBEDDINGS] = guides_df[OPENAI_EMBEDDINGS].apply(lambda x: eval(x))


def get_top_n_similarities(
    question: str, n: int = 5, desired_guide_ids: list[str] = []
):
    q_embedding = embeddings.embed_query(question)
    df_copy = guides_df.copy()
    if desired_guide_ids:
        df_copy = df_copy[df_copy["guide_id"].isin(desired_guide_ids)]

    df_copy["similarity"] = df_copy[OPENAI_EMBEDDINGS].apply(
        lambda x: 1 - cosine(x, q_embedding)
    )

    return df_copy.nlargest(n, "similarity")


llm = ChatOpenAI(model="gpt-3.5-turbo-16k")


def question_generation_with_search(
    guide: Guide, paragraphs: list[int], qg_prompt: str, answer_prompt: str
):
    """
    Uses two prompts to generate a question-answer pair with question to paragraph similarity search.
    The qg_prompt uses COT to generate a question based on the provided paragraphs.
    Using semantic search, we aim to include potentially relevant paragraphs for the answer.
    The answer_prompt generates an answer with citations based on the question and the paragraphs.
    We then use these citations to determine the final question-answer pair.
    """
    prompt = PromptTemplate.from_template(qg_prompt).format(
        paragraphs=get_paragraphs(guide, paragraphs)
    )
    response = llm.invoke(prompt)
    question = extract_question_simple(response)

    top_5 = get_top_n_similarities(question, 5, [guide.value])
    top_5_indices = set(top_5["paragraph_id"].tolist())
    top_5_and_paragraph_indices = list(set(paragraphs).union(top_5_indices))

    sentences = get_sentences(guide, top_5_and_paragraph_indices)
    prompt = PromptTemplate.from_template(answer_prompt).format(
        sentences=numbered_string(sentences), question=question
    )
    response = llm.invoke(prompt)

    cited_sentences_indices = extract_citations(
        response.content, set(range(1, len(sentences) + 1))
    )
    chosen_sentences = [sentences[i - 1] for i in cited_sentences_indices]
    if has_only_valid_citations(chosen_sentences):
        # is it a valid question?
        if "v." in question or ".?" in question:
            print(f"Invalid question: {question}")
            return None, None, None

        answer = json.dumps(chosen_sentences)
        return question, answer, top_5_and_paragraph_indices
    print("Invalid citations found in the answer")
    return None, None, None


def get_score(response: str, score: str):
    match = re.search(rf"{score} Score:.*?\s*.*?(\d+)", response, re.DOTALL)
    if not match:
        raise ValueError(f"Score not found in the response: {response}")
    return int(match.group(1))


FILTERING_PROMPT = """
You are a strict legal expert judging ECHR legal question-answer pairs. The answer might be bad, so be strict!

Question: {question}
Potential Answer: {answer}

You MUST answer each question in full sentences!

The response MUST follow this template:
Comprehensiveness Analysis: {{Go through the answer and analyze how well it answers the question. Does is cover all angles of the question? If the question is not a proper question or not a generic question (mentions a specific case), give a score of 1.}}
Comprehensiveness Score: {{A score from 1 (not comprehensive at all) to 5 (extremely comprehensive)}}
Conciseness: {{Is there any part in the answer irrelevant / unrelated to the question? If so, what is unneeded?}}
Conciseness Score: {{A score from 1 (not concise at all) to 5 (extremely concise)}}
Answer Fluency: {{Are there any bad sentence transitions in the answer? Are the sentences ordered correctly? Does the answer start with text clearly continuing previous text that is not there?}}
Answer Fluency Score: {{A score from 1 (not fluent) to 5 (perfectly fluent)}}
"""


def is_quality_pair(question: str, answer: str):
    temp_0_llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
    prompt = PromptTemplate.from_template(FILTERING_PROMPT).format(
        question=question, answer=answer
    )

    result = temp_0_llm.invoke(prompt)

    llm_fluency = get_score(result.content, "Answer Fluency")
    llm_conciseness = get_score(result.content, "Conciseness")
    llm_comprehensiveness = get_score(result.content, "Comprehensiveness")

    llm_annotation = min(llm_fluency, llm_conciseness, llm_comprehensiveness)

    return llm_annotation >= 4


df = pd.read_csv("data/echr_qa_dataset_v4.csv")
old_df = df.copy()


def is_complete(guide: Guide, paragraphs: list[int]):
    for _, row in old_df.iterrows():
        row_paragraphs = [
            int(paragraph)
            for paragraph in row["paragraphs"]
            .replace("[", "")
            .replace("]", "")
            .split(",")
        ]
        if row["guide"] == guide.value and all(
            elem in row_paragraphs for elem in paragraphs
        ):
            return True
    return False


def get_tasks():
    print("Getting tasks...")
    tasks = []
    completed_tasks = []
    invalid_tasks = []

    for guide_id in list(Guide):
        num_paragraphs = len(get_guide(guide_id))
        for i in range(0, num_paragraphs - 3, 2):
            paragraphs = [i + 1, i + 2, i + 3]
            if is_complete(guide_id, paragraphs):
                completed_tasks.append((guide_id, paragraphs))
                continue

            all_usable = all([is_all_usable_paragraph(p, guide_id) for p in paragraphs])
            if not all_usable:
                invalid_tasks.append((guide_id, paragraphs))
                continue
            tasks.append((guide_id, paragraphs))

    print(f"Completed tasks: {len(completed_tasks)}")
    print(f"Remaining tasks: {len(tasks)}")
    print(f"Invalid tasks: {len(invalid_tasks)}")
    return tasks


def try_generate(guide_id: Guide, paragraphs: list[int]):
    try:
        question, answer, pars = question_generation_with_search(
            guide_id, paragraphs, QG_LEGAL_2_1, QG_LEGAL_2_2
        )
        if question and is_quality_pair(question, answer):
            return {
                "guide": guide_id.value,
                "paragraphs": pars,
                "question": question,
                "answer": answer,
                "prompt_id": "legal-sentence-level-cot-with-search-v2",
            }

        question, answer, pars = question_generation_with_search(
            guide_id, paragraphs, QG_LEGAL_1_1, QG_LEGAL_1_2
        )
        if question and is_quality_pair(question, answer):
            return {
                "guide": guide_id.value,
                "paragraphs": pars,
                "question": question,
                "answer": answer,
                "prompt_id": "legal-sentence-level-cot-with-search-v1",
            }

        question, answer, pars = question_generation_with_search(
            guide_id, paragraphs, QG_WITH_SEARCH_1, QG_WITH_SEARCH_2
        )
        if question and is_quality_pair(question, answer):
            return {
                "guide": guide_id.value,
                "paragraphs": pars,
                "question": question,
                "answer": answer,
                "prompt_id": "sentence-level-cot-with-search",
            }
        return {"error": f"No good qa pair found for {guide_id} {paragraphs}"}
    except Exception as e:
        return {"error": str(e)}


def main():
    with concurrent.futures.ProcessPoolExecutor() as executor:
        tasks = get_tasks()
        global df
        while tasks:
            next_tasks = tasks[:20]
            tasks = tasks[20:]

            # sleep 1 minute to avoid OpenAI API rate limit
            time.sleep(60)

            futures = [
                executor.submit(try_generate, guide_id, paragraphs)
                for guide_id, paragraphs in next_tasks
            ]
            try:
                for future in concurrent.futures.as_completed(futures, timeout=600):
                    result = future.result(timeout=300)

                    if not result.get("error"):
                        # Append results to DataFrame and save to CSV
                        df = df._append(result, ignore_index=True)
                        df.to_csv("data/echr_qa_dataset_v4.csv", index=False)
                        print(
                            f"Added qa pair for {result['guide']} {result['paragraphs']}"
                        )
                    else:
                        print(result["error"])
            except concurrent.futures.TimeoutError:
                print("Task exceeded the time limit and was aborted.")
            print(f"Tasks remaining: {len(tasks)}")
            print(f"Completed tasks: {len(df)}")


if __name__ == "__main__":
    main()
