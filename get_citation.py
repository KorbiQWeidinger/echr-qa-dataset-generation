import json
import sqlite3
from string import Template

import requests


def find_paragraph(docs: list[dict], paragraph: int):
    def search_paragraph(doc, paragraph):
        if doc["content"].startswith(f"{paragraph}."):
            return doc["content"]
        for e in doc["elements"]:
            res = search_paragraph(e, paragraph)
            if res:
                return res
        return None

    for doc in docs:
        res = search_paragraph(doc, paragraph)
        if res:
            return res
    raise ValueError(f"Paragraph {paragraph} not found.")


def get_case_name_by_id(db_path, case_id):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)

    try:
        # Create a cursor object using the cursor() method
        cursor = conn.cursor()

        # Prepare SQL query to retrieve entries that are similar to the given docname
        query = 'SELECT * FROM "case" WHERE itemid = ?'
        params = (case_id,)

        # Executing the SQL command
        cursor.execute(query, params)

        columns = [description[0] for description in cursor.description]

        # Fetch all rows from the last executed statement
        results = cursor.fetchall()

        if not results:
            raise ValueError("No results found")

        if len(results) > 1:
            raise ValueError("Multiple entries found for the given case_id")

        row = results[0]
        for col, val in zip(columns, row):
            if col == "docname":
                return val

        raise ValueError("Case name not found.")

    except sqlite3.Error as error:
        raise error
    finally:
        # Closing the connection
        if conn:
            conn.close()


def get_citation(
    db_path,
    docname: str,
    paragraph: int,
    year: str = None,
):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)

    try:
        # Create a cursor object using the cursor() method
        cursor = conn.cursor()

        # Prepare SQL query to retrieve entries that are similar to the given docname
        query = 'SELECT * FROM "case" WHERE docname LIKE ?'
        fuzzy_docname = f"%{docname}%"
        params = (fuzzy_docname,)

        if year:
            query += " AND judgementdate LIKE ?"
            fuzzy_year = f"%{year}%"
            params = params + (fuzzy_year,)

        # Executing the SQL command
        cursor.execute(query, params)

        columns = [description[0] for description in cursor.description]

        # Fetch all rows from the last executed statement
        results = cursor.fetchall()

        if not results:
            raise ValueError("No results found")

        if len(results) > 1:
            raise ValueError("Multiple entries found for the given docname, year")

        row = results[0]
        for col, val in zip(columns, row):
            if col == "judgment":
                val = json.loads(val)
                paragraph = find_paragraph(val, paragraph)
                return paragraph

        raise ValueError("Paragraph not found.")

    except sqlite3.Error as error:
        raise error
    finally:
        # Closing the connection
        if conn:
            conn.close()


def attempt_to_get_citation(case_name: str, paragraph: int, year: str = None):
    db_path = "data/echr_2_0_0.db"
    docname = "CASE OF " + case_name.upper()
    return get_citation(db_path, docname, paragraph, year)


def attempt_to_get_case_name(case_id: int):
    db_path = "data/echr_2_0_0.db"
    return get_case_name_by_id(db_path, case_id)


print(attempt_to_get_case_name("001-114082"))

## what to do if case is not in db:
from bs4 import BeautifulSoup

from guide_parser import available_paragraphs


def get_paragraphs_for_case_id(case_id: str):
    url_template = Template(
        "https://hudoc.echr.coe.int/app/conversion/docx/html/body?library=ECHR&id=$case_id"
    )
    url = url_template.substitute(case_id=case_id)
    res = requests.get(url)
    data = res.text

    soup = BeautifulSoup(data, "html.parser")

    text = soup.get_text()
    n = available_paragraphs(text)

    paragraphs = {}
    for i in range(1, n):
        after = text.split(f"{i}.", 1)[1]
        paragraph = after.split(f"{i+1}.")[0]
        text = after.split(f"{i+1}.", 1)[1]
        text = f"{i+1}.{text}"
        paragraph = f"{i}.{paragraph}"
        paragraphs[i] = paragraph
    paragraphs[n] = text[0:600]
