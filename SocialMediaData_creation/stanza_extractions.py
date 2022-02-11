from typing import List
import pandas as pd
import stanza
from tqdm import tqdm
import collections

stanza.download("en", package="mimic", processors={"ner": "i2b2"})
nlp = stanza.Pipeline("en", package="mimic", processors={"ner": "i2b2"})


def get_entity_frequency(
    problems: List, treatments: List, tests: List, output_name: str
):
    """
    Get frequency counts of symptoms, treatments, and tests given
    lists of entities extracted by stanza.

    Args:
        problems: List of symptoms extracted by Stanza
        treatments: List of treatments extracted by Stanza
        tests: List of tests extracted by Stanza
        output_name: Output filename to save frequencies in an excel sheet

    """
    count_problems = collections.Counter(problems)
    count_tests = collections.Counter(tests)
    count_treatments = collections.Counter(treatments)

    df_problems = pd.DataFrame(
        count_problems.items(), columns=["problem", "frequency"]
    )
    df_treats = pd.DataFrame(
        count_treatments.items(), columns=["treatment", "frequency"]
    )
    df_test = pd.DataFrame(count_tests.items(), columns=["test", "frequency"])

    writer = pd.ExcelWriter(output_name + "_freq.xlsx", engine="xlsxwriter")
    df_problems.to_excel(writer, sheet_name="PROBLEMS")
    df_treats.to_excel(writer, sheet_name="TREATMENTS")
    df_test.to_excel(writer, sheet_name="TESTS")

    writer.save()


def extract_entities(df: pd.DataFrame, col_name: str, output_name: str):
    """
    Extract symptoms (problems), treatments, and test extractions with
    Stanza i2b2 NER model and get frequencies of extracted items.

    Args:
        df: Input dataframe containing column with textual data
        col_name: Column name of text data to extract entities from
        output_name: Output filename to save extractions and frequencies

    Returns:
        problems: List of symptoms extracted by Stanza
        treatments: List of treatments extracted by Stanza
        tests: List of tests extracted by Stanza
        df: Output dataframe containing columns with entity extractions

    """
    problems = []
    treatments = []
    tests = []

    df_problems = []
    df_treatments = []
    df_tests = []
    df_vals = []

    for i in tqdm(df[col_name].tolist(), desc="Extracting entities..."):
        doc = nlp(i)

        doc_problems = []
        doc_treatments = []
        doc_tests = []

        for ent in doc.entities:
            if ent.type == "PROBLEM":
                problems.append(ent.text)
                doc_problems.append(ent.text)
            elif ent.type == "TREATMENT":
                treatments.append(ent.text)
                doc_treatments.append(ent.text)
            elif ent.type == "TEST":
                tests.append(ent.text)
                doc_tests.append(ent.text)

        df_problems.append(doc_problems)
        df_treatments.append(doc_treatments)
        df_tests.append(doc_tests)
        df_vals.append(doc.entities)

    df["Entities"] = df_vals
    df["Symptoms"] = df_problems
    df["Tests"] = df_tests
    df["Treatments"] = df_treatments

    df.to_pickle(output_name + ".p")

    get_entity_frequency(problems, treatments, tests, output_name)

    return problems, treatments, tests, df


if __name__ == "__main__":

    # read in input data into a dataframe
    data = pd.read_pickle("reddit_covidlonghaulers.p")

    # specify column containing text (col_name)
    symptoms, treatments, tests, df = extract_entities(
        df=data, col_name="cleaned_text", output_name="stanza_output"
    )
   
    df.to_csv("stanza_extractions.csv")
