import pdfplumber
import spacy
import pandas as pd
import os
nlp = spacy.load("en_core_web_sm")
def extract_pdf_pages(pdf_path):
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            txt = page.extract_text()
            if txt:
                pages.append((i, txt))
    return pages
def split_sentences(pages):
    sents = []
    for page, text in pages:
        doc = nlp(text)
        for sent in doc.sents:
            clean = sent.text.strip()
            if len(clean) > 12:
                sents.append((page, clean))
    return sents
def extract_case_core(sentences):
    keywords = [
        "FIR", "allegation", "accused", "charge", "offence", "offense",
        "victim", "statement", "witness", "testimony", "evidence",
        "crime", "injury", "recovered", "weapon", "motive", "arrest"
    ]
    core = []
    for page, s in sentences:
        if any(k.lower() in s.lower() for k in keywords):
            core.append((page, s))
    return core

def page_summary(pages):
    summaries = []
    for page, text in pages:
        doc = nlp(text)
        sents = [sent.text.strip() for sent in doc.sents][:4]
        summaries.append((page, " ".join(sents)))
    return summaries

def export_selected_outputs(core, page_summaries, out_path):

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    df_core = pd.DataFrame([
        {"Page": p, "Relevant Sentence": s}
        for p, s in core
    ])
    df_core.to_csv(os.path.join(out_path, "case_core_points.csv"), index=False)
    df_summary = pd.DataFrame([
        {"Page": p, "Summary": s}
        for p, s in page_summaries
    ])
    df_summary.to_csv(os.path.join(out_path, "page_wise_summary.csv"), index=False)
def analyze_case(pdf_path, output_folder):
    pages = extract_pdf_pages(pdf_path)
    sentences = split_sentences(pages)
    core = extract_case_core(sentences)
    page_summaries = page_summary(pages)
    export_selected_outputs(core, page_summaries, output_folder)
    print("case_core_points.csv saved at:", output_folder)
    print("page_wise_summary.csv saved at:", output_folder)
pdf_path = input("Enter path of the PDF dataset: ").strip()
output_folder = input("Enter output folder path: ").strip()
analyze_case(pdf_path, output_folder)
