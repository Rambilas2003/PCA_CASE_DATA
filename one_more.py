import pdfplumber
import spacy
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer

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

def tfidf_scores(sentences):
    only_s = [s for _, s in sentences]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(only_s)
    scores = matrix.sum(axis=1).A1
    return scores

def extract_keywords(sentence):
    doc = nlp(sentence)
    words = [
        token.text for token in doc 
        if token.pos_ in ("NOUN", "PROPN") 
        and not token.is_stop 
        and len(token.text) > 3
    ]
    return list(dict.fromkeys(words))[:7]

def extract_entities(sentence):
    doc = nlp(sentence)
    return [(ent.text, ent.label_) for ent in doc.ents]

def get_important_sentences(sentences, scores):
    avg = sum(scores) / len(scores)
    threshold = avg * 1.12  

    imp = []
    for (page, s), sc in zip(sentences, scores):
        if sc >= threshold:
            imp.append((page, s, sc))
    return imp

def detect_headings(text):
    lines = text.split("\n")
    heads = []
    for line in lines:
        if (line.isupper() and len(line) > 5) or ("SECTION" in line.upper()):
            heads.append(line.strip())
    return heads

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

def find_contradictions(sentences):
    contra_pairs = []
    neg_words = ["not", "never", "no", "didn't", "don't", "cannot"]

    for i in range(len(sentences) - 1):
        p1, s1 = sentences[i]
        p2, s2 = sentences[i + 1]

        if any(w in s1.lower() for w in neg_words) and not any(w in s2.lower() for w in neg_words):
            if any(word in s2.lower() for word in s1.lower().split()[:5]):
                contra_pairs.append((p1, s1, p2, s2))

    return contra_pairs

def page_summary(pages):
    summaries = []
    for page, text in pages:
        doc = nlp(text)
        sents = [sent.text.strip() for sent in doc.sents][:4]
        summaries.append((page, " ".join(sents)))
    return summaries

def extract_all_entities(sentences):
    ent_list = []
    for page, s in sentences:
        ents = extract_entities(s)
        for e in ents:
            ent_list.append((page, s, e[0], e[1]))
    return ent_list

def export_all(important, core, contradictions, page_summaries, entity_list, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    df1 = pd.DataFrame([{
        "Page": p,
        "Important Sentence": s,
        "Score": round(sc, 4),
        "Keywords": ", ".join(extract_keywords(s)),
        "Entities": extract_entities(s)
    } for p, s, sc in important])
    df1.to_csv(os.path.join(output_folder, "important_sentences.csv"), index=False)

    df2 = pd.DataFrame([{
        "Page": p,
        "Relevant Sentence": s
    } for p, s in core])
    df2.to_csv(os.path.join(output_folder, "case_core_points.csv"), index=False)

    df3 = pd.DataFrame([{
        "Page1": p1,
        "Statement1": s1,
        "Page2": p2,
        "Statement2": s2
    } for p1, s1, p2, s2 in contradictions])
    df3.to_csv(os.path.join(output_folder, "contradictions.csv"), index=False)

    df4 = pd.DataFrame([{
        "Page": p,
        "Summary": s
    } for p, s in page_summaries])
    df4.to_csv(os.path.join(output_folder, "page_wise_summary.csv"), index=False)

    df5 = pd.DataFrame([{
        "Page": p,
        "Sentence": s,
        "Entity": ent,
        "Type": typ
    } for p, s, ent, typ in entity_list])
    df5.to_csv(os.path.join(output_folder, "all_entities.csv"), index=False)

def analyze_case(pdf_path, output_folder):
    pages = extract_pdf_pages(pdf_path)
    sentences = split_sentences(pages)
    scores = tfidf_scores(sentences)
    important = get_important_sentences(sentences, scores)

    core = extract_case_core(sentences)
    contradictions = find_contradictions(sentences)
    page_summaries = page_summary(pages)
    entity_list = extract_all_entities(sentences)

    export_all(important, core, contradictions, page_summaries, entity_list, output_folder)

    print("\n✔ Analysis completed.")
    print("✔ All CSV files saved in:", output_folder)

pdf_path = input("Enter path of the dataset (PDF file): ").strip()
output_folder = input("Enter output folder path: ").strip()

analyze_case(pdf_path, output_folder)
