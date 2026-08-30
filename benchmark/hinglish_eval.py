import csv
import re
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from jiwer import wer, cer

BASE = "/mnt/d/Final_year_project"

INPUT = f"{BASE}/benchmark/results/dataset_results.csv"
OUTPUT = f"{BASE}/benchmark/results/hinglish_analysis.csv"


def clean(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def romanize_hindi(text):
    """
    Convert Devanagari to Roman script.
    English/Latin words remain unchanged.
    """

    text = clean(text)

    return transliterate(
        text,
        sanscript.DEVANAGARI,
        sanscript.ITRANS
    ).lower()


rows = []

with open(INPUT, encoding="utf-8") as f:

    reader = csv.DictReader(f)

    for r in reader:

        if r["dataset"] != "MUCS":
            continue

        reference = clean(r["reference"])
        prediction = clean(r["prediction"])

        ref_roman = romanize_hindi(reference)
        pred_roman = romanize_hindi(prediction)

        w = wer(ref_roman, pred_roman)
        c = cer(ref_roman, pred_roman)

        rows.append([
            r["utterance_id"],
            reference,
            prediction,
            ref_roman,
            pred_roman,
            w,
            c
        ])


with open(
    OUTPUT,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "utterance_id",
        "reference",
        "prediction",
        "reference_roman",
        "prediction_roman",
        "romanized_WER",
        "romanized_CER"
    ])

    writer.writerows(rows)


print("=" * 70)
print("HINGLISH CODE-SWITCH ANALYSIS")
print("=" * 70)

if rows:

    avg_wer = sum(r[5] for r in rows) / len(rows)
    avg_cer = sum(r[6] for r in rows) / len(rows)

    print(f"Samples: {len(rows)}")
    print(f"Romanized WER: {avg_wer:.4f}")
    print(f"Romanized CER: {avg_cer:.4f}")

print()
print("Saved:")
print(OUTPUT)
