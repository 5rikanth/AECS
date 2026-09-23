import json
from pathlib import Path

BASE = Path("emergency_dataset/emergency_dataset/splits")
OUT = Path("emergency_dataset/emergency_dataset/instruction_data")
OUT.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = """You are an emergency-call information extraction system.

Extract only information explicitly stated or clearly expressed in the caller's transcript.

Return ONLY valid JSON with exactly these fields:
emergency_type
severity
required_service
location
injuries
people_involved
additional_information

Use null when information is not available.
Do not invent or infer missing information."""

def convert(input_file, output_file):
    count = 0

    with open(BASE / input_file, encoding="utf-8") as f_in, \
         open(OUT / output_file, "w", encoding="utf-8") as f_out:

        for line in f_in:
            item = json.loads(line)

            answer = {
                "emergency_type": item["emergency_type"],
                "severity": item["severity"],
                "required_service": item["required_service"],
                "location": item["location"],
                "injuries": item["injuries"],
                "people_involved": item["people_involved"],
                "additional_information": item["additional_information"]
            }

            example = {
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": item["transcript"]
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            answer,
                            ensure_ascii=False
                        )
                    }
                ]
            }

            f_out.write(
                json.dumps(example, ensure_ascii=False) + "\n"
            )

            count += 1

    print(f"{output_file}: {count}")

convert("train.jsonl", "train.jsonl")
convert("validation.jsonl", "validation.jsonl")
convert("test.jsonl", "test.jsonl")
