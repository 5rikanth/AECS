import json
from pathlib import Path

BASE = Path("emergency_dataset/emergency_dataset/splits")
OUT = Path("emergency_dataset/emergency_dataset/agents")

AGENTS = {
    "emergency_type": "emergency_type",
    "severity": "severity",
    "required_service": "required_service",
    "location": "location",
    "injuries": "injuries",
    "people_involved": "people_involved",
    "additional_information": "additional_information",
}

SYSTEM = """You are an emergency-call information extraction agent.

Extract ONLY the requested field from the caller's transcript.

Use null when the requested information is not explicitly stated.
Do not infer, assume, or add information that is not present.

Return ONLY valid JSON with exactly the requested field."""

for agent, field in AGENTS.items():
    agent_dir = OUT / agent
    agent_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "validation", "test"]:
        src = BASE / f"{split}.jsonl"
        dst = agent_dir / f"{split}.jsonl"

        count = 0

        with open(src, encoding="utf-8") as fin, \
             open(dst, "w", encoding="utf-8") as fout:

            for line in fin:
                x = json.loads(line)

                target = {field: x.get(field)}

                messages = [
                    {
                        "role": "system",
                        "content": SYSTEM
                    },
                    {
                        "role": "user",
                        "content": x["transcript"]
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            target,
                            ensure_ascii=False
                        )
                    }
                ]

                fout.write(
                    json.dumps(
                        {"messages": messages},
                        ensure_ascii=False
                    ) + "\n"
                )

                count += 1

        print(f"{agent:25} {split:10} {count}")

print("\nCreated all agent datasets.")
