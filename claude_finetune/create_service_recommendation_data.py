import json
import re
from pathlib import Path
from collections import Counter

BASE = Path("emergency_dataset/emergency_dataset/splits")
OUT = Path("emergency_dataset/emergency_dataset/service_recommendation")

OUT.mkdir(parents=True, exist_ok=True)

SERVICES = {
    "ambulance",
    "police",
    "fire_brigade",
    "rescue",
    "unknown",
}

# Explicit service requests in the transcript take priority.
EXPLICIT_SERVICE = {
    "ambulance": [
        "ambulance",
        "paramedic",
        "medical team",
        "medical help",
        "emergency medical",
    ],
    "police": [
        "police",
        "police ko",
        "police bhejo",
        "police bulao",
        "police help",
    ],
    "fire_brigade": [
        "fire brigade",
        "fire department",
        "fire service",
        "fire truck",
    ],
    "rescue": [
        "rescue",
        "rescue team",
        "rescue bhejo",
    ],
}


def normalize(text):
    return text.lower().strip()


def contains_any(text, phrases):
    return any(p in text for p in phrases)


def recommend_service(item):
    """
    Service recommendation policy.

    1. Explicit service request is strong evidence.
    2. Otherwise infer from the emergency situation.
    3. Fire without injury -> fire_brigade.
    4. Fire + explicit injury/medical problem -> ambulance.
    5. Personal safety/security/missing/domestic violence -> police.
    6. Trapped/flooding/natural disaster -> rescue.
    7. Medical/accident/injury -> ambulance.
    8. Insufficient information -> unknown.
    """

    messages = item.get("messages")

    if messages is None:
        transcript = item.get("transcript") or item.get("text")

        if transcript is None:
            raise ValueError(
                f"Could not find transcript/messages in record: {item}"
            )

        messages = [
            {"role": "user", "content": transcript}
        ]

    transcript = ""
    emergency_type = ""
    injuries = ""

    for m in messages:
        if m["role"] == "user":
            transcript = m["content"]

    # Extract original metadata if available.
    metadata = item.get("metadata", {})

    # Some datasets contain the structured source information in metadata.
    emergency_type = normalize(
        metadata.get("emergency_type", "")
    )

    injuries = normalize(
        metadata.get("injuries", "")
    )

    text = normalize(transcript)

    # ---------------------------------------------------------
    # 1. EXPLICIT SERVICE REQUEST
    # ---------------------------------------------------------

    for service, phrases in EXPLICIT_SERVICE.items():
        if contains_any(text, phrases):
            return service, "explicit"

    # ---------------------------------------------------------
    # 2. FIRE
    # ---------------------------------------------------------

    fire_words = [
        "fire",
        "aag",
        "burning",
        "jal gayi",
        "jal gaya",
        "smoke",
        "dhuaan",
        "gas cylinder",
    ]

    injury_words = [
        "injured",
        "injury",
        "hurt",
        "bleeding",
        "bleed",
        "burn hua",
        "burned",
        "burnt",
        "chot",
        "wound",
        "pain",
        "breathing problem",
        "saans lene mein problem",
        "unconscious",
        "respond nahi",
    ]

    has_fire = contains_any(text, fire_words)
    has_injury = (
        contains_any(text, injury_words)
        or injuries not in ("", "null", "none")
    )

    if has_fire:
        # Fire + injured person -> medical response.
        if has_injury:
            return "ambulance", "injury"

        # Fire without injury -> fire brigade.
        return "fire_brigade", "fire"

    # ---------------------------------------------------------
    # 3. POLICE / PERSONAL SAFETY
    # ---------------------------------------------------------

    police_words = [
        "follow kar raha",
        "peecha kar raha",
        "threaten",
        "threatening",
        "threat",
        "dar lag raha",
        "safe feel nahi",
        "security problem",
        "security issue",
        "fight",
        "lad rahe",
        "violence",
        "violent",
        "attack",
        "ghar mein violence",
        "missing",
        "mil nahi rahi",
        "mil nahi raha",
        "pata nahi chal",
        "wapas nahi aaya",
        "wapas nahi aayi",
    ]

    if contains_any(text, police_words):
        return "police", "safety"

    police_types = {
        "personal_safety",
        "police_security",
        "domestic_emergency",
        "missing_person",
    }

    if emergency_type in police_types:
        return "police", "type"

    # ---------------------------------------------------------
    # 4. RESCUE
    # ---------------------------------------------------------

    rescue_words = [
        "flood",
        "flooding",
        "paani bhar",
        "paani mein doob",
        "road paani",
        "trapped",
        "stuck",
        "phans gaya",
        "phans gaye",
        "phans gayi",
        "phase gaye",
        "lift mein",
        "basement mein",
        "rescue",
        "building collapse",
        "collapse",
    ]

    if contains_any(text, rescue_words):
        return "rescue", "situation"

    rescue_types = {
        "trapped_person",
        "natural_disaster",
    }

    if emergency_type in rescue_types:
        return "rescue", "type"

    # ---------------------------------------------------------
    # 5. MEDICAL / ACCIDENT
    # ---------------------------------------------------------

    medical_words = [
        "accident",
        "crash",
        "collision",
        "takra",
        "injured",
        "injury",
        "bleeding",
        "bleed",
        "chot",
        "hurt",
        "breathing problem",
        "saans lene mein problem",
        "chest pain",
        "heart pain",
        "severe pain",
        "unconscious",
        "behosh",
        "hospital",
        "doctor",
        "medical emergency",
    ]

    if contains_any(text, medical_words):
        return "ambulance", "medical"

    medical_types = {
        "accident",
        "medical_emergency",
    }

    if emergency_type in medical_types:
        return "ambulance", "type"

    # ---------------------------------------------------------
    # 6. UNKNOWN
    # ---------------------------------------------------------

    return "unknown", "uncertain"


def convert_split(split):
    source = BASE / f"{split}.jsonl"
    destination = OUT / f"{split}.jsonl"

    counts = Counter()
    evidence_counts = Counter()

    output = []

    with open(source, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            service, evidence = recommend_service(item)

            transcript = ""
            for m in item.get("messages", [
        {"role": "user", "content": item.get("transcript", item.get("text", ""))}
    ]):
                if m["role"] == "user":
                    transcript = m["content"]

            new_item = {
                "messages": [
                    {
                        "role": "system",
                        "content": """You are an emergency service recommendation agent.

Determine the emergency service that should be dispatched based on
the caller's complete situation.

Choose exactly one:
ambulance
police
fire_brigade
rescue
unknown

Infer the appropriate emergency response even when the caller does
not explicitly request a service.

Rules:
- Medical emergencies, accidents, injuries, bleeding or breathing problems -> ambulance.
- Fire without an injured person -> fire_brigade.
- Fire with an injured person or medical emergency -> ambulance.
- Following, threats, fights, violence, domestic danger or missing persons -> police.
- Flooding, trapped people and rescue situations -> rescue.
- If there is insufficient information to determine a service -> unknown.

Return ONLY valid JSON:
{"required_service": "..."}"""
                    },
                    {
                        "role": "user",
                        "content": transcript
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {"required_service": service},
                            ensure_ascii=False
                        )
                    }
                ],
                "metadata": {
                    "recommendation_evidence": evidence,
                    "original_required_service":
                        item.get("metadata", {}).get(
                            "original_required_service",
                            "unknown"
                        )
                }
            }

            output.append(new_item)
            counts[service] += 1
            evidence_counts[evidence] += 1

    with open(destination, "w", encoding="utf-8") as f:
        for item in output:
            f.write(
                json.dumps(item, ensure_ascii=False) + "\n"
            )

    print(f"\n{split}: {len(output)}")
    print("Service distribution:")
    for service, count in counts.most_common():
        print(f"  {service:15} {count}")

    print("Evidence:")
    for evidence, count in evidence_counts.most_common():
        print(f"  {evidence:15} {count}")


for split in ["train", "validation", "test"]:
    convert_split(split)

print("\nCreated:")
print(OUT)
