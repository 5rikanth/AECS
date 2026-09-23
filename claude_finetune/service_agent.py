"""
service_agent.py

Service Agent: recommends required_service (and can flag a severity
adjustment) by grounding the already-extracted structured info +
transcript against the knowledge_base/ protocol notes, via the SAME
already-loaded LLM instance used by the Emergency Agent
(emergency_processor.EmergencyExtractor) — no second model load, no
fine-tuning, no LoRA.

This replaces the collapsed LoRA-based "service_recommender_lora".
"""

import json
import re

from rag_retriever import load_kb, get_context

VALID_SERVICES = ["ambulance", "police", "fire_brigade", "rescue", "multiple", "unknown"]
VALID_SEVERITIES = ["low", "medium", "high", "critical", "unknown"]

SERVICE_SYSTEM_PROMPT = """You are the Service Recommendation component of an \
academic emergency-call prototype. You are given:
1. The original caller transcript (Hinglish, may mix Hindi/English).
2. A structured extraction already produced by another component.
3. A reference protocol note for the relevant emergency category.

Your job: recommend required_service and (if you think the extraction's \
severity is clearly wrong given the protocol note) a corrected severity, \
with a short reasoning grounded ONLY in the transcript and the protocol \
note — never invent facts not present in either.

required_service must be exactly one of: ambulance, police, fire_brigade, \
rescue, multiple, unknown.
severity must be exactly one of: low, medium, high, critical, unknown.

Output VALID JSON ONLY, no markdown fences, no extra text, with exactly \
these keys:
{"recommended_service": "...", "severity_adjustment": "... or null if you \
agree with the given severity", "reasoning": "one short sentence"}
"""


def _build_prompt(transcript, extraction, protocol_text):
    return (
        f'Transcript: "{transcript}"\n\n'
        f"Current extraction: {json.dumps(extraction, ensure_ascii=False)}\n\n"
        f"Protocol note for this category:\n{protocol_text}\n\n"
        f"JSON:"
    )


def _try_parse_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


class ServiceAgent:
    def __init__(self, extractor, kb_dir=None):
        """
        extractor: an already-constructed
                   emergency_processor.EmergencyExtractor instance
                   (its .model / .tokenizer / .torch are reused directly —
                   no new model is loaded here).
        """
        self.extractor = extractor
        self.kb = load_kb(kb_dir) if kb_dir else load_kb()

    def _generate(self, transcript, extraction, protocol_text) -> str:
        messages = [
            {"role": "system", "content": SERVICE_SYSTEM_PROMPT},
            {"role": "user", "content": _build_prompt(transcript, extraction, protocol_text)},
        ]
        tok = self.extractor.tokenizer
        model = self.extractor.model
        torch = self.extractor.torch

        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.05,
                pad_token_id=tok.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return tok.decode(new_tokens, skip_special_tokens=True).strip()

    def recommend(self, transcript: str, extraction: dict, max_retries: int = 2) -> dict:
        emergency_type = extraction.get("emergency_type", "unknown")
        source_category, protocol_text, was_fallback = get_context(
            emergency_type, transcript, self.kb
        )

        raw = self._generate(transcript, extraction, protocol_text)
        parsed = _try_parse_json(raw)

        attempt = 0
        while parsed is None and attempt < max_retries:
            attempt += 1
            raw = self._generate(
                transcript, extraction,
                protocol_text + "\n\nYour previous reply was not valid JSON. "
                "Output ONLY the JSON object.",
            )
            parsed = _try_parse_json(raw)

        if parsed is None:
            return {
                "recommended_service": "unknown",
                "severity_adjustment": None,
                "reasoning": "Service agent could not produce valid JSON.",
                "source_category": source_category,
                "used_fallback_kb_search": was_fallback,
                "_parse_error": True,
                "_raw_model_output": raw,
            }

        service = str(parsed.get("recommended_service", "unknown")).strip().lower().replace(" ", "_")
        if service not in VALID_SERVICES:
            service = "unknown"

        sev_adj = parsed.get("severity_adjustment", None)
        if isinstance(sev_adj, str):
            sev_adj = sev_adj.strip().lower().replace(" ", "_")
            if sev_adj in ("null", "none", ""):
                sev_adj = None
            elif sev_adj not in VALID_SEVERITIES:
                sev_adj = None

        return {
            "recommended_service": service,
            "severity_adjustment": sev_adj,
            "reasoning": parsed.get("reasoning"),
            "source_category": source_category,
            "used_fallback_kb_search": was_fallback,
        }
