"""
fusion_agent.py

Decision / Fusion Agent: deterministically merges the Emergency Agent's
extraction with the Service Agent's KB-grounded recommendation into the
final structured result. Deliberately rule-based (not another LLM call)
so the merge logic is transparent, fast, and auditable — the two LLM
agents have already done the language understanding; this step just
combines two structured JSON objects.
"""

SEVERITY_ORDER = {"unknown": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


def _more_urgent(sev_a, sev_b):
    a = SEVERITY_ORDER.get(sev_a, 0)
    b = SEVERITY_ORDER.get(sev_b, 0)
    return sev_a if a >= b else sev_b


def fuse(transcript: str, extraction: dict, service_rec: dict) -> dict:
    extracted_service = extraction.get("required_service", "unknown")
    recommended_service = service_rec.get("recommended_service", "unknown")

    # Service decision:
    #  - if they agree (or one is unknown), take the concrete one
    #  - if both concrete but different, dispatch "multiple" and note the
    #    disagreement rather than silently picking one
    if extracted_service == recommended_service:
        final_service = extracted_service
        service_conflict = False
    elif extracted_service == "unknown":
        final_service = recommended_service
        service_conflict = False
    elif recommended_service == "unknown":
        final_service = extracted_service
        service_conflict = False
    else:
        final_service = "multiple"
        service_conflict = True

    # Severity decision: take the more urgent of the extractor's severity
    # and any adjustment the service agent proposed (grounded in the KB
    # protocol note) — never move to LESS urgent than the extractor said.
    extracted_sev = extraction.get("severity", "unknown")
    sev_adjustment = service_rec.get("severity_adjustment")
    final_severity = _more_urgent(extracted_sev, sev_adjustment) if sev_adjustment else extracted_sev

    additional_info_parts = []
    if extraction.get("additional_information"):
        additional_info_parts.append(str(extraction["additional_information"]))
    if service_rec.get("reasoning"):
        additional_info_parts.append(f"Service agent: {service_rec['reasoning']}")
    if service_conflict:
        additional_info_parts.append(
            f"Note: extractor suggested '{extracted_service}', service agent "
            f"recommended '{recommended_service}' — dispatching both."
        )
    additional_information = " | ".join(additional_info_parts) if additional_info_parts else None

    return {
        "transcription": transcript,
        "emergency_type": extraction.get("emergency_type", "unknown"),
        "severity": final_severity,
        "location": extraction.get("location"),
        "people_involved": extraction.get("people_involved"),
        "injuries": extraction.get("injuries"),
        "additional_information": additional_information,
        "recommended_service": final_service,
        "agent_agreement": not service_conflict,
        "kb_source_category": service_rec.get("source_category"),
        "kb_fallback_search_used": service_rec.get("used_fallback_kb_search", False),
    }
