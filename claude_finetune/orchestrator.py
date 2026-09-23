#!/usr/bin/env python3
"""
orchestrator.py

Wires the three agents together:

  transcript -> Emergency Agent (emergency_processor.EmergencyExtractor)
             -> Service Agent (service_agent.ServiceAgent, KB-grounded,
                                reuses the SAME loaded LLM, no LoRA)
             -> Fusion/Decision Agent (fusion_agent.fuse, deterministic)
             -> FINAL EMERGENCY RESULT

This file assumes emergency_processor.py (your existing, unmodified
extractor) sits in the same directory. It does not touch ASR, the
microphone pipeline, or emergency_processor.py itself.

Usage:
    python orchestrator.py --test
    python orchestrator.py                      # interactive REPL
    python orchestrator.py --once "transcript here"
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from emergency_processor import EmergencyExtractor, print_env_info, TEST_TRANSCRIPTS  # noqa: E402
from service_agent import ServiceAgent  # noqa: E402
from fusion_agent import fuse  # noqa: E402


class MultiAgentPipeline:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct", use_4bit=False):
        self.extractor = EmergencyExtractor(model_name, use_4bit=use_4bit)
        self.service_agent = ServiceAgent(self.extractor)

    def run(self, transcript: str) -> dict:
        extraction = self.extractor.extract(transcript)
        service_rec = self.service_agent.recommend(transcript, extraction)
        final = fuse(transcript, extraction, service_rec)
        return final


def print_result(result: dict):
    print("\n" + "=" * 70)
    print("FINAL EMERGENCY RESULT")
    print("=" * 70)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def run_tests(pipeline: MultiAgentPipeline):
    for i, transcript in enumerate(TEST_TRANSCRIPTS, 1):
        print(f"\n--- Test {i} ---")
        print(f"Transcript: {transcript}")
        t0 = time.time()
        result = pipeline.run(transcript)
        dt = time.time() - t0
        print_result(result)
        print(f"({dt:.1f}s)")


def repl(pipeline: MultiAgentPipeline):
    print("\nEnter a Hinglish transcript (or 'quit' to exit):")
    while True:
        try:
            transcript = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if transcript.lower() in ("quit", "exit", "q"):
            break
        if not transcript:
            continue
        t0 = time.time()
        result = pipeline.run(transcript)
        dt = time.time() - t0
        print_result(result)
        print(f"({dt:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description="AECI multi-agent emergency pipeline")
    parser.add_argument("--model", default=os.environ.get("EMERGENCY_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"))
    parser.add_argument("--4bit", dest="use_4bit", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--once", type=str, default=None)
    parser.add_argument("--skip-env-check", action="store_true")
    args = parser.parse_args()

    if not args.skip_env_check:
        print_env_info()

    pipeline = MultiAgentPipeline(args.model, use_4bit=args.use_4bit)

    if args.once:
        print_result(pipeline.run(args.once))
    elif args.test:
        run_tests(pipeline)
    else:
        repl(pipeline)


if __name__ == "__main__":
    main()
