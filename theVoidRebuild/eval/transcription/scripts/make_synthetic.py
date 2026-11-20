#!/usr/bin/env python3
"""Generate synthetic smoke-test audio with macOS say. No real recordings."""

import argparse
import datetime
import json
import pathlib
import platform
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice", default="Samantha", help="An installed macOS say voice")
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "synthetic"
    audio = root / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    clips = [
        ("synthetic-01-reminder.aiff", 175,
         "Remind me to buy coffee and call my sister after lunch tomorrow."),
        ("synthetic-02-reflection.aiff", 145,
         "I felt distracted this morning. A short walk helped me slow down and focus on one thing at a time."),
        ("synthetic-03-plan.aiff", 205,
         "Let's keep the first version simple. Record the idea, save the transcript, and add a title. We can review the details on Friday."),
    ]
    references = []
    commands = []
    for filename, rate, text in clips:
        target = audio / filename
        command = ["/usr/bin/say", "-v", args.voice, "-r", str(rate), "-o", str(target), text]
        subprocess.run(command, check=True, timeout=60)
        if not target.is_file() or target.stat().st_size <= 0:
            raise RuntimeError(f"say produced no audio: {target}")
        # say can exit 0 but write only a header when speech-service access is
        # restricted. Verify actual audio data before accepting a fixture.
        info = subprocess.run(["/usr/bin/afinfo", str(target)], check=True,
                              capture_output=True, text=True, timeout=10).stdout
        audio_bytes = re.search(r"^audio bytes:\s+(\d+)", info, re.MULTILINE)
        duration = re.search(r"^estimated duration:\s+([\d.]+)", info, re.MULTILINE)
        if not audio_bytes or int(audio_bytes.group(1)) == 0 or not duration or float(duration.group(1)) <= 0:
            raise RuntimeError(f"say wrote no audio frames: {target}. Check voice availability and system speech-service access.")
        references.append({"file": filename, "text": text, "source": "synthetic"})
        commands.append({"file": filename, "voice": args.voice, "wordsPerMinute": rate,
                         "durationSeconds": float(duration.group(1)), "text": text})
        print(f"SYNTHETIC: {target.relative_to(root)} ({target.stat().st_size} bytes)")
    (root / "reference.jsonl").write_text(
        "".join(json.dumps(reference, ensure_ascii=False) + "\n" for reference in references), encoding="utf-8"
    )
    (root / "generation.json").write_text(json.dumps({
        "source": "synthetic",
        "generator": "/usr/bin/say",
        "generatedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "macOS": platform.mac_ver()[0],
        "format": "say default AIFF, no recording or microphone",
        "clips": commands,
        "limitation": "Text-to-speech smoke tests only; not evidence of real-world transcription accuracy.",
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
