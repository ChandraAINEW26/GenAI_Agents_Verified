"""Audio -> DealPulse transcript, fully local.

Uses faster-whisper (install with `uv sync --extra audio`) instead of the
original's paid OpenAI Whisper API. Whisper gives timestamps but not speakers,
so an LLM labels each segment as rep or customer from what is said; check the
labels in the written transcript before trusting the analysis.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from agentkit import LLM


class SpeakerLabel(BaseModel):
    index: int
    role: str  # rep | customer


class SpeakerLabels(BaseModel):
    labels: list[SpeakerLabel]


def decode_audio(audio: Path, rate: int = 16000):
    """Decode any audio file to 16 kHz mono float32 with ffmpeg.

    faster-whisper's own decoder depends on PyAV, whose newer releases break its
    API; ffmpeg on PATH is simpler and reads every format.
    """
    import shutil
    import subprocess

    import numpy as np

    if not shutil.which("ffmpeg"):
        raise SystemExit("Audio needs ffmpeg on PATH (brew install ffmpeg).")
    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(audio), "-f", "f32le", "-ac", "1", "-ar", str(rate), "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32)


def transcribe_audio(audio: Path, llm: LLM, rep: str, account: str, model_size: str = "base") -> str:
    try:
        from faster_whisper import WhisperModel
    except ImportError as err:
        raise SystemExit("Audio needs faster-whisper: run `uv sync --extra audio`.") from err

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _info = model.transcribe(decode_audio(audio), vad_filter=True)
    segments = [(round(s.start), s.text.strip()) for s in segments if s.text.strip()]
    if not segments:
        raise SystemExit(f"No speech found in {audio}")

    numbered = "\n".join(f"{i}: {text}" for i, (_, text) in enumerate(segments))
    labels = llm.structured(
        "You label who is speaking in a sales call transcript that has no speaker names. The rep sells the "
        "product and asks discovery questions; the customer describes their business and raises concerns.",
        f"Label every numbered segment as 'rep' or 'customer'.\n\n{numbered}",
        SpeakerLabels,
        effort="low",
    )
    roles = {label.index: label.role for label in labels.labels}

    lines = ["---", f"call_id: {audio.stem}", f"rep: {rep}", f"account: {account}", "stage: Unknown",
             "speakers:", "  Rep: rep", "  Customer: customer", "---"]
    for i, (start, text) in enumerate(segments):
        speaker = "Rep" if roles.get(i) == "rep" else "Customer"
        lines.append(f"[{start // 60:02d}:{start % 60:02d}] {speaker}: {text}")
    return "\n".join(lines) + "\n"
