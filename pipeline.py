"""
Speech Annotation Pipeline
- Speaker Diarization via pyannote.audio
- Transcription via openai-whisper (CPU-friendly, no av dependency)
"""

import os
import json
import csv
import time
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

import torch
import whisper
from pyannote.audio import Pipeline as PyannotePipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")


@dataclass
class AnnotationSegment:
    speaker: str
    start: float
    end: float
    text: str

    @property
    def start_fmt(self):
        m, s = divmod(int(self.start), 60)
        return f"{m:02d}:{s:02d}"

    @property
    def end_fmt(self):
        m, s = divmod(int(self.end), 60)
        return f"{m:02d}:{s:02d}"

    def to_abab(self):
        return f"{self.speaker} {self.start_fmt} {self.text}"


def _build_speaker_map(raw_labels):
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    seen, mapping = [], {}
    for lbl in raw_labels:
        if lbl not in seen:
            seen.append(lbl)
    for i, lbl in enumerate(seen):
        mapping[lbl] = letters[i] if i < len(letters) else f"SPK{i}"
    return mapping


class SpeechAnnotationPipeline:
    def __init__(self):
        self._whisper_model = None
        self._diarization_pipeline = None

    def _load_whisper(self):
        if self._whisper_model is None:
            logger.info(f"Loading Whisper model '{WHISPER_MODEL}' on {DEVICE}...")
            self._whisper_model = whisper.load_model(WHISPER_MODEL, device=DEVICE)
        return self._whisper_model

    def _load_diarization(self):
        if self._diarization_pipeline is None:
            if not HF_TOKEN:
                raise EnvironmentError(
                    "HF_TOKEN environment variable is required. "
                    "Set it in Space Settings > Variables and secrets."
                )
            logger.info("Loading pyannote diarization pipeline...")
            self._diarization_pipeline = PyannotePipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=HF_TOKEN,
            )
            if DEVICE == "cuda":
                self._diarization_pipeline = self._diarization_pipeline.to(torch.device("cuda"))
        return self._diarization_pipeline

    def _transcribe(self, audio_path):
        logger.info("Step 1/2 - Transcribing with Whisper...")
        model = self._load_whisper()
        result = model.transcribe(audio_path, verbose=False)
        return result["segments"]

    def _diarize(self, audio_path):
        logger.info("Step 2/2 - Diarizing speakers...")
        pipeline = self._load_diarization()
        diarization = pipeline(audio_path)
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})
        return segments

    def _assign_speakers(self, transcript_segments, diarization_segments):
        raw_labels = [d["speaker"] for d in diarization_segments]
        speaker_map = _build_speaker_map(raw_labels)
        results = []
        for seg in transcript_segments:
            t_start, t_end = seg["start"], seg["end"]
            text = seg.get("text", "").strip()
            if not text:
                continue
            best_speaker, best_overlap = "A", 0.0
            for d in diarization_segments:
                overlap = max(0, min(t_end, d["end"]) - max(t_start, d["start"]))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = d["speaker"]
            results.append(AnnotationSegment(
                speaker=speaker_map.get(best_speaker, best_speaker),
                start=t_start, end=t_end, text=text,
            ))
        return results

    def process(self, audio_path):
        t0 = time.time()
        transcript = self._transcribe(audio_path)
        diarization = self._diarize(audio_path)
        annotations = self._assign_speakers(transcript, diarization)
        logger.info(f"Done in {time.time()-t0:.1f}s — {len(annotations)} segments")
        return annotations


def to_json(segments, path):
    data = [{"speaker": s.speaker, "start": round(s.start,3), "end": round(s.end,3),
              "start_fmt": s.start_fmt, "end_fmt": s.end_fmt, "text": s.text} for s in segments]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def to_csv(segments, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["speaker", "start", "end", "text"])
        for s in segments:
            writer.writerow([s.speaker, s.start_fmt, s.end_fmt, s.text])

def to_abab_text(segments):
    return "\n".join(s.to_abab() for s in segments)

_pipeline_instance = None

def get_pipeline():
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = SpeechAnnotationPipeline()
    return _pipeline_instance
