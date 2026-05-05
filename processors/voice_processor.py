#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Voice Processor Module — Powered by Whisper (faster-whisper)

This module implements voice processing capabilities for StackMind
using OpenAI's Whisper model via the faster-whisper library for
high-accuracy, offline, GPU-accelerated speech-to-text transcription.
"""

import logging
import os
import tempfile
import time
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Default Whisper model size — balance between speed and accuracy
# Options: "tiny", "base", "small", "medium", "large-v3"
DEFAULT_MODEL_SIZE = "base"


class VoiceProcessor:
    """
    Voice processor powered by faster-whisper (local Whisper).

    Provides high-accuracy, offline, GPU-accelerated speech-to-text
    transcription with automatic language detection, word-level
    timestamps, and segment-level confidence scores.

    Attributes:
        llm: Optional language model for enhanced voice understanding
        settings: Application settings
        model: The loaded faster-whisper model instance
    """

    def __init__(self, llm=None, settings=None):
        """
        Initialize the Voice Processor.

        Args:
            llm: Optional language model for enhanced voice understanding
            settings: Optional application settings
        """
        self.llm = llm
        self.settings = settings
        self.model = None

        # Load the Whisper model eagerly so first transcription is fast
        self._load_model()

        logger.info("Voice processor initialized (faster-whisper)")

    # ------------------------------------------------------------------
    # Model Loading
    # ------------------------------------------------------------------

    def _load_model(self):
        """Load the faster-whisper model, preferring GPU when available."""
        try:
            from faster_whisper import WhisperModel

            model_size = DEFAULT_MODEL_SIZE
            if self.settings:
                model_size = (
                    self.settings.get("voice", "model_size") or model_size
                )

            # Detect best available device
            device, compute_type = self._detect_device()

            logger.info(
                f"Loading Whisper model '{model_size}' on "
                f"{device} (compute_type={compute_type})"
            )

            self.model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
            )

            logger.info("Whisper model loaded successfully")

        except ImportError:
            logger.error(
                "faster-whisper is not installed. "
                "Install it with: pip install faster-whisper"
            )
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")

    @staticmethod
    def _detect_device() -> Tuple[str, str]:
        """
        Detect the best available compute device.

        Returns:
            Tuple of (device, compute_type) — e.g. ("cuda", "float16")
        """
        try:
            import torch

            if torch.cuda.is_available():
                logger.info("CUDA GPU detected — using GPU acceleration")
                return "cuda", "float16"
        except ImportError:
            pass

        logger.info("No GPU detected — falling back to CPU (int8)")
        return "cpu", "int8"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, audio_path: str) -> Dict[str, Any]:
        """
        Process an audio file and extract transcription + metadata.

        Args:
            audio_path: Path to the audio file

        Returns:
            Dict with keys: content_type, file_path, metadata,
            transcription, segments, and optionally analysis.
        """
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return {
                "error": f"Audio file not found: {audio_path}",
                "content_type": "error",
            }

        if self.model is None:
            logger.error("Whisper model is not loaded")
            return {
                "error": "Whisper model is not loaded. "
                         "Install faster-whisper: pip install faster-whisper",
                "content_type": "error",
            }

        try:
            result = {
                "content_type": "voice",
                "file_path": audio_path,
                "metadata": self._extract_metadata(audio_path),
                "transcription": "",
                "segments": [],
            }

            # Run Whisper transcription
            transcription, segments, info = self._transcribe(audio_path)

            result["transcription"] = transcription
            result["segments"] = segments
            result["metadata"].update({
                "detected_language": info.get("language", "unknown"),
                "language_probability": round(
                    info.get("language_probability", 0.0), 4
                ),
                "duration_seconds": round(info.get("duration", 0.0), 2),
            })

            # Set content field for downstream compatibility
            result["content"] = transcription

            # Optional LLM-enhanced analysis
            if transcription and self.llm:
                result["analysis"] = self._analyze_transcription(transcription)

            logger.debug(f"Processed audio: {audio_path}")
            return result

        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return {
                "error": f"Error processing audio: {e}",
                "content_type": "error",
            }

    def record_audio(
        self, duration: int = 5, output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Record audio from the microphone and transcribe it.

        Args:
            duration: Duration in seconds to record
            output_path: Optional path to save the recorded audio

        Returns:
            Dict with transcription results
        """
        try:
            import sounddevice as sd
            import numpy as np
            import wave

            sample_rate = 16000  # Whisper expects 16 kHz

            if not output_path:
                output_path = os.path.join(
                    tempfile.gettempdir(),
                    f"stackmind_recording_{int(time.time())}.wav",
                )

            logger.info(f"Recording audio for {duration}s …")
            audio_data = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype="int16",
            )
            sd.wait()  # Block until recording is done

            # Save as WAV
            with wave.open(output_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes(audio_data.tobytes())

            logger.info(f"Audio saved to {output_path}")

            # Transcribe the recording
            return self.process(output_path)

        except ImportError:
            logger.error(
                "sounddevice/numpy not installed. "
                "Install with: pip install sounddevice numpy"
            )
            return {
                "error": "Recording dependencies missing. "
                         "Run: pip install sounddevice numpy",
                "content_type": "error",
            }
        except Exception as e:
            logger.error(f"Error recording audio: {e}")
            return {
                "error": f"Error recording audio: {e}",
                "content_type": "error",
            }

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _transcribe(
        self, audio_path: str
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Run Whisper transcription on an audio file.

        Args:
            audio_path: Path to the audio file

        Returns:
            Tuple of (full_text, segments_list, info_dict)
        """
        segments_iter, info = self.model.transcribe(
            audio_path,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,  # Skip silence for speed
        )

        segments_list = []
        full_text_parts = []

        for seg in segments_iter:
            seg_dict = {
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
                "avg_logprob": round(seg.avg_logprob, 4),
                "no_speech_prob": round(seg.no_speech_prob, 4),
            }

            # Include word-level timestamps if available
            if seg.words:
                seg_dict["words"] = [
                    {
                        "word": w.word.strip(),
                        "start": round(w.start, 2),
                        "end": round(w.end, 2),
                        "probability": round(w.probability, 4),
                    }
                    for w in seg.words
                ]

            segments_list.append(seg_dict)
            full_text_parts.append(seg.text.strip())

        full_text = " ".join(full_text_parts)

        info_dict = {
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
        }

        logger.info(
            f"Transcription complete — language={info.language} "
            f"({info.language_probability:.1%}), "
            f"segments={len(segments_list)}, "
            f"duration={info.duration:.1f}s"
        )

        return full_text, segments_list, info_dict

    def _extract_metadata(self, audio_path: str) -> Dict[str, Any]:
        """
        Extract basic file metadata from an audio file.

        Args:
            audio_path: Path to the audio file

        Returns:
            Dict with file metadata
        """
        metadata = {
            "file_size_bytes": os.path.getsize(audio_path),
            "file_name": os.path.basename(audio_path),
            "file_extension": os.path.splitext(audio_path)[1].lower(),
        }

        # Try to read WAV header for extra info
        if metadata["file_extension"] == ".wav":
            try:
                import wave

                with wave.open(audio_path, "rb") as wf:
                    metadata["channels"] = wf.getnchannels()
                    metadata["sample_width"] = wf.getsampwidth()
                    metadata["frame_rate"] = wf.getframerate()
                    metadata["num_frames"] = wf.getnframes()
                    metadata["duration_seconds"] = round(
                        wf.getnframes() / wf.getframerate(), 2
                    )
            except Exception as e:
                logger.debug(f"Could not read WAV header: {e}")

        return metadata

    def _analyze_transcription(self, transcription: str) -> Dict[str, Any]:
        """
        Use the LLM to perform enhanced analysis on the transcription.

        Args:
            transcription: The transcribed text

        Returns:
            Dict with analysis results
        """
        analysis: Dict[str, Any] = {
            "word_count": len(transcription.split()),
            "contains_question": "?" in transcription,
        }

        try:
            from langchain.schema import HumanMessage, SystemMessage

            messages = [
                SystemMessage(
                    content=(
                        "You are an AI assistant that analyzes transcribed "
                        "speech. Extract key information, intent, and "
                        "sentiment. Respond concisely."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Analyze this transcribed speech:\n\n{transcription}"
                    )
                ),
            ]

            response = self.llm.generate([messages])
            analysis["enhanced_analysis"] = (
                response.generations[0][0].text.strip()
            )

        except Exception as e:
            logger.warning(f"Enhanced analysis failed: {e}")

        return analysis