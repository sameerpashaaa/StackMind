#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Voice Processor Module

This module implements voice processing capabilities for the AI Problem Solver,
allowing it to handle and analyze voice-based inputs.
"""

import logging
import os
import tempfile
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)

class VoiceProcessor:
    """
    Voice processor for handling voice-based inputs.
    
    This processor transcribes and analyzes voice inputs,
    converting them to text for further processing.
    
    Attributes:
        llm: Optional language model for enhanced voice understanding
        settings: Application settings
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
        
        # Check for required dependencies
        self._check_dependencies()
        
        logger.info("Voice processor initialized")
    
    def _check_dependencies(self):
        """
        Check if required dependencies are installed.
        """
        try:
            import speech_recognition
            import pydub
        except ImportError:
            logger.warning("Voice processing dependencies are not installed. "
                          "Install them with: pip install SpeechRecognition pydub")
    
    def process(self, audio_path: str) -> Dict[str, Any]:
        """
        Process an audio file and extract relevant information.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dict[str, Any]: Processed audio with transcription and metadata
        """
        # Check if file exists
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return {
                "error": f"Audio file not found: {audio_path}",
                "content_type": "error"
            }
        
        try:
            # Initialize result
            result = {
                "content_type": "voice",
                "file_path": audio_path,
                "metadata": {},
                "transcription": ""
            }
            
            # Extract basic metadata
            result["metadata"] = self._extract_metadata(audio_path)
            
            # Transcribe audio
            result["transcription"] = self._transcribe_audio(audio_path)
            
            # Analyze audio if transcription is successful
            if result["transcription"]:
                result["analysis"] = self._analyze_audio(result["transcription"])
            
            logger.debug(f"Processed audio: {audio_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            return {
                "error": f"Error processing audio: {str(e)}",
                "content_type": "error"
            }
    
    def _extract_metadata(self, audio_path: str) -> Dict[str, Any]:
        """
        Extract basic metadata from an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dict[str, Any]: Audio metadata
        """
        try:
            from pydub import AudioSegment
            
            # Load audio file
            audio = AudioSegment.from_file(audio_path)
            
            metadata = {
                "duration_seconds": len(audio) / 1000,  # Convert milliseconds to seconds
                "channels": audio.channels,
                "sample_width": audio.sample_width,
                "frame_rate": audio.frame_rate,
                "file_size_bytes": os.path.getsize(audio_path)
            }
            
            return metadata
            
        except ImportError:
            logger.warning("pydub is not installed. Install it with: pip install pydub")
            return {
                "file_size_bytes": os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
            }
        except Exception as e:
            logger.warning(f"Error extracting audio metadata: {str(e)}")
            return {
                "file_size_bytes": os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
            }
    
    def _transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe audio to text.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            str: Transcribed text
        """
        try:
            import speech_recognition as sr
            from pydub import AudioSegment
            
            # Initialize recognizer
            recognizer = sr.Recognizer()
            
            # Convert audio to WAV format if needed
            audio_format = audio_path.split('.')[-1].lower()
            if audio_format != 'wav':
                logger.debug(f"Converting {audio_format} to WAV format for transcription")
                audio = AudioSegment.from_file(audio_path)
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                    temp_wav_path = temp_wav.name
                audio.export(temp_wav_path, format='wav')
                audio_path_for_transcription = temp_wav_path
            else:
                audio_path_for_transcription = audio_path
            
            # Load audio file
            with sr.AudioFile(audio_path_for_transcription) as source:
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source)
                # Record audio
                audio_data = recognizer.record(source)
            
            # Clean up temporary file if created
            if audio_format != 'wav':
                os.unlink(temp_wav_path)
            
            # Use appropriate recognition service based on settings
            if self.settings and hasattr(self.settings, 'speech_recognition') and \
               self.settings.speech_recognition.get('service') == 'google_cloud':
                # Use Google Cloud Speech API if configured
                if not self.settings.speech_recognition.get('google_cloud_credentials'):
                    logger.warning("Google Cloud Speech API credentials not configured")
                    return self._fallback_transcription(audio_path)
                
                return recognizer.recognize_google_cloud(
                    audio_data,
                    credentials_json=self.settings.speech_recognition.get('google_cloud_credentials')
                )
            elif self.settings and hasattr(self.settings, 'speech_recognition') and \
                 self.settings.speech_recognition.get('service') == 'whisper':
                # Use OpenAI Whisper API if configured
                if not self.settings.speech_recognition.get('whisper_api_key'):
                    logger.warning("Whisper API key not configured")
                    return self._fallback_transcription(audio_path)
                
                # This would require a custom implementation or a library that supports Whisper API
                return self._transcribe_with_whisper(audio_path)
            else:
                # Default to Google Speech Recognition (free, but limited)
                return recognizer.recognize_google(audio_data)
            
        except ImportError:
            logger.warning("speech_recognition or pydub is not installed. "
                          "Install them with: pip install SpeechRecognition pydub")
            return ""
        except sr.UnknownValueError:
            logger.warning("Speech recognition could not understand audio")
            return ""
        except sr.RequestError as e:
            logger.warning(f"Could not request results from speech recognition service: {str(e)}")
            return ""
        except Exception as e:
            logger.warning(f"Error transcribing audio: {str(e)}")
            return ""
    
    def _transcribe_with_whisper(self, audio_path: str) -> str:
        """
        Transcribe audio using OpenAI's Whisper API.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            str: Transcribed text
        """
        try:
            import openai
            
            # Set API key from settings
            if self.settings and hasattr(self.settings, 'speech_recognition'):
                openai.api_key = self.settings.speech_recognition.get('whisper_api_key')
            
            # Open audio file
            with open(audio_path, "rb") as audio_file:
                # Call Whisper API
                response = openai.Audio.transcribe("whisper-1", audio_file)
            
            # Extract transcription
            return response.get("text", "")
            
        except ImportError:
            logger.warning("openai package is not installed. Install it with: pip install openai")
            return ""
        except Exception as e:
            logger.warning(f"Error transcribing with Whisper: {str(e)}")
            return ""
    
    def _fallback_transcription(self, audio_path: str) -> str:
        """
        Fallback transcription method when primary methods fail.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            str: Transcribed text or error message
        """
        try:
            import speech_recognition as sr
            
            # Initialize recognizer
            recognizer = sr.Recognizer()
            
            # Load audio file
            with sr.AudioFile(audio_path) as source:
                # Record audio
                audio_data = recognizer.record(source)
            
            # Try multiple services
            try:
                return recognizer.recognize_google(audio_data)
            except:
                try:
                    return recognizer.recognize_sphinx(audio_data)
                except:
                    return "[Transcription failed with all available methods]"
                
        except ImportError:
            return "[Speech recognition libraries not available]"
        except Exception as e:
            return f"[Transcription error: {str(e)}]"
    
    def _analyze_audio(self, transcription: str) -> Dict[str, Any]:
        """
        Analyze transcribed audio to extract additional information.
        
        Args:
            transcription: Transcribed text from audio
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        # Initialize analysis
        analysis = {
            "detected_language": self._detect_language(transcription),
            "contains_question": '?' in transcription,
            "word_count": len(transcription.split())
        }
        
        # Use LLM for enhanced analysis if available
        if self.llm and transcription:
            try:
                from langchain.schema import HumanMessage, SystemMessage
                
                # Create prompt for analysis
                messages = [
                    SystemMessage(content="You are an AI assistant that analyzes transcribed speech. "
                                        "Extract key information, intent, and sentiment."),
                    HumanMessage(content=f"Analyze this transcribed speech: {transcription}")
                ]
                
                # Generate analysis
                response = self.llm.generate([messages])
                llm_analysis = response.generations[0][0].text.strip()
                
                # Add LLM analysis
                analysis["enhanced_analysis"] = llm_analysis
                
            except Exception as e:
                logger.warning(f"Error generating enhanced audio analysis: {str(e)}")
        
        return analysis
    
    def _detect_language(self, text: str) -> str:
        """
        Detect the language of the transcribed text.
        
        This is a simple implementation that could be enhanced with NLP libraries.
        
        Args:
            text: Transcribed text
            
        Returns:
            str: Detected language code
        """
        try:
            from langdetect import detect
            
            if not text or len(text.strip()) < 5:
                return "unknown"
                
            return detect(text)
            
        except ImportError:
            logger.warning("langdetect is not installed. Install it with: pip install langdetect")
            return "unknown"
        except Exception as e:
            logger.warning(f"Error detecting language: {str(e)}")
            return "unknown"
    
    def record_audio(self, duration: int = 5, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Record audio from the microphone.
        
        Args:
            duration: Duration in seconds to record
            output_path: Optional path to save the recorded audio
            
        Returns:
            Dict[str, Any]: Result with file path and transcription
        """
        try:
            import speech_recognition as sr
            import wave
            import numpy as np
            
            # Initialize recognizer
            recognizer = sr.Recognizer()
            
            # Create output path if not provided
            if not output_path:
                output_dir = tempfile.gettempdir()
                output_path = os.path.join(output_dir, f"recorded_audio_{int(time.time())}.wav")
            
            logger.info(f"Recording audio for {duration} seconds...")
            
            # Record audio from microphone
            with sr.Microphone() as source:
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source)
                
                # Record audio
                audio_data = recognizer.listen(source, timeout=duration)
                
                # Save audio to file
                with open(output_path, "wb") as f:
                    f.write(audio_data.get_wav_data())
            
            logger.info(f"Audio recorded and saved to {output_path}")
            
            # Process the recorded audio
            result = self.process(output_path)
            
            return result
            
        except ImportError:
            logger.error("Required libraries for audio recording are not installed")
            return {
                "error": "Required libraries for audio recording are not installed",
                "content_type": "error"
            }
        except Exception as e:
            logger.error(f"Error recording audio: {str(e)}")
            return {
                "error": f"Error recording audio: {str(e)}",
                "content_type": "error"
            }