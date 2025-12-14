#!/usr/bin/env python3
"""
This is an extra gRPC server of LocalAI for VibeVoice 1.5B long-form model.
Supports multi-speaker TTS with voice cloning from WAV files.

This backend uses the standalone vibevoice package with VibeVoiceProcessor
for handling multi-speaker input format.
"""
from concurrent import futures
import time
import argparse
import signal
import sys
import os
import traceback
from pathlib import Path
import backend_pb2
import backend_pb2_grpc
import torch
import numpy as np
import re

import grpc

def is_float(s):
    """Check if a string can be converted to float."""
    try:
        float(s)
        return True
    except ValueError:
        return False

def is_int(s):
    """Check if a string can be converted to int."""
    try:
        int(s)
        return True
    except ValueError:
        return False

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

# If MAX_WORKERS are specified in the environment use it, otherwise default to 1
MAX_WORKERS = int(os.environ.get('PYTHON_GRPC_MAX_WORKERS', '1'))

class BackendServicer(backend_pb2_grpc.BackendServicer):
    """
    BackendServicer for VibeVoice 1.5B long-form multi-speaker model.

    NOTE: The 1.5B model uses a different architecture than the 0.5B streaming model.
    This backend uses VibeVoiceProcessor for multi-speaker script processing and
    voice sample cloning.
    """
    def Health(self, request, context):
        return backend_pb2.Reply(message=bytes("OK", 'utf-8'))

    def LoadModel(self, request, context):
        # Parse options
        self.options = {}
        for opt in request.Options:
            if ":" not in opt:
                continue
            key, value = opt.split(":", 1)
            if is_float(value):
                value = float(value)
            elif is_int(value):
                value = int(value)
            elif value.lower() in ["true", "false"]:
                value = value.lower() == "true"
            self.options[key] = value

        # Get device
        if torch.cuda.is_available():
            print("CUDA is available", file=sys.stderr)
            device = "cuda"
        else:
            print("CUDA is not available", file=sys.stderr)
            device = "cpu"
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if mps_available:
            device = "mps"
        if not torch.cuda.is_available() and request.CUDA:
            return backend_pb2.Result(success=False, message="CUDA is not available")

        if device == "mps" and not torch.backends.mps.is_available():
            print("Warning: MPS not available. Falling back to CPU.", file=sys.stderr)
            device = "cpu"

        self.device = device
        self._torch_device = torch.device(device)

        # Get model path
        model_path = request.Model
        if not model_path:
            model_path = "microsoft/VibeVoice-1.5B"

        # Parameters
        self.inference_steps = self.options.get("inference_steps", 20)
        if not isinstance(self.inference_steps, int) or self.inference_steps <= 0:
            self.inference_steps = 20

        self.cfg_scale = self.options.get("cfg_scale", 1.3)
        if not isinstance(self.cfg_scale, (int, float)) or self.cfg_scale <= 0:
            self.cfg_scale = 1.3

        try:
            return self._load_model(request, model_path)
        except Exception as err:
            print(f"Error loading model: {err}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            return backend_pb2.Result(success=False, message=f"Unexpected {err=}, {type(err)=}")

    def _load_model(self, request, model_path):
        """Load the 1.5B long-form multi-speaker model."""
        # Import from the vibevoice package
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

        # For the model, we need to use AutoModelForCausalLM with trust_remote_code
        # since the 1.5B model has custom modeling code
        from transformers import AutoModelForCausalLM, AutoConfig

        # Setup voices directory for .wav audio files
        self.voices_dir = self._find_voices_dir(request)
        self.voice_samples = {}

        if self.voices_dir and os.path.exists(self.voices_dir):
            self._load_voice_samples()
        else:
            print(f"Warning: Voices directory not found. Voice samples will not be available.", file=sys.stderr)

        print(f"Loading VibeVoice 1.5B from {model_path}", file=sys.stderr)

        # Load processor from vibevoice package
        self.processor = VibeVoiceProcessor.from_pretrained(model_path)
        print("Loaded VibeVoiceProcessor", file=sys.stderr)

        # Determine dtype
        if self.device == "cuda":
            load_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            device_map = "cuda"
        else:
            load_dtype = torch.float32
            device_map = self.device

        print(f"Using device: {self.device}, dtype: {load_dtype}", file=sys.stderr)

        # Load model with trust_remote_code to use the model's custom code
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=load_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.eval()

        print("Model loaded successfully", file=sys.stderr)
        return backend_pb2.Result(message="VibeVoice 1.5B loaded successfully", success=True)

    def _find_voices_dir(self, request):
        """Find voices directory, checking multiple locations."""
        search_paths = []

        # Check options first
        if "voices_dir" in self.options:
            voices_dir = self.options["voices_dir"]
            if isinstance(voices_dir, str) and voices_dir.strip():
                if os.path.isabs(voices_dir):
                    search_paths.append(voices_dir)
                else:
                    if hasattr(request, 'ModelPath') and request.ModelPath:
                        search_paths.append(os.path.join(request.ModelPath, voices_dir))
                    if request.ModelFile:
                        search_paths.append(os.path.join(os.path.dirname(request.ModelFile), voices_dir))

        # Standard locations
        if hasattr(request, 'ModelPath') and request.ModelPath:
            search_paths.append(os.path.join(request.ModelPath, "voices"))
            search_paths.append(os.path.join(request.ModelPath, "voices", "longform"))

        if request.ModelFile:
            base = os.path.dirname(request.ModelFile)
            search_paths.append(os.path.join(base, "voices"))
            search_paths.append(os.path.join(base, "voices", "longform"))

        # Backend directory
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        search_paths.append(os.path.join(backend_dir, "voices"))

        for path in search_paths:
            if path and os.path.exists(path):
                print(f"Found voices directory: {path}", file=sys.stderr)
                return path

        return None

    def _load_voice_samples(self):
        """Load .wav voice sample files."""
        self.voice_samples = {}
        audio_extensions = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')
        audio_files = [f for f in os.listdir(self.voices_dir)
                       if f.lower().endswith(audio_extensions) and os.path.isfile(os.path.join(self.voices_dir, f))]

        for audio_file in audio_files:
            name = os.path.splitext(audio_file)[0]
            full_path = os.path.join(self.voices_dir, audio_file)
            self.voice_samples[name] = full_path

        self.voice_samples = dict(sorted(self.voice_samples.items()))
        print(f"Found {len(self.voice_samples)} voice samples: {', '.join(self.voice_samples.keys())}", file=sys.stderr)

    def _get_voice_sample_paths(self, voice_names):
        """Get voice sample paths for the given names."""
        if not self.voice_samples:
            return []

        paths = []
        for name in voice_names:
            if name in self.voice_samples:
                paths.append(self.voice_samples[name])
            else:
                # Try partial match
                for sample_name, path in self.voice_samples.items():
                    if name.lower() in sample_name.lower() or sample_name.lower() in name.lower():
                        paths.append(path)
                        break

        # If no matches, use first available samples
        if not paths and self.voice_samples:
            paths = list(self.voice_samples.values())[:len(voice_names)]

        return paths

    def TTS(self, request, context):
        """Handle TTS request for 1.5B long-form model."""
        try:
            text = request.text.strip()

            # Parse speaker names from text (format: "Speaker 1:", "Speaker 2:", etc.)
            speaker_pattern = re.compile(r'Speaker\s+(\d+):', re.IGNORECASE)
            speakers = list(set(speaker_pattern.findall(text)))
            speakers.sort(key=lambda x: int(x))
            num_speakers = len(speakers) if speakers else 1

            # Get voice samples for speakers
            voice_sample_paths = []
            if request.AudioPath:
                # AudioPath can be comma-separated list
                voice_sample_paths = [p.strip() for p in request.AudioPath.split(',') if p.strip()]
            elif request.voice:
                # Voice can be comma-separated list of voice names
                voice_names = [v.strip() for v in request.voice.split(',') if v.strip()]
                voice_sample_paths = self._get_voice_sample_paths(voice_names)

            if not voice_sample_paths and self.voice_samples:
                # Use default voices
                voice_sample_paths = list(self.voice_samples.values())[:num_speakers]

            cfg_scale = self.options.get("cfg_scale", self.cfg_scale)

            print(f"Generating with {num_speakers} speaker(s), cfg_scale={cfg_scale}", file=sys.stderr)
            print(f"Voice samples: {voice_sample_paths}", file=sys.stderr)

            # Prepare inputs using VibeVoiceProcessor
            inputs = self.processor(
                text=[text],
                voice_samples=[voice_sample_paths] if voice_sample_paths else None,
                return_tensors="pt",
                padding=True,
            )

            # Move tensors to device
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self._torch_device)

            print("Generating audio...", file=sys.stderr)

            # Generate using the model
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    tokenizer=self.processor.tokenizer,
                    cfg_scale=cfg_scale,
                    max_new_tokens=None,
                )

            # Save output
            if hasattr(outputs, 'speech_outputs') and outputs.speech_outputs and outputs.speech_outputs[0] is not None:
                audio_output = outputs.speech_outputs[0]
                self.processor.save_audio(
                    audio_output,
                    output_path=request.dst,
                )
                print(f"Saved audio to {request.dst}", file=sys.stderr)
            else:
                return backend_pb2.Result(success=False, message="No audio generated")

        except Exception as err:
            print(f"TTS error: {err}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            return backend_pb2.Result(success=False, message=f"Error: {err}")

        return backend_pb2.Result(success=True)

    def TTSStream(self, request, context):
        """Streaming TTS - NOT supported for 1.5B model."""
        yield backend_pb2.TTSStreamChunk(
            audio=b'',
            sample_rate=24000,
            chunk_index=0,
            is_final=True,
            error="Streaming TTS is not supported for VibeVoice 1.5B. "
                  "Use the vibevoice (0.5B) backend for streaming support."
        )


def serve(address):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_WORKERS),
        options=[
            ('grpc.max_message_length', 50 * 1024 * 1024),
            ('grpc.max_send_message_length', 50 * 1024 * 1024),
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
        ])
    backend_pb2_grpc.add_BackendServicer_to_server(BackendServicer(), server)
    server.add_insecure_port(address)
    server.start()
    print("Server started. Listening on: " + address, file=sys.stderr)

    def signal_handler(sig, frame):
        print("Received termination signal. Shutting down...")
        server.stop(0)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the gRPC server.")
    parser.add_argument("--addr", default="localhost:50051", help="The address to bind the server to.")
    args = parser.parse_args()
    serve(args.addr)
