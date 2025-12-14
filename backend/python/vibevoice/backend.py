#!/usr/bin/env python3
"""
This is an extra gRPC server of LocalAI for VibeVoice 0.5B streaming model.
Supports real-time TTS with preconfigured voice presets (.pt files).

NOTE: For the 1.5B long-form multi-speaker model, use the vibevoice-1.5b backend instead.
"""
from concurrent import futures
import time
import argparse
import signal
import sys
import os
import copy
import traceback
from pathlib import Path
import backend_pb2
import backend_pb2_grpc
import torch
import numpy as np
from threading import Thread

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
    BackendServicer for VibeVoice 0.5B streaming model.
    """
    def Health(self, request, context):
        return backend_pb2.Reply(message=bytes("OK", 'utf-8'))

    def LoadModel(self, request, context):
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

        if device == "mpx":
            print("Note: device 'mpx' detected, treating it as 'mps'.", file=sys.stderr)
            device = "mps"

        if device == "mps" and not torch.backends.mps.is_available():
            print("Warning: MPS not available. Falling back to CPU.", file=sys.stderr)
            device = "cpu"

        self.device = device
        self._torch_device = torch.device(device)

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

        # Get model path
        model_path = request.Model
        if not model_path:
            model_path = "microsoft/VibeVoice-Realtime-0.5B"

        # Parameters
        self.inference_steps = self.options.get("inference_steps", 5)
        if not isinstance(self.inference_steps, int) or self.inference_steps <= 0:
            self.inference_steps = 5

        self.cfg_scale = self.options.get("cfg_scale", 1.5)
        if not isinstance(self.cfg_scale, (int, float)) or self.cfg_scale <= 0:
            self.cfg_scale = 1.5

        try:
            return self._load_model(request, model_path)
        except Exception as err:
            print(f"Error loading model: {err}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            return backend_pb2.Result(success=False, message=f"Unexpected {err=}, {type(err)=}")

    def _load_model(self, request, model_path):
        """Load the 0.5B streaming model with preconfigured voice presets."""
        from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
        from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor

        # Setup voices directory for .pt preset files
        voices_dir = self._find_voices_dir(request)
        self.voices_dir = voices_dir
        self.voice_presets = {}
        self._voice_cache = {}
        self.default_voice_key = None

        if self.voices_dir and os.path.exists(self.voices_dir):
            self._load_voice_presets()
        else:
            print(f"Warning: Voices directory not found. Voice presets will not be available.", file=sys.stderr)

        print(f"Loading VibeVoice 0.5B from {model_path}", file=sys.stderr)
        self.processor = VibeVoiceStreamingProcessor.from_pretrained(model_path)

        # Determine dtype and attention
        if self.device == "mps":
            load_dtype = torch.float32
            device_map = None
            attn_impl = "sdpa"
        elif self.device == "cuda":
            load_dtype = torch.bfloat16
            device_map = "cuda"
            attn_impl = "flash_attention_2"
        else:
            load_dtype = torch.float32
            device_map = "cpu"
            attn_impl = "sdpa"

        print(f"Using device: {self.device}, dtype: {load_dtype}, attn: {attn_impl}", file=sys.stderr)

        try:
            self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                model_path,
                torch_dtype=load_dtype,
                device_map=device_map if self.device != "mps" else None,
                attn_implementation=attn_impl,
            )
            if self.device == "mps":
                self.model.to("mps")
        except Exception as e:
            if attn_impl == 'flash_attention_2':
                print(f"Flash attention failed: {e}, falling back to SDPA", file=sys.stderr)
                self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    model_path,
                    torch_dtype=load_dtype,
                    device_map=device_map if self.device != "mps" else None,
                    attn_implementation='sdpa'
                )
                if self.device == "mps":
                    self.model.to("mps")
            else:
                raise e

        self.model.eval()
        self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)

        if self.voice_presets:
            preset_name = os.environ.get("VOICE_PRESET")
            self.default_voice_key = self._determine_voice_key(preset_name)
            print(f"Default voice preset: {self.default_voice_key}", file=sys.stderr)
        else:
            print("Warning: No voice presets available.", file=sys.stderr)

        return backend_pb2.Result(message="VibeVoice 0.5B loaded successfully", success=True)

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

        if request.ModelFile:
            base = os.path.dirname(request.ModelFile)
            search_paths.append(os.path.join(base, "voices"))

        # Backend directory
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        search_paths.append(os.path.join(backend_dir, "voices"))

        for path in search_paths:
            if path and os.path.exists(path):
                print(f"Found voices directory: {path}", file=sys.stderr)
                return path

        return None

    def _load_voice_presets(self):
        """Load .pt voice preset files."""
        self.voice_presets = {}
        pt_files = [f for f in os.listdir(self.voices_dir)
                    if f.lower().endswith('.pt') and os.path.isfile(os.path.join(self.voices_dir, f))]

        for pt_file in pt_files:
            name = os.path.splitext(pt_file)[0]
            full_path = os.path.join(self.voices_dir, pt_file)
            self.voice_presets[name] = full_path

        self.voice_presets = dict(sorted(self.voice_presets.items()))
        print(f"Found {len(self.voice_presets)} voice presets: {', '.join(self.voice_presets.keys())}", file=sys.stderr)

    def _determine_voice_key(self, name):
        """Determine voice key from name or use default."""
        if name and name in self.voice_presets:
            return name

        for default_key in ["en-WHTest_man", "en-Frank_man", "Frank"]:
            if default_key in self.voice_presets:
                return default_key

        if self.voice_presets:
            return next(iter(self.voice_presets))
        return None

    def _get_voice_path(self, speaker_name):
        """Get voice file path."""
        if not self.voice_presets:
            return None

        if speaker_name and speaker_name in self.voice_presets:
            return self.voice_presets[speaker_name]

        if speaker_name:
            speaker_lower = speaker_name.lower()
            for preset_name, path in self.voice_presets.items():
                if preset_name.lower() in speaker_lower or speaker_lower in preset_name.lower():
                    return path

        if self.default_voice_key and self.default_voice_key in self.voice_presets:
            return self.voice_presets[self.default_voice_key]
        elif self.voice_presets:
            return list(self.voice_presets.values())[0]

        return None

    def _ensure_voice_cached(self, voice_path):
        """Load and cache voice preset."""
        if not voice_path or not os.path.exists(voice_path):
            return None

        if voice_path not in self._voice_cache:
            print(f"Loading prefilled prompt from {voice_path}", file=sys.stderr)
            prefilled_outputs = torch.load(
                voice_path,
                map_location=self._torch_device,
                weights_only=False,
            )
            self._voice_cache[voice_path] = prefilled_outputs

        return self._voice_cache[voice_path]

    def TTS(self, request, context):
        """Handle TTS request."""
        try:
            voice_path = None
            if request.voice:
                voice_path = self._get_voice_path(request.voice)
            elif request.AudioPath:
                voice_path = request.AudioPath if os.path.isabs(request.AudioPath) else None
            elif self.default_voice_key:
                voice_path = self._get_voice_path(self.default_voice_key)

            if not voice_path or not os.path.exists(voice_path):
                return backend_pb2.Result(
                    success=False,
                    message=f"Voice file not found: {voice_path}"
                )

            prefilled_outputs = self._ensure_voice_cached(voice_path)
            if prefilled_outputs is None:
                return backend_pb2.Result(success=False, message=f"Failed to load voice preset")

            cfg_scale = self.options.get("cfg_scale", self.cfg_scale)
            do_sample = self.options.get("do_sample", False)
            temperature = self.options.get("temperature", 0.9)
            top_p = self.options.get("top_p", 0.9)

            text = request.text.strip().replace("'", "'").replace('"', '"').replace('"', '"')

            inputs = self.processor.process_input_with_cached_prompt(
                text=text,
                cached_prompt=prefilled_outputs,
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )

            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(self._torch_device)

            print(f"Generating with cfg_scale={cfg_scale}", file=sys.stderr)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg_scale,
                tokenizer=self.processor.tokenizer,
                generation_config={
                    'do_sample': do_sample,
                    'temperature': temperature if do_sample else 1.0,
                    'top_p': top_p if do_sample else 1.0,
                },
                verbose=False,
                all_prefilled_outputs=copy.deepcopy(prefilled_outputs),
            )

            if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
                self.processor.save_audio(outputs.speech_outputs[0], output_path=request.dst)
                print(f"Saved to {request.dst}", file=sys.stderr)
            else:
                return backend_pb2.Result(success=False, message="No audio generated")

        except Exception as err:
            print(f"TTS error: {err}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            return backend_pb2.Result(success=False, message=f"Error: {err}")

        return backend_pb2.Result(success=True)

    def TTSStream(self, request, context):
        """Stream TTS audio chunks."""
        try:
            from vibevoice.modular.streamer import AudioStreamer

            voice_path = None
            if request.voice:
                voice_path = self._get_voice_path(request.voice)
            elif self.default_voice_key:
                voice_path = self._get_voice_path(self.default_voice_key)

            if not voice_path or not os.path.exists(voice_path):
                yield backend_pb2.TTSStreamChunk(
                    audio=b'', sample_rate=24000, chunk_index=0, is_final=True,
                    error=f"Voice file not found: {voice_path}"
                )
                return

            prefilled_outputs = self._ensure_voice_cached(voice_path)
            if prefilled_outputs is None:
                yield backend_pb2.TTSStreamChunk(
                    audio=b'', sample_rate=24000, chunk_index=0, is_final=True,
                    error="Failed to load voice preset"
                )
                return

            cfg_scale = self.options.get("cfg_scale", self.cfg_scale)
            text = request.text.strip().replace("'", "'").replace('"', '"').replace('"', '"')

            inputs = self.processor.process_input_with_cached_prompt(
                text=text,
                cached_prompt=prefilled_outputs,
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )

            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(self._torch_device)

            print(f"TTSStream: cfg_scale={cfg_scale}", file=sys.stderr)

            audio_streamer = AudioStreamer(batch_size=1)
            generation_error = [None]

            def run_generation():
                try:
                    self.model.generate(
                        **inputs,
                        max_new_tokens=None,
                        cfg_scale=cfg_scale,
                        tokenizer=self.processor.tokenizer,
                        generation_config={'do_sample': False, 'temperature': 1.0, 'top_p': 1.0},
                        verbose=False,
                        all_prefilled_outputs=copy.deepcopy(prefilled_outputs),
                        audio_streamer=audio_streamer,
                    )
                except Exception as e:
                    generation_error[0] = e
                    print(f"Generation error: {e}", file=sys.stderr)

            thread = Thread(target=run_generation, daemon=True)
            thread.start()

            chunk_index = 0
            try:
                for audio_chunk in audio_streamer.get_stream(0):
                    audio_np = audio_chunk.detach().cpu().to(torch.float32).numpy()
                    if audio_np.ndim > 1:
                        audio_np = audio_np.reshape(-1)
                    peak = np.max(np.abs(audio_np)) if audio_np.size else 0.0
                    if peak > 1.0:
                        audio_np = audio_np / peak
                    audio_int16 = (audio_np * 32767).astype(np.int16)

                    yield backend_pb2.TTSStreamChunk(
                        audio=audio_int16.tobytes(),
                        sample_rate=24000,
                        chunk_index=chunk_index,
                        is_final=False
                    )
                    chunk_index += 1
            except Exception as stream_err:
                print(f"Stream error: {stream_err}", file=sys.stderr)

            thread.join(timeout=60)

            if generation_error[0] is not None:
                yield backend_pb2.TTSStreamChunk(
                    audio=b'', sample_rate=24000, chunk_index=chunk_index, is_final=True,
                    error=f"Generation error: {generation_error[0]}"
                )
                return

            yield backend_pb2.TTSStreamChunk(
                audio=b'', sample_rate=24000, chunk_index=chunk_index, is_final=True
            )

        except Exception as err:
            print(f"TTSStream error: {err}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            yield backend_pb2.TTSStreamChunk(
                audio=b'', sample_rate=24000, chunk_index=0, is_final=True,
                error=f"Error: {err}"
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
