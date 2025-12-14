#!/usr/bin/env python3
"""
This is an extra gRPC server of LocalAI for Kokoro TTS
"""
from concurrent import futures
import time
import argparse
import signal
import sys
import os
import backend_pb2
import backend_pb2_grpc

import numpy as np
import torch
from kokoro import KPipeline
import soundfile as sf

import grpc


_ONE_DAY_IN_SECONDS = 60 * 60 * 24

# If MAX_WORKERS are specified in the environment use it, otherwise default to 1
MAX_WORKERS = int(os.environ.get('PYTHON_GRPC_MAX_WORKERS', '1'))
KOKORO_LANG_CODE = os.environ.get('KOKORO_LANG_CODE', 'a')

# Implement the BackendServicer class with the service methods
class BackendServicer(backend_pb2_grpc.BackendServicer):
    """
    BackendServicer is the class that implements the gRPC service
    """
    def Health(self, request, context):
        return backend_pb2.Reply(message=bytes("OK", 'utf-8'))
    
    def LoadModel(self, request, context):
        try:
            print("Preparing Kokoro TTS pipeline, please wait", file=sys.stderr)
            # empty dict
            self.options = {}
            options = request.Options
            # The options are a list of strings in this form optname:optvalue
            # We are storing all the options in a dict so we can use it later when
            # generating the images
            for opt in options:
                if ":" not in opt:
                    continue
                key, value = opt.split(":")
                self.options[key] = value

            # Initialize Kokoro pipeline with language code
            lang_code = self.options.get("lang_code", KOKORO_LANG_CODE)
            self.pipeline = KPipeline(lang_code=lang_code)
            print(f"Kokoro TTS pipeline loaded with language code: {lang_code}", file=sys.stderr)
        except Exception as err:
            return backend_pb2.Result(success=False, message=f"Unexpected {err=}, {type(err)=}")
        
        return backend_pb2.Result(message="Kokoro TTS pipeline loaded successfully", success=True)

    def TTS(self, request, context):
        try:
            # Get voice from request, default to 'af_heart' if not specified
            voice = request.voice if request.voice else 'af_heart'
            
            # Generate audio using Kokoro pipeline
            generator = self.pipeline(request.text, voice=voice)
            
            speechs = []
            # Get all the audio segment
            for i, (gs, ps, audio) in enumerate(generator):
                speechs.append(audio)
                print(f"Generated audio segment {i}: gs={gs}, ps={ps}", file=sys.stderr)
            # Merges the audio segments and writes them to the destination
            speech = torch.cat(speechs, dim=0)
            sf.write(request.dst, speech, 24000)

        except Exception as err:
            return backend_pb2.Result(success=False, message=f"Unexpected {err=}, {type(err)=}")

        return backend_pb2.Result(success=True)

    def TTSStream(self, request, context):
        """Stream TTS audio chunks as they are generated."""
        try:
            # Get voice from request, default to 'af_heart' if not specified
            voice = request.voice if request.voice else 'af_heart'

            # Generate audio using Kokoro pipeline - it's already a generator!
            generator = self.pipeline(request.text, voice=voice)

            chunk_index = 0
            for i, (gs, ps, audio) in enumerate(generator):
                # audio is a torch tensor - convert to 16-bit PCM bytes
                audio_np = audio.cpu().numpy()
                # Normalize to int16 range
                audio_int16 = (audio_np * 32767).astype(np.int16)
                audio_bytes = audio_int16.tobytes()

                yield backend_pb2.TTSStreamChunk(
                    audio=audio_bytes,
                    sample_rate=24000,
                    chunk_index=chunk_index,
                    is_final=False
                )
                chunk_index += 1
                print(f"Streamed audio segment {i}: gs={gs}, ps={ps}", file=sys.stderr)

            # Send final empty chunk to signal completion
            yield backend_pb2.TTSStreamChunk(
                audio=b'',
                sample_rate=24000,
                chunk_index=chunk_index,
                is_final=True
            )

        except Exception as err:
            print(f"TTSStream error: {err}", file=sys.stderr)
            yield backend_pb2.TTSStreamChunk(
                audio=b'',
                sample_rate=24000,
                chunk_index=0,
                is_final=True,
                error=f"Unexpected {err=}, {type(err)=}"
            )

def serve(address):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_WORKERS),
        options=[
            ('grpc.max_message_length', 50 * 1024 * 1024),  # 50MB
            ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50MB
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),  # 50MB
        ])
    backend_pb2_grpc.add_BackendServicer_to_server(BackendServicer(), server)
    server.add_insecure_port(address)
    server.start()
    print("Server started. Listening on: " + address, file=sys.stderr)

    # Define the signal handler function
    def signal_handler(sig, frame):
        print("Received termination signal. Shutting down...")
        server.stop(0)
        sys.exit(0)

    # Set the signal handlers for SIGINT and SIGTERM
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the gRPC server.")
    parser.add_argument(
        "--addr", default="localhost:50051", help="The address to bind the server to."
    )
    args = parser.parse_args()

    serve(args.addr)
