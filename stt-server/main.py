"""
Device ASR with Emformer RNN-T
==============================

**Author**: `Moto Hira <moto@meta.com>`__, `Jeff Hwang <jeffhwang@meta.com>`__.

This tutorial shows how to use Emformer RNN-T and streaming API
to perform speech recognition on a streaming device input, i.e. microphone
on laptop.
"""

######################################################################
# 1. Overview
# -----------
#
# We use streaming API to fetch audio from audio device (microphone)
# chunk by chunk, then run inference using Emformer RNN-T.
#
# For the basic usage of the streaming API and Emformer RNN-T
# please refer to
# `StreamReader Basic Usage <./streamreader_basic_tutorial.html>`__ and
# `Online ASR with Emformer RNN-T <./online_asr_tutorial.html>`__.

from fastapi import FastAPI, WebSocket
import torch
import torchaudio
import io
import json
import numpy as np

app = FastAPI(
    title="STT Streaming server",
    description="""Use Emformer RNN-T to transcribe audio into text in realtime""",
    version="0.0.1",
    docs_url="/",
)

class Pipeline:
    """Build inference pipeline from RNNTBundle.

    Args:
        bundle (torchaudio.pipelines.RNNTBundle): Bundle object
        beam_width (int): Beam size of beam search decoder.
    """

    def __init__(self, bundle: torchaudio.pipelines.RNNTBundle, beam_width: int = 10):
        self.bundle = bundle
        self.feature_extractor = bundle.get_streaming_feature_extractor()
        self.decoder = bundle.get_decoder()
        self.token_processor = bundle.get_token_processor()

        self.beam_width = beam_width

        self.state = None
        self.hypotheses = None

    def infer(self, segment: torch.Tensor) -> str:
        """Perform streaming inference"""
        features, length = self.feature_extractor(segment)
        self.hypotheses, self.state = self.decoder.infer(
            features, length, self.beam_width, state=self.state, hypothesis=self.hypotheses
        )
        transcript = self.token_processor(self.hypotheses[0][0], lstrip=False)
        return transcript

class ContextCacher:
    """Cache the end of input data and prepend the next input data with it.

    Args:
        segment_length (int): The size of main segment.
            If the incoming segment is shorter, then the segment is padded.
        context_length (int): The size of the context, cached and appended.
    """

    def __init__(self, segment_length: int, context_length: int):
        self.segment_length = segment_length
        self.context_length = context_length
        self.context = torch.zeros([context_length])

    def __call__(self, chunk: torch.Tensor):
        if chunk.size(0) < self.segment_length:
            chunk = torch.nn.functional.pad(chunk, (0, self.segment_length - chunk.size(0)))
        chunk_with_context = torch.cat((self.context, chunk))
        self.context = chunk[-self.context_length :]
        return chunk_with_context

@app.get('/status')
async def status():
    return "good"

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # initialize the pipeline
    bundle = torchaudio.pipelines.EMFORMER_RNNT_BASE_LIBRISPEECH
    pipeline = Pipeline(bundle)

    # initialize the ContextCacher
    sample_rate = bundle.sample_rate
    segment_length = bundle.segment_length * bundle.hop_length
    context_length = bundle.right_context_length * bundle.hop_length

    print(f"Sample rate: {sample_rate}")
    print(f"Main segment: {segment_length} frames ({segment_length / sample_rate} seconds)")
    print(f"Right context: {context_length} frames ({context_length / sample_rate} seconds)")

    cacher = ContextCacher(segment_length, context_length)

    while True:
        # Receive audio data from the client
        audio_data = await websocket.receive_bytes()
        try:
            json.loads(audio_data)
            continue
        except json.JSONDecodeError:
            pass
        except UnicodeDecodeError:
            pass

        # Convert the bytes object to a file-like object
        audio_array = np.frombuffer(audio_data, dtype=np.int16)

        # Normalize the audio data to [-1, 1]
        audio_array = audio_array / 32768.0

        # Load the audio data as a tensor
        audio_tensor = torch.from_numpy(audio_array).float()

        # Pass the audio data through the ContextCacher
        cached_audio_tensor = cacher(audio_tensor)

        # Process the audio data using the pipeline
        transcript = pipeline.infer(cached_audio_tensor)

        # Send the results back to the client
        await websocket.send_text(transcript)