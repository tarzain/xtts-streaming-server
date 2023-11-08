import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Iterator

import requests
import asyncio
import websockets
import pyaudio

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 2560

# Used for microphone streaming only.
async def run_loop():
    audio_queue = asyncio.Queue()
    start_event = asyncio.Event()  # Create an Event
    semaphore = asyncio.Semaphore(1)  # Create a Semaphore

    def mic_callback(input_data, frame_count, time_info, status_flag):
        if not semaphore.locked():
            audio_queue.put_nowait(input_data)
        return (input_data, pyaudio.paContinue)
    
    # Set up microphone if streaming from mic
    async def microphone():
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            stream_callback=mic_callback,
        )

        stream.start_stream()
        start_event.set()  # Set the event to trigger the other coroutines to start

        global SAMPLE_SIZE
        SAMPLE_SIZE = audio.get_sample_size(FORMAT)

        while stream.is_active():
            await asyncio.sleep(0.1)

        stream.stop_stream()
        stream.close()

    # Open a websocket connection
    async with websockets.connect(f'ws://localhost:8080/ws') as ws:
        # If the request is successful, print the request ID from the HTTP header
        print('🟢 Successfully opened connection')

        async def send_speech_to_text_stream():
            nonlocal ws
            """Stream audio from the microphone to Deepgram using websockets."""
            await start_event.wait()
            print("microphone started")
            try:    
                while True:
                    print("Connecting to send audio to deepgram for transcriptions.")
                    try:
                        while True:
                            if semaphore.locked():
                                print("AI is speaking, skipping mic data", end='\r', flush=True)
                                await ws.send(json.dumps({ "type": "KeepAlive" }))
                            else:
                                mic_data = await audio_queue.get()
                                await ws.send(mic_data)
                    except websockets.exceptions.ConnectionClosedError:
                        print("Connection closed unexpectedly. Reconnecting...")
                        ws = await websockets.connect(f'ws://localhost:8080/ws')
                        print('🟢 Successfully reopened connection')
                    except websockets.exceptions.InvalidStatusCode as e:
                        print(f'🔴 ERROR: Could not connect to server')
            except KeyboardInterrupt as _:
                await ws.close()
                print(f'🔴 ERROR: Closed stream via keyboard interrupt')

        async def receive_speech_to_text_stream():
            nonlocal ws
            """Receive text from Deepgram and pass it to the chat completion function."""
            while True:
                print("Connecting to receive transcriptions")
                try:
                    while True:
                        response = await ws.recv()
                        print(response, sep=' ', end='\r', flush=True)
                            
                except websockets.exceptions.ConnectionClosedError:
                        print("Connection closed unexpectedly. Reconnecting...")
                        ws = await websockets.connect(f'ws://localhost:8080/ws')
                        print('🟢 Successfully reopened connection')
                except websockets.exceptions.InvalidStatusCode as e:
                    # If the request fails, print both the error message and the request ID from the HTTP headers
                    print(f'🔴 ERROR: Could not connect to server')

        return await asyncio.gather(
            asyncio.ensure_future(microphone()),
            asyncio.ensure_future(send_speech_to_text_stream()),
            asyncio.ensure_future(receive_speech_to_text_stream()),
        )

# Main execution
if __name__ == "__main__":
    user_query = "Hello, tell me a story that's only 1 sentence long."

    asyncio.run(run_loop())