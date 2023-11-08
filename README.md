# XTTS streaming server

## Running the TTS server

1. Create a conda environment and activate it `conda create -n tts python=3.11; conda activate tts`
2. Install requirements with `pip install -r requirements.txt --use-deprecated=legacy-resolver`
3. Run the server
```bash
$ cd tts-server
$ uvicorn main:app --host 0.0.0.0 --port 8000
```
4. Run the client
```bash
$ cd client
$ python tts.py --ref_file reference_audio_file_for_cloning.wav --text "text to dictate"
```