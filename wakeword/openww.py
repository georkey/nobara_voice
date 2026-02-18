from pathlib import Path
import logging

import openwakeword as oww
import numpy as np
import pyaudio as pa

logger = logging.getLogger()
# Some settings
# Choose and download model if it is not downloaded
MODEL_NAME = "hey_mycroft"
MODEL_PATH = Path(oww.MODELS[MODEL_NAME]["model_path"])
# ONNX works both in Windows and Linux
INFERENCE = "onnx"
# Mic settings
FORMAT = pa.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280

if not MODEL_PATH.exists():
    oww.utils.download_models([MODEL_NAME])
if INFERENCE == "onnx":
    MODEL_PATH = Path(str(MODEL_PATH).replace(".tflite", ".onnx"))

model = oww.Model([str(MODEL_PATH)], inference_framework=INFERENCE)
md_key = list(model.models.keys())[0]

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.DEBUG)
    audio = pa.PyAudio()
    mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    logger.info("Start listening...")
    while True:
        sample = np.frombuffer(mic_stream.read(CHUNK), dtype=np.int16)
        prediction = model.predict(sample)
        score = model.prediction_buffer[md_key][-1]
        if score > 0.5:
            logger.debug("Wakeword detected!")
