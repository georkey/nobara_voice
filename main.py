import asyncio
from pathlib import Path
import logging

import openwakeword as oww
import pyaudio

from wakeword.wakeword_service import WakeWordOWW
from speech2txt.recog_service import VOSKRecignizer

logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)

SAMPLE_RATE = 16000
CHUNK_SIZE = 1280

waiting_for_command = asyncio.Event()


async def stream_task(queue: asyncio.Queue):
    p = pyaudio.PyAudio()
    loop = asyncio.get_running_loop()

    def audio_callback(in_data, frame_count, time_info, status):
        asyncio.run_coroutine_threadsafe(queue.put(in_data), loop)
        return (None, pyaudio.paContinue)

    stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK_SIZE, stream_callback=audio_callback)
    stream.start_stream()
    while stream.is_active():
        await asyncio.sleep(0.1)
    # while True:
    #     logging.debug(f"Available data {stream.get_read_available()}")
    #     data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
    #     await queue.put(data)
    #     await asyncio.sleep(0.08)


async def recog_task(queue: asyncio.Queue, recognizer):
    logging.info("Start recognizing...")
    cmd = await recognizer.recognize(queue)
    logging.info(f"Command: {cmd}")
    waiting_for_command.clear()


async def wakeword_task(queue: asyncio.Queue, ww_detector, recognizer):
    logging.info("Wakeword starts listening...")
    while True:
        if waiting_for_command.is_set():
            logging.debug("Wakeword loop, waiting for command finising")
            await asyncio.sleep(0.1)
            continue
        try:
            chunk = await asyncio.wait_for(queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            continue
        detected = await ww_detector.predict_chunk(chunk)
        if detected:
            logging.info("Wakeword detected")
            waiting_for_command.set()
            asyncio.create_task(recog_task(queue, recognizer))
            await asyncio.sleep(3)


async def main():
    audio_q = asyncio.Queue()
    wakeword = WakeWordOWW(model_path=Path(oww.MODELS["hey_mycroft"]["model_path"]))
    recog = VOSKRecignizer(model_path="./vosk_model/vosk-model-small-ru-0.22", sample_rate=SAMPLE_RATE)
    receive_task = asyncio.create_task(stream_task(audio_q))
    ww_task = asyncio.create_task(wakeword_task(audio_q, wakeword, recog))
    await receive_task

if __name__ == "__main__":
    asyncio.run(main())
