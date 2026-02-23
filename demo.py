import asyncio
import logging
import json
import time
import sys

import pyaudio
import numpy as np
# import openwakeword
# from openwakeword.model import Model as WakeModel
from vosk import Model as VoskModel, KaldiRecognizer

from wakeword.openww import model, md_key
from command_service.dbus_commands import set_volume

logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)

# Wakeword model
ww_block = model

VOSK_MODEL_PATH = "./vosk_model/vosk-model-small-ru-0.22"
SAMPLE_RATE = 16000
CHUNK_SIZE = 1280  # openWakeWord рекомендует 1280 (80 мс при 16кГц)
# -----------------------------------------------

# Инициализация VOSK
vosk_model = VoskModel(VOSK_MODEL_PATH)
recognizer = KaldiRecognizer(vosk_model, SAMPLE_RATE)
recognizer.SetWords(False)  # не нужны детали слов

# Настройка PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE)

# Состояния
listening_for_wake = True  # True = ждём wakeword, False = слушаем команду
command_audio = []          # буфер для команды
last_voice_time = None      # время последнего голоса (для таймаута)
SILENCE_TIMEOUT = 3       # секунд тишины = конец команды

logger.info("Start listening...")

try:
    while True:
        # Читаем аудио с микрофона
        audio_chunk = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)

        if listening_for_wake:
            # Режим ожидания wakeword
            prediction = ww_block.predict(audio_int16)

            # Проверяем порог для нашего ключевого слова (по умолчанию 0.5)
            if prediction[md_key] > 0.5:
                logging.info("Wakeword detected, waiting for command!")
                listening_for_wake = False
                command_audio = []          # очищаем буфер команды
                last_voice_time = time.time()
                # Не сбрасываем recognizer? VOSK продолжит с того места, но лучше начать заново
                recognizer = KaldiRecognizer(vosk_model, SAMPLE_RATE)  # свежий recognizer

        else:
            # Режим распознавания команды через VOSK
            # Добавляем аудио в буфер (VOSK всё равно нужен непрерывный поток)
            command_audio.append(audio_int16)

            # Передаём в VOSK
            if recognizer.AcceptWaveform(audio_chunk):
                # Получили финальный результат (конец фразы)
                result = json.loads(recognizer.Result())
                text = result.get("text", "")
                if text:
                    logging.info(f"Command received: {text}")
                # Возвращаемся к прослушиванию wakeword
                listening_for_wake = True
                logging.info("Return to waiting mode")
            else:
                # Ещё не конец фразы, но можно проверить таймаут по тишине
                # Простейшая детекция тишины: если амплитуда мала, считаем тишиной
                if (np.max(np.abs(audio_int16)) < 500):
                    if (time.time() - last_voice_time > SILENCE_TIMEOUT):  # порог тишины
                        partial = json.loads(recognizer.PartialResult())
                        text = partial.get("partial", "")
                        if text:
                            logging.info(f"Command received (timeout): {text}")
                        listening_for_wake = True
                        logging.info("Return to waiting mode")
                # else:
                #     last_voice_time = time.time()  # обновляем время последнего голоса

except KeyboardInterrupt:
    logging.info("Shutdown...")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()