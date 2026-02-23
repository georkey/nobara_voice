import asyncio
from abc import ABC, abstractmethod
import logging
from pathlib import Path
import time
import json

from vosk import Model, KaldiRecognizer

logger = logging.getLogger()

class BaseRecognizer(ABC):

    @abstractmethod
    async def recognize(self, queue: asyncio.Queue) -> str:
        pass


class VOSKRecignizer(BaseRecognizer):

    def __init__(self, model_path: str | Path, sample_rate: int, timeout: float=3.0):
        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, sample_rate)
        self.recognizer.SetWords(False)
        self.timeout = timeout
        self.sample_rate = sample_rate

    async def recognize(self, queue: asyncio.Queue) -> str:
        start = time.time()
        command_chunks = []
        while time.time() - start < self.timeout:
            try:
                chunk = await asyncio.wait_for(queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                logging.debug("Empty audio queue")
                return ""
            result = self.recognizer.AcceptWaveform(chunk)
            if result:
                command_chunks.append(json.loads(self.recognizer.Result()).get("text", ""))
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
        self.recognizer.SetWords(False)
        return " ".join(command_chunks)
