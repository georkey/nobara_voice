from abc import ABC, abstractmethod
import logging
from pathlib import Path

import numpy as np
import openwakeword as oww

logger = logging.getLogger(__name__)

class BaseWakeWord(ABC):
    """Abstract base class for all wakeword variants"""

    # @abstractmethod
    # def listen(self, stream: _Stream):
    #     pass

    # @abstractmethod
    # def invoke_cmd_service(self):
    #     pass
    @abstractmethod
    async def predict_chunk(self, chunk: bytes) -> bool:
        pass


class WakeWordOWW(BaseWakeWord):
    """Wakeword based on openWakeWord package"""

    def __init__(self, model_name: str="hey_mycroft", inference: str="onnx", 
                 model_path: Path | None=None, threshold: float=0.5):
        self.model_name = model_name
        if model_path is None:
            self.model_path = Path(oww.MODELS[model_name]["model_path"])
        else:
            self.model_path = model_path
        self.inference = inference
        if not self.model_path.exists():
            oww.utils.download_models([model_name])
        if inference == "onnx":
            self.model_path = Path(str(self.model_path).replace(".tflite", ".onnx"))
        self.model = oww.Model([str(self.model_path)], inference_framework=self.inference)
        self.listening = False
        self.threshold = threshold
        self.md_key = list(self.model.models.keys())[0]

    # def listen(self, stream: _Stream, chunksize=int) -> None:
    #     self.listening = True
    #     logging.debug("Start listening loop...")
    #     while self.listening:
    #         chunk = stream.read(chunksize, exception_on_overflow=False)
    #         signal = np.frombuffer(chunk, dtype=np.int16)
    #         prediction = self.model.predict(signal)
    #         if prediction[self.model_name] > self.threshold:
    #             logging.debug(f"Wakeword detected")
    #             self.listening = False
    #             self.invoke_cmd_service(stream)
    async def predict_chunk(self, chunk: bytes) -> bool:
        signal = np.frombuffer(chunk, dtype=np.int16)
        prediction = self.model.predict(signal)
        p = prediction[self.md_key]
        if p > self.threshold:
            logging.debug(f"Wakeword detected! Probability = {p}")
            return True
        return False



        

    


        






