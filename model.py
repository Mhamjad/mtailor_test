import os
import sys
import onnxruntime as ort
import numpy as np
from PIL import Image

from utils import ReadKeyMapValueFromFile
from convert_to_onnx import ConvertToOnnx

class Preprocessor:
    def __init__(self):
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.size = (224, 224)

    def SetMeanValue(self, value):
        self.mean = np.array(value, dtype=np.float32)
    
    def SetStdValue(self, value):
        self.std = np.array(value, dtype=np.float32)
        
    def FromFile(self, path):
        if os.path.exists(path):
            img = Image.open(path).convert("RGB")
            return self.DoPreprocessing(img)
        else:
            print("Given Image Path not found.")
            return np.array([])

    def FromData(self, img):
        return self.DoPreprocessing(img)
    
    def DoPreprocessing(self, img):
        img = img.resize( self.size, Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        ### HWC to CHW
        return np.transpose(arr, (2, 0, 1))[None, ...]
        
 
class OnnxClassifier:
    def __init__(self, model_path: str = "/app/classifier.onnx"):
        self.classes = ReadKeyMapValueFromFile("/app/classes.txt")
        self.onnx_path = model_path
        self.ConvertPytorchToOnnx(model_path)
        self.session = self.CreateOnnxSession(self.onnx_path)
    
    def ConvertPytorchToOnnx(self, model_path):
        if os.path.exists(model_path):
            if model_path[-4:] != "onnx":
                res = ConvertToOnnx(model_path, "/app/classifier.onnx")
                if not res:
                    print("Unable to convert to Onnx")
                    sys.exit(1)
                self.onnx_path = "/app/classifier.onnx"
        else:
            print("Given model path is not valid")
            sys.exit(1)
    
    def CreateOnnxSession(self, onnx_path):
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            print("Using GPU for ONNX inference.")
            return ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider"])
        else:
            print("CUDA not available, using CPU for ONNX inference.")
            return ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        
    def Predict(self, img_tensor):
        result = self.session.run(
            ["probabilities"],
            {"input": img_tensor.astype(np.float32)}
        )
        return result[0]

    def GetOutput(self, img_tensor):
        if img_tensor.size == 0:
            return "Unknown", 0.0
        result = self.Predict(img_tensor)
        max_index, max_value = max(enumerate(result[0]), key=lambda x: x[1])
        return self.classes[max_index], max_value
        
        