import onnxruntime as ort
import numpy as np
from PIL import Image
 
from utils import ReadKeyMapValueFromFile
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
        img = Image.open(path).convert("RGB")
        return self.DoPreprocessing(img)
    
    def DoPreprocessing(self, img):
        img = img.resize( self.size, Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        ### HWC to CHW
        return np.transpose(arr, (2, 0, 1))[None, ...]
        
 
class OnnxClassifier:
    def __init__(self, model_path: str = "/app/classifier.onnx"):
        self.session = ort.InferenceSession(model_path)
        self.classes = ReadKeyMapValueFromFile("/app/classes.txt")
        
    def Predict(self, img_tensor):
        result = self.session.run(
            ["probabilities"],
            {"input": img_tensor}
        )
        return result[0]

    def GetOutput(self, img_tensor):
        result = self.Predict(img_tensor)
        max_index, max_value = max(enumerate(result[0]), key=lambda x: x[1])
        return self.classes[max_index], max_value
        
        