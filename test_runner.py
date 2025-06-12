import pytest
import numpy as np
import torch
from PIL import Image
 
from pytorch_model import Classifier, BasicBlock
from model import Preprocessor, OnnxClassifier

sample_image_path = "/app/samples/n01440764_tench.jpeg"
onnx_path = "/app/classifier.onnx"
preprocessor = Preprocessor()
np_tensor = preprocessor.FromFile(sample_image_path)
onnx_classifier = OnnxClassifier(model_path=onnx_path)
out_class, out_score = onnx_classifier.GetOutput(np_tensor)
print(out_class)
