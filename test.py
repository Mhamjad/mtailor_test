import pytest
import numpy as np
import torch
from PIL import Image
 
from pytorch_model import Classifier, BasicBlock
from model import Preprocessor, OnnxClassifier
 
@pytest.fixture(scope="module")
def weights_path():
    return "/app/weights/pytorch_model_weights.pth"
 
@pytest.fixture(scope="module")
def onnx_path():
    return "/app/classifier.onnx"
 
@pytest.fixture(scope="module")
def sample_image_path():
    return "/app/samples/n01440764_tench.jpeg"
 
@pytest.fixture(scope="module")
def pytorch_model(weights_path):
    model = Classifier(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=1000)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model
 
@pytest.fixture(scope="module")
def onnx_classifier(onnx_path):
    return OnnxClassifier(model_path=onnx_path)
 
def test_pytorch_inference_output_shape_and_type(pytorch_model, sample_image_path):
    img = Image.open(sample_image_path).convert("RGB")
    tensor = pytorch_model.preprocess_numpy(img).unsqueeze(0)
    with torch.no_grad():
        logits = pytorch_model(tensor)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (1, 1000)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    assert probs.shape == (1, 1000)
    assert np.all(probs >= 0) and np.all(probs <= 1)
 
def test_onnx_inference_output_shape_and_type(onnx_classifier, sample_image_path):
    preprocessor = Preprocessor()
    np_tensor = preprocessor.FromFile(sample_image_path)
    out = onnx_classifier.Predict(np_tensor)
    assert isinstance(out, np.ndarray)
    assert out.shape == (1, 1000)
    assert np.all(out >= 0) and np.all(out <= 1)
 
def test_top1_prediction_matches(pytorch_model, onnx_classifier, sample_image_path):
    img = Image.open(sample_image_path).convert("RGB")
    pt_tensor = pytorch_model.preprocess_numpy(img).unsqueeze(0)
    with torch.no_grad():
        pt_probs = torch.softmax(pytorch_model(pt_tensor), dim=1).cpu().numpy()[0]
    pt_idx = int(np.argmax(pt_probs))
    preprocessor = Preprocessor()
    onnx_probs = onnx_classifier.Predict(preprocessor.FromFile(sample_image_path))[0]
    print(onnx_probs)
    onnx_idx = int(np.argmax(onnx_probs))
 
    assert pt_idx == onnx_idx, f"PyTorch predicted {pt_idx}, ONNX predicted {onnx_idx}"

def test_infered_class(sample_image_path):
    preprocessor = Preprocessor()
    np_tensor = preprocessor.FromFile(sample_image_path)
    onnx_classifier = OnnxClassifier(model_path="/app/classifier.onnx")
    out_class, out_score = onnx_classifier.GetOutput(np_tensor)
    assert out_class == "tench, Tinca tinca"