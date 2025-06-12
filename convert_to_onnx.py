# convert_to_onnx.py
 
import torch
import torch.nn as nn
from pytorch_model import Classifier, BasicBlock
 
def ConvertToOnnx(model_path, onnx_path):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Classifier(
            block=BasicBlock,
            layers=[2, 2, 2, 2],
            num_classes=1000
        ).to(device)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.eval() 
        wrapped = nn.Sequential(model, nn.Softmax(dim=1)).to(device)
        dummy = torch.randn(1, 3, 224, 224, device=device)
        torch.onnx.export(
            wrapped,
            dummy,
            onnx_path,
            input_names=["input"],
            output_names=["probabilities"],
            opset_version=11
        )
        return True
    except Exception as e:
        print(e)
        return False