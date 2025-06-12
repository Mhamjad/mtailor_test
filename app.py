import io
import uvicorn
from fastapi import FastAPI, File, UploadFile
from PIL import Image

from pytorch_model import Classifier, BasicBlock
from model import Preprocessor, OnnxClassifier

app = FastAPI()

ONNX_PATH = "/app/classifier.onnx"
preprocessor = Preprocessor()
onnx_classifier = OnnxClassifier(model_path=ONNX_PATH)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    np_tensor = preprocessor.FromData(image)
    out_class, out_score = onnx_classifier.GetOutput(np_tensor)
    result = {"class": str(out_class), "confidence": float(out_score)}
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)