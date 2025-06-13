# MTailor Test

A image classification project which run on cerebrium and serve endpoint for image classifications

## Project Structure
mtailor_test/

├── samples/ #sample image files

├── weights/ #pytorch weights which will be converted to onnx for this project.

├── requirements.txt #Project dependencies

├── README.md #Project overview and instructions

├── Dokcerfile #Dockerfile which will make container and start fastapi server at 8080 at local host

└── .gitignore #Ignored files and folders

└── app.py #FastApi based endpoint which serves results.

└── convert_to_onnx.py #A file responsible to convert pytorch weights to onnx

└── model.py #Responsible to do preprocessing and run onnx inference

├── pytorch_model.py # provided file

├── test.py #A basic test cases to test system is running fine or not.

├── utils.py 

├── test_server.py #A test file which will request fastapi to get results from service deployed on cerebrium

## Requirements
1: Ubuntu OS
2: You need to have Docker in your host enviroment
3: Image Classification model. Can be download from this path 'https://www.dropbox.com/scl/fi/73oon7vuthlsihcogcgga/pytorch_model_weights.pth?rlkey=og32lr3cme04a3tbk64tr3znx&e=1&dl=0'

## How to run
```
git clone https://github.com/Mhamjad/mtailor_test
cd mtailor_test
docker build -t img_classification:latest -f Dockerfile .
// run to serve as api endpoint
docker run -p 8080:8080 img_classification:latest 
```

## How to test
Once the above command is running successfully
```
curl -X POST http://localhost:8080/predict   -H "accept: application/json"   -H "Content-Type: multipart/form-data"   -F "file=@path-to-image"
```
You will get reponse as below
```
{"class":"mud turtle","confidence":0.6958247423171997}
```

## Test Via test_server
As service is also deployed on cerebrium, so you can use test_service.py file to test images.

## Module Overview
This whole provides a simple pipeline for image classification using an ONNX model. It includes image preprocessing, ONNX model conversion (from PyTorch if needed), inference execution, and output in json format.

## model.py

### Preprocessor Class

Handles image normalization and formatting before feeding into the ONNX model.

- **Attributes**:
  - `mean`: Mean pixel values for normalization (default: `[0.485, 0.456, 0.406]`)
  - `std`: Standard deviation values (default: `[0.229, 0.224, 0.225]`)
  - `size`: Target image size (default: `(224, 224)`)

- **Methods**:
  - `SetMeanValue(value)`: Update the normalization mean.
  - `SetStdValue(value)`: Update the normalization std.
  - `FromFile(path)`: Load and preprocess an image from a file path.
  - `FromData(img)`: Preprocess a PIL Image object directly.
  - `DoPreprocessing(img)`: Resize, normalize, and convert image to CHW format with batch dimension.


### OnnxClassifier Class

Encapsulates loading, running, and post-processing of an ONNX model for classification tasks.

- **Attributes**:
  - `classes`: A dictionary mapping class indices to human-readable labels.
  - `onnx_path`: Path to the ONNX model file.

- **Initialization**:
  - Automatically converts a given PyTorch model to ONNX if required
---

## convert_to_onnx.py

This module provides a utility function to convert a PyTorch classification model into the ONNX format for optimized and hardware-independent inference.

### Function: `ConvertToOnnx(model_path, onnx_path)`

Converts a PyTorch model to an ONNX file.

#### Parameters:

- `model_path` *(str)*: Path to the saved PyTorch `.pt` or `.pth` model weights.
- `onnx_path` *(str)*: Target path where the `.onnx` file will be saved.
