# DeepMIR-fall-2025_HW1

## Task 1 – Audio Classification with SVM

### Setup
Install all required packages:
```bash
pip install -r requirements.txt
```

Download the pre-trained models:
```bash
bash get_models.sh
```

### Inference
Run inference using the command below:
```bash
python3 task1_inference.py \
    --audios_dir <path_to_audio_folder> \
    --model_pth <path_to_model.pkl> \
    --output_pth <path_to_output.json>
```

#### Arguments
- `audios_dir`: Directory containing the input audio files for prediction  
- `model_pth`: Path to the saved model file (.pkl)  
- `output_pth`: Path where the prediction JSON file will be saved  

---

## Task 2 – Audio Classification with Deep Learning

### Inference
Run inference using the command below:
```bash
python3 task2_inference.py \
    --model_path <path_to_model.pth> \
    --input_json <path_to_input_json_or_audio_dir> \
    --prediction_json <path_to_output.json>
```

#### Arguments
- `model_path`: Path to the saved model file (.pth)  
- `input_json`: Path to either a JSON file listing input audio paths or a directory containing audio files  
- `prediction_json`: Path where the prediction JSON file will be saved