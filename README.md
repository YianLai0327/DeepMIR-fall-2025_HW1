# DeepMIR-fall-2025_HW1

# Task 1 - Audio Classification with SVM

## Setup
Install the required packages:
```bash
pip install -r requirements.txt
```
Also, install the trained models by execute the bash file:
```bash
bash get_models.sh
```

# Task 1 - Train a Traditional Machine Learning Model

## Inference
Run inference with:
```bash
python3 task1_inference.py \
    --audios_dir <path_to_audio_folder> \
    --model_pth <path_to_model.pkl> \
    --output_pth <path_to_output.json>
```

## Arguments
`audios_dir`: directory containing the input audio files for prediction

`model_pth`: path to the saved model (.pkl)

`output_pth`: path where the prediction JSON file will be saved

# Task 2 - Train a Deep Learning Model

## Inference
Run inference with:
```bash
python3 task2_inference.py

```

## Arguments
`audios_dir`: directory containing the input audio files for prediction

`model_pth`: path to the saved model (.pth)

`output_pth`: path where the prediction JSON file will be saved