# Efficient CNN-Based System for Automated Beetle Elytra Coordinates Prediction

This repository contains code and resources for our deep learning framework to **localize beetles in images** (object detection) and **predict elytra coordinates** (regression), enabling automated measurement of beetle morphological traits.

---

## 1. Installation

- **Python Version**: 3.8 or higher required  
- **Dependencies**:  
  - [PyTorch](https://pytorch.org/)  
  - [OpenCV](https://opencv.org/)  
  - [tqdm](https://github.com/tqdm/tqdm)

Install via pip:
```bash
pip install torch opencv-python tqdm
```

## 2. Dataset
Our dataset is hosted on **Hugging Face** for easy download and version control. \
[Hugging Face Dataset Link](https://huggingface.co/datasets/yoohj0416/predictbeetle)

It includes:
- High-resolution images containing multiple beetles (grouped and individual).
- Corresponding bounding box annotations for object detection.
- Manually annotated elytra coordinates for regression tasks.

It is **re-created** from the original beetle dataset found here: \
[2018-NEON-beetles](https://huggingface.co/datasets/imageomics/2018-NEON-beetles)

## 3. Pre-trained Models
We provide two sets of pre-trained models:

1. Object Detection Models (YOLO)
2. Regression Models (for beetle elytra coordinates)

## 3.1. Object Detection (YOLO)

Below is a summary of **YOLOv8** models we trained (or fine-tuned) for beetle detection, along with their AP50 and mAP on our test set.

| Model | AP50 | mAP | Download Link |
| ----- | ---- | --- | ------------- |
|YOLOv8n|0.968|0.800|[Download](https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/yoo_515_buckeyemail_osu_edu/EaK46wLT91JLn-P7lQ3_zqABtVOC0jDQojpQvxPZwus97A?e=g0zsUz)|
|YOLOv8s|0.970|**0.805**|[Download](https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/yoo_515_buckeyemail_osu_edu/ERKTdw2_b7xMsefdVtUqNKEBs4Rit7-QguYgCTaGYG9YAA?e=3uZVG2)|
|YOLOv8m|**0.971**|0.804|[Download](https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/yoo_515_buckeyemail_osu_edu/EdQQwiRiQHNAkBhRMLlH5xMBe5rbv6M00l5hi-6PVABI0w?e=e0V9ge)|

For additional details on how to install and run YOLOv8, please refer to the official [Ultralytics YOLOv8 documentation](https://docs.ultralytics.com/quickstart/).

## 3.2 Regression Models

Below is a summary table for our **Regression Models** using various backbones. Each row includes a download link and its corresponding Mean Squared Error (MSE) and Points Defference performances on our test set.

| Backbone | MSE | Points Difference (cm) | Params | Download Link |
| -------- | --- | ---------------------- | ------ | ------------- |
|ResNet50|1.941E-03|0.128|23.5M|[Download](https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/yoo_515_buckeyemail_osu_edu/ERFLdPMjX35AqUcCP2XSfkMBmWsqnImLn05twt4jiYx8IA?e=vUsVck)|
|ResNet101|1.971E-03|0.132|42.5M|[Download](https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/yoo_515_buckeyemail_osu_edu/EasL9EdBxl5BiK6vNpQXGS0BFuqjDbtlGiloebnPhNx3fQ?e=1vcIyZ)|
|MobileNetV3-Large|1.952E-03|0.118|4.2M|[Download](https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/yoo_515_buckeyemail_osu_edu/EeVJyRKNPKhOuTmILIpmFNoBOxRiCnNz9kz9FkjWK6a5zw?e=ubxvGK)|
|EfficientNetV2-S|1.870E-03|**0.110**|20.2M|[Download](https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/yoo_515_buckeyemail_osu_edu/EUsNtMTUMc5OsaeoPT3FvHgBYiDhMWVefzLbcMTJErJApg?e=Jc3h6l)|
|EfficientNetV2-M|**1.756E-03**|**0.110**|52.9M|[Download](https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/yoo_515_buckeyemail_osu_edu/ERKpN7vb4iBCpOSv19ZxFOkBuYTNMCu4SqUp0SDyVCEgJw?e=ffsiIG)|

## 4. Train & Test
### A. Update the Config File
Before training or testing, edit your config file (e.g., `config.yaml`) and ensure:

```swift
dataset:
  source_dir: /absolute/path/to/individual/images
```
Here, `source_dir` should point to the folder where you downloaded and extracted the individual images from our dataset on Hugging Face.

### B. Train
Run:

```bash
python train.py --config /path/to/config/file
```

- Parses the specified config file
- Loads the dataset
- Initializes and trains the selected model (backbone, epochs, etc.)

### C. Test / Inference  
For inference or to evaluate a trained model, run:

```bash
python predict.py --config /path/to/config/file
```

- Loads the trained weights from the config
- Runs prediction on the test set or custom images
- Outputs bounding boxes or elytra keypoints, plus measurement logs

## 4. Citation
### BibTeX:

## 5. Acknowledgements
This work was supported by the NSF OAC 2118240 Imageomics Institute award and was initiated at Beetlepalooza 2024. More details about Beetlepalooza can be found on https://github.com/Imageomics/BeetlePalooza-2024.