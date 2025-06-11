# FER_Pipeline

## Overview
FER_Pipeline is a facial expression recognition system that processes images or videos to detect and classify human emotions using advanced machine learning techniques. The project includes tools for image preprocessing, feature extraction, model training, inference, and results visualization.

## Features
- **Model Training:** Tools to train and evaluate various machine learning models.
- **Inference:** Real-time and batch predictions on input data.
- **Visualization:** Graphical display of prediction results and performance metrics.

## Pipeline preview

![Pipeline result](./Results/pipeline_samples/test2_output.jpg)

## DEMO
<p align="center">
  <img src="./Results/demos/demo.gif" alt="Demo run" width="239" height="400"/>
</p>

## Requirements
- Python 3.9+
- Dependencies listed in `requirements.txt`

## Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/username/FER_Pipeline.git
   ```
2. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download models checkpoints and / or datasets:**
Place the checkpoints in the correct directory in Models or adjust the loading path in pipeline. :arrow_right: [GoogleDrive CP](https://drive.google.com/drive/folders/10u2uMyfBatGi3vSSX9nMqy0iLtQPwDwN?usp=sharing)
For accessing datasets use the Google Drive provided in here. :arrow_right:  [GoogleDrive DS](https://drive.google.com/drive/folders/1sFIIpjxYzDPPdGXCrBJCEjeVX5QVyhGZ?usp=sharing)
4. **Navigate into the project directory and follow the README from there:**
   ```bash
   cd Detection_pipeline
   ```
5. **Read the ``usage.md`` then run ``pipeline.py`` accordingly.**

## Project Structure

- **Datasets**: Contains the externally downloaded datasets used in this project due to their large size.
- **Detection_Pipeline**: Have the core pipeline for face detection, the ensemble model module, and the ResEmoteNet architecture. It also includes an experimental pipeline and separate components for dedicated testing.
- **Evaluation**: Contains scripts for evaluating individual models as well as the ensemble performance.
- **Models**: Includes code for model training and a directory for storing model checkpoints.
- **Results**: Stores sample outputs including detection results on test images, evaluation graphs, CSV files of metrics, and examples of the pipeline applied to images. Additionally some demos are here.

## Benchmark results
#### Evaluation on FER2013 dataset
![FER2013](./Results/graphs/test_benchmark_fer2013_adv.png)
#### Evaluation on different datasets
![CrossDB](./Results/graphs/crossdb_evaluation.png)
#### Advanced evaluation (performance vs speed vs size)
![Benchmark Results](./Results/graphs/performance_vs_speed_vs_size.png)

## License
Specify your project license here or refer to the [LICENSE](LICENSE) file.