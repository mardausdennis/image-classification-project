# Image Classification Project

This project is for the Programming in Python II course, focusing on building a machine learning model to classify images into 20 different categories.

## Project Structure
- `data/`: Contains the dataset files (images and labels).
  - `training_data/`: Directory where the training images and CSV file are stored.
  - `validation_indices.npy`: Numpy file storing the indices for the validation set.
- `models/`: Contains the saved model files.
- `src/`: Contains the source code files.
  - `architecture.py`: Contains the model architecture.
  - `dataset.py`: Contains data loading and preprocessing code.
  - `train.py`: Contains code for training the model.
  - `evaluate.py`: Contains code for evaluating the model.
- `requirements.txt`: Lists the dependencies required for the project.
- `README.md`: Project overview and setup instructions.

## Setup Instructions
1. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download and place the dataset in the `data/training_data/` directory.

## Usage
- To train the model, run:
    ```bash
    python src/train.py
    ```

- To evaluate the model, run:
    ```bash
    python src/evaluate.py
    ```

## Notes
- The dataset is expected to be in grayscale, and the model architecture and training process have been set up accordingly.
- The `validation_indices.npy` file is used to separate the training and validation datasets.
