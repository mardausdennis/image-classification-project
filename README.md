# Image Classification Project

This project is for the Programming in Python II course, focusing on building a machine learning model to classify images into 20 different categories.

## Project Structure
- `data/`: Contains the dataset files (images and labels).
- `models/`: Contains the saved model files.
- `src/`: Contains the source code files.
  - `__init__.py`: Makes this directory a Python package.
  - `architecture.py`: Contains the model architecture.
  - `dataset.py`: Contains data loading and preprocessing code.
  - `evaluate.py`: Contains code for model evaluation.
  - `train.py`: Contains code for training the model.
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

3. Download and place the dataset in the `data/` directory.

## Usage
- To train the model, run:
    ```bash
    python src/train.py
    ```

- To evaluate the model, run:
    ```bash
    python src/evaluate.py
    ```