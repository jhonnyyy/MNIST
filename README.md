# MNIST ML Project with CI/CD

This project implements a 3-layer DNN for MNIST classification with automated testing and deployment.

## Project Structure
- `model.py`: Contains the neural network architecture
- `train.py`: Training script
- `test.py`: Testing script with validation checks
- `requirements.txt`: Project dependencies

## Local Setup

1. Clone the repository: 

bash 
git clone <repository_url>
cd <repo-name>

2. Create and activate virtual environment:

bash 
python -m venv .venv
source .venv/bin/activate

3. Install dependencies:

bash 
pip install -r requirements.txt

4. Run tests:

bash 
pytest test.py

5. Train the model:

bash 
python train.py

## CI/CD Pipeline
The project uses GitHub Actions for continuous integration and deployment. The pipeline:
- Runs on every push to main branch
- Tests model architecture and performance
- Trains the model
- Saves the trained model as an artifact

## Model Requirements
- Parameters < 25,000
- Input size: 28x28
- Output size: 10 (digits 0-9)
- Accuracy > 95%
