# Financial Transaction Matching with Siamese NNs

This project demonstrates how to use Siamese Neural Networks to match financial transactions (between internal ledgers and bank statements) w/ pytorch. The model uses text embeddings and numeric features to identify matching and non-matching transaction pairs effectively.

## project structure

```
.
├── data/
│   ├── ledger_transactions_real.csv   # ledger transactions dataset
│   └── bank_transactions_real.csv     # bank transactions dataset (Note: Data was created manually)
│
├── model/
│   ├── model.py                       # pytorch siamese network definition
│   └── siamese_model.pth              # weights from trained model
│
├── scripts/
│   ├── generate_data.py               # script to create data
│   └── train.py                       # model training script
│
├── api/
│   └── app.py                         # pyflask API to serve model predictions
│
├── requirements.txt                   # dependencies
└── README.txt                         # hi
```

## Running the Project

### Setup

1. Clone repo.
2. Create and activate a Python venv:

python3 -m venv venv
source venv/bin/activate

3. Install dependencies:

pip install -r requirements.txt

### Training the Model

Run the training script

python scripts/train.py

### Serving the Model

To run the Flask API locally:

```bash
python api/app.py
```

Send POST requests to `http://127.0.0.1:5000/predict` to get transaction match predictions.

---

Abishek Shankara
