from flask import Flask, request, jsonify
import torch
from torch.nn.utils.rnn import pad_sequence
from model import SiameseNetwork

app = Flask(__name__)

device = torch.device("cpu")
model = SiameseNetwork()
model.load_state_dict(torch.load('siamese_model.pth', map_location=device))
model.eval()

def encode_text(text):
    return torch.tensor([ord(c) % 1000 for c in text], dtype=torch.long)

def prep_input(text, amount_diff, date_diff):
    text_tensor = encode_text(text)
    text_padded = pad_sequence([text_tensor], batch_first=True, padding_value=0)
    numeric_tensor = torch.tensor([[amount_diff, date_diff]], dtype=torch.float)
    return text_padded, numeric_tensor

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    text1, amt_diff1, date_diff1 = data['text1'], data['amount_diff1'], data['date_diff1']
    text2, amt_diff2, date_diff2 = data['text2'], data['amount_diff2'], data['date_diff2']

    text1_t, numeric1 = prep_input(text1, amt_diff1, date_diff1)
    text2_t, numeric2 = prep_input(text2, amt_diff2, date_diff2)

    with torch.no_grad():
        out1, out2 = model(text1_t, numeric1, text2_t, numeric2)
        distance = torch.nn.functional.pairwise_distance(out1, out2).item()

    prediction = 'match' if distance < 0.5 else 'no_match'

    return jsonify({
        'distance': distance,
        'prediction': prediction
    })

if __name__ == '__main__':
    app.run(debug=True)
