from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

model = pickle.load(open("learning_style_model.pkl", "rb"))

@app.route("/")
def home():
    return "LearnMate AI API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]
    prediction = model.predict([data])
    return jsonify({"learning_style": str(prediction[0])})

if __name__ == "__main__":
    app.run()
