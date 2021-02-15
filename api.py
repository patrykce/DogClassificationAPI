import base64

import flask
from flask import jsonify
import dog_breed_recognition
from PIL import Image
from flask import request
from io import BytesIO
import tensorflow as tf
import pandas as pd
import os

from constants import HOST, PORT, MODEL_FILENAME

app = flask.Flask(__name__)
app.config["DEBUG"] = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

model = tf.keras.models.load_model(MODEL_FILENAME)
path = "Dog Breed Identification/"
labels_path = os.path.join(path, "labels.csv")

labels_df = pd.read_csv(labels_path)
breed = labels_df["breed"].unique()

breeds_labels = {i: name for i, name in enumerate(breed)}


@app.route('/', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        content = request.get_json()
        base64_message = content["image"]
        base64_bytes = base64_message.encode('ascii')
        message_bytes = base64.b64decode(base64_bytes)
        stream = BytesIO(message_bytes)

        image = Image.open(stream).convert("RGB")
        stream.close()
        image.show()
        image.save("dog.jpg")
        result = dog_breed_recognition.recogniseImg("dog.jpg", model, breeds_labels)
        return jsonify(
            breed_1=result[0].breed,
            probability_1=result[0].probability,
            breed_2=result[1].breed,
            probability_2=result[1].probability,
            breed_3=result[2].breed,
            probability_3=result[2].probability,
        )
    else:
        return "<h2>not implemented handling to this request</h2>"


app.run(host=HOST, port=PORT)