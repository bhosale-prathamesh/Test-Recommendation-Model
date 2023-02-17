from flask import Flask
import tensorflow as tf
import os
from flask import request

app = Flask(__name__)

@app.route("/")
def predict():
    user_id = request.args.get('user_id', type = str)

    print(user_id)
    path = os.path.join("D:\Githhub Projects\Test Recommendation Model\model_v0")

    loaded = tf.keras.models.load_model(path)

    scores, titles = loaded([str(user_id)])

    return ' '.join(list(map(str,titles[0].numpy()))) + user_id

    # return "Hello, World!"
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)