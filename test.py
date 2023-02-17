import os
import tensorflow as tf

def predict(user_id):
    path = os.path.join("D:\Githhub Projects\Test Recommendation Model\model_v1")

    loaded = tf.keras.models.load_model(path,compile=False)

    scores, titles = loaded([str(user_id)])

    return titles

print(predict('A1N070NS9CJQ2I'))