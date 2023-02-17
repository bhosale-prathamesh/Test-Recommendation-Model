import os
import pandas as pd
import tensorflow as tf
from typing import Dict, Text
import numpy as np
import tensorflow_recommenders as tfrs

data = pd.read_csv('Electronics.csv',names=['ItemId','UserId','Rating','Timestamp'],nrows=1000000)
item = pd.DataFrame(np.unique(data['ItemId']),columns=['ItemId'])

data = tf.data.Dataset.from_tensor_slices(dict(data))
item = tf.data.Dataset.from_tensor_slices(dict(item))

ratings = data.map(lambda x: {
    "ItemId": x["ItemId"],
    "UserId": x["UserId"],
    "Rating": x["Rating"],
})

item = item.map(lambda x: x["ItemId"])

# Randomly shuffle data and split between train and test.
tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

item_id = item.batch(1_000)
user_ids = ratings.batch(1_000_000).map(lambda x: x["UserId"])

unique_item_ids = np.unique(np.concatenate(list(item_id)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))

class MovielensModel(tfrs.models.Model):

  def __init__(self, rating_weight: float, retrieval_weight: float) -> None:
    # We take the loss weights in the constructor: this allows us to instantiate
    # several model objects with different loss weights.

    super().__init__()

    embedding_dimension = 32

    # User and movie models.
    self.item_model: tf.keras.layers.Layer = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=unique_item_ids, mask_token=None),
      tf.keras.layers.Embedding(len(unique_item_ids) + 1, embedding_dimension)
    ])
    self.user_model: tf.keras.layers.Layer = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=unique_user_ids, mask_token=None),
      tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
    ])

    # A small model to take in user and movie embeddings and predict ratings.
    # We can make this as complicated as we want as long as we output a scalar
    # as our prediction.
    self.rating_model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1),
    ])

    # The tasks.
    self.rating_task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )
    self.retrieval_task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
            candidates=item.batch(128).map(self.item_model)
        )
    )

    # The loss weights.
    self.rating_weight = rating_weight
    self.retrieval_weight = retrieval_weight

  def call(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
    # We pick out the user features and pass them into the user model.
    user_embeddings = self.user_model(features["UserId"])
    # And pick out the movie features and pass them into the movie model.
    movie_embeddings = self.item_model(features["ItemId"])

    return (
        user_embeddings,
        movie_embeddings,
        self.rating_model(
            tf.concat([user_embeddings, movie_embeddings], axis=1)
        ),
    )

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:

    ratings = features.pop("Rating")

    user_embeddings, movie_embeddings, rating_predictions = self(features)

    # We compute the loss for each task.
    rating_loss = self.rating_task(
        labels=ratings,
        predictions=rating_predictions,
    )
    retrieval_loss = self.retrieval_task(user_embeddings, movie_embeddings)

    # And combine them using the loss weights.
    return (self.rating_weight * rating_loss

            + self.retrieval_weight * retrieval_loss)

model = MovielensModel(rating_weight=1.0, retrieval_weight=1.0)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

print(model.layers)

cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()

model.fit(cached_train, epochs=3)

index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
index = index.index_from_dataset(candidates=item.batch(100).map(lambda ItemId : (ItemId, model.item_model(ItemId))))
path = os.path.join("D:\Githhub Projects\Test Recommendation Model\model_v0")
_, titles = index(tf.constant(["42"]))
print(titles)
index.save(path,"model_v0")