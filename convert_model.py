import tensorflow as tf

# load model lama
model = tf.keras.models.load_model("model/eggrow_vision_model.h5", compile=False)

# rebuild model (INIT ulang graph)
model = tf.keras.models.clone_model(model)
model.set_weights(model.get_weights())

# save ulang bersih
model.save("model/clean_model.keras")
