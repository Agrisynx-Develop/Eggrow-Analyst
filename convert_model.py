import tensorflow as tf

# load model lama
model = tf.keras.models.load_model("model/eggrow_vision_model.keras", compile=False)

# save ke H5
model.save("model/model_fix.h5")

print("✅ Model berhasil di-convert")
