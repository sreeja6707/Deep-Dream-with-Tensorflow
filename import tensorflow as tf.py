import tensorflow as tf
import numpy as np
import PIL.Image
import io
import base64
import webbrowser
import time

# Define a different URL if the original one is inaccessible
url = 'https://i.pinimg.com/736x/d6/69/8b/d6698bc9bc07783a4826bb650269f488.jpg'  # Replace with a working URL

def download(url, max_dim=None):
    try:
        name = url.split('/')[-1]
        image_path = tf.keras.utils.get_file(name, origin=url)
        img = PIL.Image.open(image_path)
        if max_dim:
            img.thumbnail((max_dim, max_dim))
        return np.array(img)
    except Exception as e:
        print(f"Failed to download image from {url}: {e}")
        return None

def deprocess(img):
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)  # Ensure the tensor is float32
    img = 255 * (img + 1.0) / 2.0
    return tf.cast(img, tf.uint8)  # Convert the final result to uint8

def save_image(img, filename='output_image.jpg'):
    img = PIL.Image.fromarray(np.array(img))
    img.save(filename)

def run_deep_dream_simple(img, steps=100, step_size=0.01):
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    step_size = tf.convert_to_tensor(step_size)
    steps_remaining = steps
    step = 0
    while steps_remaining > 0:
        run_steps = min(100, steps_remaining)
        steps_remaining -= run_steps
        step += run_steps

        loss, img = deepdream(img, run_steps, step_size)
        img = deprocess(img)
    return img

def display_image_in_browser(img):
    img_base64 = image_to_base64(img)
    html = f'<html><body><img src="data:image/jpeg;base64,{img_base64}"></body></html>'
    with open('temp_image.html', 'w') as f:
        f.write(html)
    webbrowser.open('temp_image.html')

def image_to_base64(img):
    buffered = io.BytesIO()
    img = PIL.Image.fromarray(np.array(img))
    img.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_base64

def calc_loss(img, model):
    img_batch = tf.expand_dims(img, axis=0)
    layer_activations = model(img_batch)
    if len(layer_activations) == 1:
        layer_activations = [layer_activations]

    losses = []
    for act in layer_activations:
        loss = tf.math.reduce_mean(act)
        losses.append(loss)

    return tf.reduce_sum(losses)

class DeepDream(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(input_signature=(
        tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.int32),
        tf.TensorSpec(shape=[], dtype=tf.float32),
    ))
    def __call__(self, img, steps, step_size):
        print("Tracing")
        loss = tf.constant(0.0)
        for _ in tf.range(steps):
            with tf.GradientTape() as tape:
                tape.watch(img)
                loss = calc_loss(img, self.model)

            gradients = tape.gradient(loss, img)
            gradients /= tf.math.reduce_std(gradients) + 1e-8
            img = img + gradients * step_size
            img = tf.clip_by_value(img, -1, 1)

        return loss, img

# Main execution
original_img = download(url, max_dim=500)
if original_img is not None:
    save_image(original_img, 'original_image.jpg')

    base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    names = ['mixed3', 'mixed5']
    layers = [base_model.get_layer(name).output for name in names]
    dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)

    deepdream = DeepDream(dream_model)
    dream_img = run_deep_dream_simple(img=original_img, steps=100, step_size=0.01)
    save_image(dream_img, 'dream_image.jpg')

    display_image_in_browser(dream_img)

    start = time.time()

    OCTAVE_SCALE = 1.30
    img = tf.constant(np.array(original_img), dtype=tf.float32)
    base_shape = tf.shape(img)[:-1]
    float_base_shape = tf.cast(base_shape, tf.float32)

    for n in range(-2, 3):
        new_shape = tf.cast(float_base_shape * (OCTAVE_SCALE ** n), tf.int32)
        img = tf.image.resize(img, new_shape)
        img = run_deep_dream_simple(img=img, steps=100, step_size=0.01)

    display_img = tf.image.resize(img, base_shape)
    display_img = tf.image.convert_image_dtype(display_img / 255.0, dtype=tf.uint8)
    display_image_in_browser(display_img)

    end = time.time()
    print(f"Execution time: {end - start} seconds")
else:
    print("Image download failed. Exiting.")
