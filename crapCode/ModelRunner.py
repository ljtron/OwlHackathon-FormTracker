# Import TF and TF Hub libraries.
import tensorflow as tf
import os
import numpy as np
import imageio
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
import matplotlib.image as mpimg

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the input image.
image_path = 'Model/WIN_20230211_13_41_43_Pro.jpg'
image = tf.io.read_file(image_path)
image = tf.compat.v1.image.decode_jpeg(image)
image = tf.expand_dims(image, axis=0)
# Resize and pad the image to keep the aspect ratio and fit the expected size.
image = tf.image.resize_with_pad(image, 256, 256)
def calculate(_inputImage):

    # Initialize the TFLite interpreter
    interpreter = tf.lite.Interpreter(model_path="Model/lite-model_movenet_singlepose_thunder_tflite_float16_4.tflite")
    interpreter.allocate_tensors()

    # TF Lite format expects tensor type of float32.
    input_image = tf.cast(_inputImage, dtype=tf.uint8)#tf.uint8
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())

    interpreter.invoke()

    # Output is a [1, 1, 17, 3] numpy array.
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    #return keypoints_with_scores
    print(keypoints_with_scores[0][0])

print(calculate(image))
def graph():
    img = mpimg.imread(image_path)
    imgplot = plt.imshow(img)
    plt.show()

#graph()
def imageRen(_image):
    input_size = 256
   # Resize and pad the image to keep the aspect ratio and fit the expected size.
    input_image = tf.expand_dims(_image, axis=0)
    input_image = tf.image.resize_with_pad(input_image, input_size, input_size)

    # Run model inference.
    keypoints_with_scores = calculate(input_image)

    # Visualize the predictions with image.
    display_image = tf.expand_dims(image, axis=0)
    display_image = tf.cast(tf.image.resize_with_pad(
        display_image, 1280, 1280), dtype=tf.int32)
    output_overlay = draw_prediction_on_image(
        np.squeeze(display_image.numpy(), axis=0), keypoints_with_scores)

    plt.figure(figsize=(5, 5))
    plt.imshow(output_overlay)
    _ = plt.axis('off')

def draw_prediction_on_image(
    image, person, crop_region=None, close_figure=True,
    keep_input_size=False):
  """Draws the keypoint predictions on image.
 
  Args:
    image: An numpy array with shape [height, width, channel] representing the
      pixel values of the input image.
    person: A person entity returned from the MoveNet.SinglePose model.
    close_figure: Whether to close the plt figure after the function returns.
    keep_input_size: Whether to keep the size of the input image.
 
  Returns:
    An numpy array with shape [out_height, out_width, channel] representing the
    image overlaid with keypoint predictions.
  """
  # Draw the detection result on top of the image.
  image_np = utils.visualize(image, [person])
  
  # Plot the image with detection results.
  height, width, channel = image.shape
  aspect_ratio = float(width) / height
  fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
  im = ax.imshow(image_np)
 
  if close_figure:
    plt.close(fig)
 
  if not keep_input_size:
    image_np = utils.keep_aspect_ratio_resizer(image_np, (512, 512))

  return image_np

#imageRen(image_path)
