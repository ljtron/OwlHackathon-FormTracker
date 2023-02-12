# Import TF and TF Hub libraries.
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np


# Download the model from TF Hub.
model = hub.load('https://tfhub.dev/google/movenet/singlepose/thunder/3')
movenet = model.signatures['serving_default']

# Threshold for 
threshold = .3

# Loads video source (0 is for main webcam)
video_source = 0
cap = cv2.VideoCapture(video_source)

# Checks errors while opening the Video Capture
if not cap.isOpened():
    print('Error loading video')
    quit()


success, img = cap.read()

if not success:
    print('Error reding frame')
    quit()

y, x, _ = img.shape

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle

while success:
    # A frame of video or an image, represented as an int32 tensor of shape: 256x256x3. Channels order: RGB with values in [0, 255].
    tf_img = cv2.resize(img, (256,256))
    tf_img = cv2.cvtColor(tf_img, cv2.COLOR_BGR2RGB)
    tf_img = np.asarray(tf_img)
    tf_img = np.expand_dims(tf_img,axis=0)

    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    image = tf.cast(tf_img, dtype=tf.int32)

    # Run model inference.
    outputs = movenet(image)
    # Output is a [1, 1, 17, 3] tensor.
    keypoints = outputs['output_0']

    # print("right elbow angle: " + str(calculate_angle(
    #     [y * keypoints[0][0][10][0], x * keypoints[0][0][10][1]],
    #     [y * keypoints[0][0][8][0], x * keypoints[0][0][8][1]],
    #     [y * keypoints[0][0][6][0], x * keypoints[0][0][6][1]]
    # )))

    rightLegAngle = calculate_angle(
        [y * keypoints[0][0][16][0], x * keypoints[0][0][16][1]],
        [y * keypoints[0][0][14][0], x * keypoints[0][0][14][1]],
        [y * keypoints[0][0][12][0], x * keypoints[0][0][12][1]]
    )

    leftLegAngle = calculate_angle(
        [y * keypoints[0][0][15][0], x * keypoints[0][0][15][1]],
        [y * keypoints[0][0][13][0], x * keypoints[0][0][13][1]],
        [y * keypoints[0][0][11][0], x * keypoints[0][0][11][1]]
    )

    # if(      (keypoints[0][0][16][2] >=.5 or keypoints[0][0][15][2] >=.5) and 
    #       (keypoints[0][0][14][2] >=.5 or keypoints[0][0][13][2] >= .5) and
    #       (keypoints[0][0][12][2] >=.5 or keypoints[0][0][11][2] >= .5)):
    #     print("right leg angle: " + str(rightLegAngle))
    #     print("left leg angle: " + str(leftLegAngle))
    if(
        ((rightLegAngle <= 90 and rightLegAngle >84) or 
         (leftLegAngle <= 90 and rightLegAngle >84)) and 
         (keypoints[0][0][16][2] >=.5 or keypoints[0][0][15][2] >=.5) and 
         (keypoints[0][0][14][2] >=.5 or keypoints[0][0][13][2] >= .5) and
         (keypoints[0][0][12][2] >=.5 or keypoints[0][0][11][2] >= .5)):
        print("good form")
    # iterate through keypoints
    for k in keypoints[0,0,:,:]:
        # Converts to numpy array
        k = k.numpy()

        # Checks confidence for keypoint
        if k[2] > threshold:
            # The first two channels of the last dimension represents the yx coordinates (normalized to image frame, i.e. range in [0.0, 1.0]) of the 17 keypoints
            yc = int(k[0] * y)
            xc = int(k[1] * x)

            # Draws a circle on the image for each keypoint
            img = cv2.circle(img, (xc, yc), 2, (0, 255, 0), 5)
    
    # Shows image
    cv2.imshow('Movenet', img)
    # Waits for the next frame, checks if q was pressed to quit
    if cv2.waitKey(1) == ord("q"):
        break

    # Reads next frame
    success, img = cap.read()

cap.release()