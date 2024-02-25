#@markdown To better demonstrate the Pose Landmarker API, we have created a set of visualization tools that will be used in this colab. These will draw the landmarks on a detect person, as well as the expected connections between those markers.

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import mediapipe as mp
print(mp.__version__)
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
from google.colab.patches import cv2_imshow

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

def detect_pose(detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
  pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks_list[0]
  ])
  # Calculate the angle between the hip, knee, and ankle to determine if the person is standing, sitting, or laying down.
  hip_y = pose_landmarks_proto.landmark[solutions.pose.PoseLandmark.LEFT_HIP].y
  knee_y = pose_landmarks_proto.landmark[solutions.pose.PoseLandmark.LEFT_KNEE].y
  ankle_y = pose_landmarks_proto.landmark[solutions.pose.PoseLandmark.LEFT_ANKLE].y
  angle = np.degrees(np.arccos((knee_y - hip_y) * (ankle_y - knee_y) / (np.sqrt((knee_y - hip_y)**2 + (ankle_y - knee_y)**2) * np.sqrt((knee_y - hip_y)**2 + (ankle_y - knee_y)**2))))
  
  if angle < 65:
    print(angle)
    return "standing"
  elif angle < 95:
    print(angle)
    return "sitting"
  else:
    print(angle)
    return "laying down"


base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture('video.mp4')

# Get the frames per second (fps) of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Get the codec information of the video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Create a VideoWriter object to write the output video
out = cv2.VideoWriter('output.mp4', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
count = 0
while cap.isOpened():
  success, frame = cap.read()
  if not success:
    break
  count += 30
  cap.set(cv2.CAP_PROP_POS_FRAMES, count)

  # Convert the frame from BGR to RGB
  frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  # Convert to the expected uint8 data type
  frame_rgb = frame_rgb.astype(np.uint8)    

  # Convert to the expected uint8 data type
  frame_rgb = frame_rgb.astype(np.uint8)
  
  # Create the Image object
  image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb) 

  # STEP 4: Detect pose landmarks using the Image object
  detection_result = detector.detect(image)

  # STEP 5: Process the detection result. In this case, visualize it.
  annotated_image = draw_landmarks_on_image(frame_rgb, detection_result)
  cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

  pose = detect_pose(detection_result)
  print("The person is", pose)

# Add the pose label to the image
  cv2.putText(annotated_image, pose, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

  # Convert the annotated image back to BGR for writing to file
  annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

  # Write the frame into the output video file
  out.write(annotated_image_bgr)
   

cap.release()
cv2.destroyAllWindows()

