from PIL import Image # Assuming you already have Annotator defined
from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator
from torchreid.utils import FeatureExtractor
import os
import numpy as np
from scipy.spatial.distance import cosine
import math
import re
import copy


model = YOLO("yolov8m.pt")

feature_extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    model_path='reid_weights\\osnet_x0_25_market1501.pth',
    device='cpu'
)

last_coords = {}
video_path = "inputs\\footballer_camera.mp4"

#----methods----------------------------------------------------#
def get_boxes(result):
  boxes = []
  for r in result:
    for box in r.boxes:
      c = box.cls
      if int(c)!=0:
        continue
      boxes.append(box.xyxy[0].tolist())
  return boxes

def crop_detections(boxes,frame,frame_num):
  paths = []
  os.makedirs(f"content\\crops\\frame{frame_num:05d}",exist_ok=True)
  frame = Image.fromarray(frame)
  for i,box in enumerate(boxes):
    cropped_region = frame.crop(box)
    path = f"content\\crops\\frame{frame_num:05d}\\img{i:03d}.png"
    cropped_region.save(path)
    paths.append(path)
  return paths

def similarity_score(feature1,feature2):
  return 1-cosine(feature1,feature2)

def frame_annotator(frame,frame_num,boxids,boxes):
  annotated_frame = frame.copy()
  os.makedirs(f"content\\frames",exist_ok=True)
  for index, data in enumerate(zip(boxids, boxes)):
    np.random.seed(data[0])
    print("seed:",data[0])
    color = tuple(np.random.randint(0, 256, size=3).tolist())
    x1, y1, x2, y2 = data[1]
    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
  annotated_image = Image.fromarray(annotated_frame)
  annotated_image.save(f"content\\frames\\{frame_num:05d}.png")

def euclidian_distance(box1, box2):
  box1_centerX = int((box1[0]+box1[2])/2)
  box1_centerY = int((box1[1]+box1[3])/2)
  box2_centerX = int((box2[0]+box2[2])/2)
  box2_centerY = int((box2[1]+box2[3])/2)

  x_diff = math.pow(box2_centerX-box1_centerX,2)
  y_diff = math.pow(box2_centerY-box1_centerY,2)
  distance = math.pow(x_diff+y_diff,1/2)

  return distance

def update_similarity(similarity,distance,w,h,fps):
  avg_boundry = (w+h)/2
  R = int(avg_boundry/fps)

  add_factor = 0.3 if similarity<=0.65 else (0.95-similarity)
  if distance<=R:
    similarity += add_factor/(1+R)
  # elif distance<=5*R:
  #   similarity += add_factor/(1+math.pow(np.e,R))
  # else:
  #   similarity -= add_factor/(1-1/(math.pow(np.e,R)))
  
  return similarity

def feature_comparison(boxes,features,w,h,fps):
  global identified
  global MAXID
  global last_coords
  boxids = []
  identified_copy = copy.deepcopy(identified)
  # print("identified_copy:",identified_copy)
  # print("inner boxes len:",len(boxes))
  for index, cropdata in enumerate(zip(boxes,features)):
    max_similarity = [-1,0.0]
    # print("OUTER_COPY:",len(identified_copy))
    
    match_found = False
    for id in identified_copy:
      # print("INNER_COPY:",len(identified_copy))
      # print("spooky")
      similarity = similarity_score(identified_copy[id].cpu(),cropdata[1].cpu())
      if id in last_coords:
        last_box = last_coords[id]
        current_box = cropdata[0]
        distance = euclidian_distance(last_box,current_box)
        similarity = update_similarity(similarity=similarity,distance=distance,w=w,h=h,fps=fps)
      if similarity > max_similarity[1]:
        max_similarity[0] = id
        max_similarity[1] = similarity

    if max_similarity[1]>=0.7:
      # print("max_similarity:",max_similarity[1])
      match_found = True
      boxids.append(max_similarity[0])
      last_coords[id] = cropdata[0]
      identified_copy.pop(max_similarity[0])

    if not match_found:
      identified[MAXID] = cropdata[1]
      boxids.append(MAXID)
      last_coords[MAXID] = cropdata[0]
      MAXID+=1

  return boxids, boxes

def create_video(image_dir, frame_rate, height, width, output_filename='output.mp4'):
    """
    Creates an MP4 video from image frames within a directory.

    Args:
        image_dir: Path to the directory containing image frames.
        frame_rate: Desired frame rate (frames per second) of the output video.
        height: Height of the video frames in pixels.
        width: Width of the video frames in pixels.
        output_filename: Name for the output MP4 video file.
    """

    img_array = []
    for filename in sorted(os.listdir(image_dir)):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # Adjust extensions if needed
            img_path = os.path.join(image_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error loading image: {filename}")
                continue
            img_array.append(img)

    # Define the video codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For MP4 videos

    # Create the video writer with the specified properties
    out = cv2.VideoWriter(output_filename, fourcc, frame_rate, (width, height))

    # Write each frame to the video
    for img in img_array:
        out.write(img)

    out.release()
    print(f"Video created: {output_filename}")


cap = cv2.VideoCapture(video_path)
if not cap:
  print("error opening the video...")
  exit()
identified = {}
MAXID = 0
FPS = cap.get(cv2.CAP_PROP_FPS)
print("Frame rate:",FPS)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"resolution:{width}x{height}")

frame_num = 0
while True:
  rtrn, frame = cap.read()

  if not rtrn:
    break
  # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
  boxes = get_boxes(model(frame))
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  print("boxlen:",len(boxes))
  paths = crop_detections(boxes,frame,frame_num)
  print(paths)
  print("pathlength:",len(paths))
  pathlen = len(paths)
  if pathlen<=1:
    continue
  features = feature_extractor(paths)
  # print("featurelen:",len(features))
  boxids, annotation_boxes = feature_comparison(boxes,features,width,height,FPS)
  # print("lengths:",len(boxids))
  frame_annotator(frame,frame_num,boxids, annotation_boxes)
  frame_num+=1
  
  if frame_num==500:
    break


create_video("content\\frames",FPS,height,width,output_filename = "football_camera_out3.mp4")
import shutil
shutil.rmtree("content\\crops")
shutil.rmtree("content\\frames")