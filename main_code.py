from REID import REID, Person, crop, extractor
from typing import List, Dict, Optional,Tuple
import cv2
import numpy as np
import random
from dataclasses import dataclass
from ultralytics import YOLO
from ultralytics.engine.results import Results
@dataclass(frozen=True)
class VideoInfo:
  fps:float
  height:int
  width:int

video_path = "inputs\\vid2.mp4"
reid = REID(0.8,extractor)
cap=cv2.VideoCapture(video_path)
cap =  cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
video_info = VideoInfo(fps=fps, height=height,width=width)
model = YOLO("yolov8m.pt")
def generate_frames(cap):
    while True:
        success, frame = cap.read()
        if not success:
            break

        yield frame

iterator = iter(generate_frames(cap=cap)) 

def draw_rectangle(frame: np.ndarray, track_id: int, bbox: Tuple):
    """
    This method draws a rectangle on the frame with a color seeded from the track_id and returns the annotated frame.

    Args:
        frame: The frame to draw on. (numpy array)
        track_id: The track ID of the object. (int)
        bbox: The bounding box of the object in (x1, y1, x2, y2) format. (tuple)

    Returns:
        The frame with the rectangle drawn on it. (numpy array)
    """
    bbox = [int(val) for val in bbox]
    # Unpack the bounding box coordinates
    x1, y1, x2, y2 = bbox

    # Seed the random number generator with the track ID
    random.seed(track_id)

    # Generate random color components in the range 0-255
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    # Draw the rectangle on the frame
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)

    return frame

def create_video(frames, video_info, output_path="outputs/videoname1.mp4"):
    """
    Creates and stores a video from a list of NumPy frames.

    Args:
        frames (list): A list of NumPy arrays representing video frames.
        video_info (VideoInfo): An object containing FPS, height, and width information.
        output_path (str): The desired output path for the video file.
    """

    # Define the video codec (adjust if needed)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Create the video writer object
    print(video_info)
    out = cv2.VideoWriter(output_path, fourcc, int(video_info.fps), (int(video_info.width), int(video_info.height)))

    # Write each frame to the video
    for frame in frames:
        out.write(frame)
    # Release the video writer object
    out.release()

def annotate(frame:np.array, stracks:List[Person], videoinfo:VideoInfo):
  for index, track in enumerate(stracks):
    frame = draw_rectangle(frame,track.track_id,track.tlbr)
  return frame

def detections2boxes(results:Results):
  data = []
  for box,conf,cls in zip(results[0].boxes.xyxy.tolist(),results[0].boxes.conf.tolist(), results[0].boxes.cls.tolist()):
    if int(cls)!=0:
       continue
    box.append(conf)
    data.append(box)
  return np.array(data)

def main():
    frame_count = 0
    frames = []
    while True:
        try:
            frame = next(iterator)
        except StopIteration:
            print("Video ended")
            break
        
        results = model(frame)
        boxes = detections2boxes(results=results)
        persons = reid.update(boxes,frame=frame)
        for person in persons:
            # print("person boxes:",person.box)
            returned_frame = draw_rectangle(frame=frame,track_id=person.id,bbox=person.box)
            frames.append(returned_frame)
        if frame_count==600:
           break
        frame_count+=1
    
    create_video(frames=frames,video_info=video_info)
        
main()