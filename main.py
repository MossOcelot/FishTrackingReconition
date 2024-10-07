import torch
import cv2
from ultralytics import YOLO
from Tracking_methods.TrackingModel import Tracking
import math
import json
import uuid
from collections import deque
import random
import os
import torch

model = YOLO('Models/best-5.pt')
cap = cv2.VideoCapture('output_20241003.mp4')

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {device}")

tracks_in_pound = []

track_uniques_image = []

class TrackImage:
    def __init__(self, own_track):
        self.own_track = own_track
        self.image_path = [
                            deque(maxlen=5),
                            deque(maxlen=5),
                            deque(maxlen=5),
                            deque(maxlen=5)
                          ] # (path, step)
    
    def countImage(self):
        n = 0
        
        for deque in self.image_path:
            n += len(deque)
        
        return n
    
    def save(self):
        data = {
            "own_track": str(self.own_track),
            "image_path": []
        }
        
        for image_deque in self.image_path:
            data['image_path'].extend(list(image_deque))
        
        return data
    
    def add_image(self, own_track, step, i, cropped_object):
        path = f"Storage/ImageStorage/{own_track}_{step}_{i}.jpg"
        
        # เพิ่ม path ลงใน deque ตัวแรกทุก ๆ step
        if len(self.image_path[0]) == 5:
            # ถ้า deque ตัวแรกเต็ม ลบไฟล์จาก path เก่าก่อน
            old_image = self.image_path[0].popleft()
            if os.path.exists(old_image):
                os.remove(old_image)
        
        self.image_path[0].append(path)

        # เช็คว่า deque ตัวที่ 1 เต็มหรือไม่
        if len(self.image_path[0]) == 5:
            if step % 10 == 0:  # ทุก ๆ 10 step
                random_image = random.choice(self.image_path[0])
                self.image_path[0].remove(random_image)
                if len(self.image_path[1]) == 5:
                    # ถ้า deque ตัวที่ 2 เต็ม ลบไฟล์จาก path เก่าก่อน
                    old_image = self.image_path[1].popleft()
                    if os.path.exists(old_image):
                        os.remove(old_image)
                        
                self.image_path[1].append(random_image)

        # เช็คว่า deque ตัวที่ 2 เต็มหรือไม่
        if len(self.image_path[1]) == 5:
            if step % 100 == 0:  # ทุก ๆ 100 step
                random_image = random.choice(self.image_path[1])
                self.image_path[1].remove(random_image)
                if len(self.image_path[2]) == 5:
                    # ถ้า deque ตัวที่ 3 เต็ม ลบไฟล์จาก path เก่าก่อน
                    old_image = self.image_path[2].popleft()
                    if os.path.exists(old_image):
                        os.remove(old_image)
                        
                self.image_path[2].append(random_image)

        # เช็คว่า deque ตัวที่ 3 เต็มหรือไม่
        if len(self.image_path[2]) == 5:
            if step % 100 == 0:  # ทุก ๆ 100 step
                random_image = random.choice(self.image_path[2])
                self.image_path[2].remove(random_image)
                if len(self.image_path[3]) == 5:
                    # ถ้า deque ตัวที่ 4 เต็ม ลบไฟล์จาก path เก่าก่อน
                    old_image = self.image_path[3].popleft()
                    if os.path.exists(old_image):
                        os.remove(old_image)
                        
                self.image_path[3].append(random_image)
        
        cv2.imwrite(path, cropped_object)

def check_track_image(own_track):
    result = list(filter(lambda track: track.own_track == own_track, track_uniques_image))
    
    if len(result) == 0:
        return None
    
    return result[0]
    
def addTrackImage(own_track:str, step:int, i:int, image):
    track_image = check_track_image(own_track)

    if track_image is None:
        track = TrackImage(own_track)
        track.add_image(own_track, step, i, image)
        
        track_uniques_image.append(track)
    else:
        track_image.add_image(own_track, step, i, image)

def update_pound(new_track, step)->Tracking:
    track_in_range_of_newTrack = sorted([track for track in tracks_in_pound if track.get_distance(new_track) <= new_track.radius], key=lambda track: track.get_distance(new_track))
    
    # Discovery
    if len(track_in_range_of_newTrack) == 0 or step == 0:
        new_track.own_track = uuid.uuid4()
        tracks_in_pound.append(new_track)
        return new_track
    
    track_similar = set()
    
    track_similar.add(track_in_range_of_newTrack[0])
    
    min_distance = float("inf")
    minRangePredictNextSteptrackInRange = None
    for track in track_in_range_of_newTrack:
        next_pos_x, next_pos_y = track.predict_next_step()
        
        distance = math.sqrt((next_pos_x - new_track.position[0])**2 + (next_pos_y - new_track.position[1])**2)
        if distance < min_distance:
            min_distance = distance 
            minRangePredictNextSteptrackInRange = track
    
    track_similar.add(minRangePredictNextSteptrackInRange)
    
    # know track but unsure
    if len(track_similar) > 1:
        new_track.own_track = uuid.uuid4()
        
        new_track.unsure_track.extend(list([str(track.own_track) for track in track_similar]))
        tracks_in_pound.append(new_track)
        return new_track
    
    # same track
    for track in track_similar:
        track.update_status(new_track)
        return track

def update_time_out(tracks_in_pool):
    for track in tracks_in_pool:
        track.time_out -= 1
        if track.time_out == 0:
            tracks_in_pool.remove(track)
    
    print(f"Track pools size: {len(tracks_in_pool)}")

def close():
    with open('Storage/TrackInformation/output.json', 'a') as json_file:
            data = {
                'step': -1,
                'track_no': f"",
                'own_track': f"",
                'unsure_tracks': [],
                'position': (-1, -1),
                'radius': -1.0
            }
            
            json_file.write("\n")
            json.dump(data, json_file)
            json_file.write("\n]")
            
    with open('Storage/TrackInformation/output2.json', 'a') as json_file2:
        json_file2.write("[\n")
        for track in track_uniques_image:
            data = track.save()
            
            json.dump(data, json_file2)
            json_file2.write(",\n")
        
        json.dump(data, json_file2)
        json_file2.write("\n]")
        
    # ปล่อยทรัพยากรวิดีโอ
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        with open('Storage/TrackInformation/output.json', 'a') as json_file:
            json_file.write("[\n") # start json
            
            step = 0
            
            while cap.isOpened():
                print(f"step: {step}")
                ret, frame = cap.read()
                image = frame.copy()
                if not ret:
                    print("video closed")
                    close()
                    break

                # เรียกใช้งานโมเดล YOLO เพื่อตรวจจับวัตถุ
                results = model.predict(source=frame, imgsz=2048, conf=0.5)
                update_time_out(tracks_in_pound)
                for result in results:
                    boxes = result.boxes
                    
                    for i, box in enumerate(boxes):
                        xyxy_box = box.xyxy[0].to(device)
                        x1, y1, x2, y2 = xyxy_box
                        cropped_object = image[int(y1):int(y2), int(x1):int(x2)]
                        
                        new_track = Tracking(step, xyxy_box, no=f"{step}_{i}")
                        
                        track = update_pound(new_track, step)
                        
                        track.update_position(xyxy_box)
                        
                        # row csv writing 
                        data = {
                            'step': step,
                            'track_no': f"{track.no}",
                            'own_track': f"{track.own_track}",
                            'unsure_tracks': track.unsure_track,
                            'position': (float(track.position[0]), float(track.position[1])),
                            'radius': float(track.radius)
                        }
                        
                        data2 = {
                            'own_track': f"{track.own_track}",
                            'image_file': f'{track.own_track}_{step}_{i}.csv'
                        }
                        
                        json_file.write("\n")
                        json.dump(data, json_file)
                        
                        json_file.write(",")
                            
                        # cv2.imwrite(f"Storage/ImageStorage/{track.own_track}_{step}_{i}.jpg", cropped_object)
                        addTrackImage(track.own_track, step, i, cropped_object)

                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), track.color, 2)
                        cv2.putText(frame, f"{track.no}", (int(x1), int(y1) - 10),  # ตำแหน่งข้อความที่ x1, y1-10
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, track.color, 2, cv2.LINE_AA)
                
                cv2.imwrite(f"image_label/{step}.jpg", frame)
                cv2.imwrite(f"image_test/{step}.jpg", image)
                # แสดงภาพที่มี bounding box
                cv2.imshow("YOLOv8n-obb Detection", frame)
                step += 1
                print(f"ImageStorageSize: {len(os.listdir('Storage/ImageStorage'))}")
                print(f"track_uniques_imageSize: {len(track_uniques_image)}")
                # กด 'q' เพื่อหยุดการทำงาน
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            json_file.write("]\n")
            
        close()
    
    except KeyboardInterrupt:
        close()