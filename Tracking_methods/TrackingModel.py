import math
from collections import deque
from PredictorModel.PositionPredictor import KalmanFilter
import random

class Tracking:
    def __init__(self, time_serial, xyxy, no=""):
        self.no = no
        self.time_serial = time_serial
        self.own_track = None
        self.unsure_track = []
        
        self.time_out = 5
        
        position = ((xyxy[0] + xyxy[2])//2, (xyxy[1] + xyxy[3])//2)
        self.position = position
        
        range_x = xyxy[2] - xyxy[0]
        range_y = xyxy[3] - xyxy[1]

        if range_x > range_y:
            self.radius = range_x
        else:
            self.radius = range_y
        
        self.PositionPredictor = KalmanFilter(dt=0.01, process_noise=1e-3, measurement_noise=0.1, initial_position=self.position)
        self.history_movement = deque(maxlen=100)
        self.predict_movement = deque(maxlen=100)
        
        self.color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))

    def get_distance(self, other_track):
        x_0, y_0 = self.position 
        x_1, y_1 = other_track.position
        
        return math.sqrt((x_1 - x_0)**2 + (y_1 - y_0)**2)
    
    def predict_next_step(self):
        return self.PositionPredictor.predict()
    
    def update_position(self, xyxy):
        return self.PositionPredictor.update(((xyxy[0] + xyxy[2])//2, (xyxy[1] + xyxy[3])//2))
        
    def update_status(self, new_track):
        self.time_out = 10
        self.radius = new_track.radius
        self.position = new_track.position
    
            
    