import logging
import math
import threading


# monitors progress
class ExeControl:
    def __init__(self):
        self.total_frames = []
        self.progress = []

    def increase(self, id):
        self.progress[id] = self.progress[id] + 1.0

    def end_process(self):
        self.total_frames = []
        self.progress = []

    def set_frames(self, frame_count):
        logging.info(f"Frame counts: {frame_count}")
        self.total_frames = frame_count

    def get_progress(self):
        if len(self.progress)==0:
            return []
        if len(self.total_frames)==0:
            return [0.0 for _ in self.progress]
        return [math.ceil(self.progress[index] / self.total_frames[index]*100) for index in range(len(self.progress))]

    def start_process(self, file_names):
        self.progress = [0.0 for _ in file_names]


job_monitor = ExeControl()
