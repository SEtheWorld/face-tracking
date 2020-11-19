import time

class Controller:
    """
    Handle trigger events for re-detecting and sending result to server
    """
    def __init__(self, reset_limit_frame, num_frame_capture):
        self.reset_limit_frame = reset_limit_frame
        self.count_frame = 0
        self.interval = int(reset_limit_frame // num_frame_capture)

    def trigger_redetect(self):
        return self.count_frame >= self.reset_limit_frame

    def reset(self):
        self.count_frame = 0

    def increase_frame_and_trigger_send(self):
        self.count_frame += 1
        if self.count_frame % self.interval == 0:
            return True
        return False

    def increase_frame(self):
        self.count_frame += 1