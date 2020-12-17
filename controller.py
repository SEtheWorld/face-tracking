class Controller:
    """
    Handle trigger events for re-detecting and sending result to server
    """

    def __init__(self, reset_limit_frame, interval):
        self.reset_limit_frame = reset_limit_frame
        self.count_frame = 1
        self.interval = interval

    def trigger_redetect(self):
        return self.count_frame >= self.reset_limit_frame

    def reset(self):
        self.count_frame = 1

    def trigger_extract(self):
        return self.count_frame % self.interval == 0

    def increase_frame(self):
        self.count_frame += 1
