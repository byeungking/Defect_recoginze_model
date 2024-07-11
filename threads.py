from PyQt5.QtCore import QThread
from train import train_model

class TrainingThread(QThread):
    def __init__(self, img_width, img_height, img_dir, epochs, model_dir, model_name, progress_signal, finished_signal):
        super().__init__()
        self.img_width = img_width
        self.img_height = img_height
        self.img_dir = img_dir
        self.epochs = epochs
        self.model_dir = model_dir
        self.model_name = model_name
        self.progress_signal = progress_signal
        self.finished_signal = finished_signal

    def run(self):
        train_model(self.img_width, self.img_height, self.img_dir, self.epochs, self.model_dir, self.model_name, self.progress_signal)
        self.finished_signal.emit()
