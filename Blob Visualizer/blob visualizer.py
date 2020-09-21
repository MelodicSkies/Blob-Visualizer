import pyaudio
from scipy.fftpack import fft

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

import pyqtgraph.opengl as gl
import numpy as np

import sys
import time

class FFT():

    def calculate_fft(self, live_data, alpha):
        self.live_data = live_data
        self.data_splice()

        left_fft = abs(fft(self.audio_left))
        right_fft = abs(fft(self.audio_right))

        final_fft = [self.convert_to_dB(self.exponential_smooth(left_fft, alpha)), self.convert_to_dB(self.exponential_smooth(right_fft, alpha))]

        return final_fft

    #splits stereo data into left and right channels
    def data_splice(self):
        self.audio_left = self.live_data[0::2]
        self.audio_right = self.live_data[1::2]

    def convert_to_dB(self, data):
        for item in data:
            if item != 0:
                item = (10 * np.log10(item) ** 2) #accentuates peaks by converting to logarithmic scale

        return data

    #exponential smoothing to help reduce noise, smoothing factor is denoted as alpha (0 - 1)
    def exponential_smooth(self, fft_array, alpha):
        i = 0
        adjusted_fft = []

        while i < len(fft_array):
            if i == 0:
                adjusted_fft.append(abs(((1- alpha) * fft_array[i])))
            else:
                adjusted_fft.append(abs(alpha * adjusted_fft[i - 1] + ((1- alpha) * fft_array[i])))

            i += 1
    
        return adjusted_fft

class Blob_Visualizer(object):
    
    def __init__(self, *args, **kwargs):
        pg.setConfigOptions(antialias=True)
        self.app = QtGui.QApplication(sys.argv)

        self.visualizer_window = gl.GLViewWidget()
        self.visualizer_window.show()

        self.visualizer_window.setCameraPosition( distance = 20, azimuth = -90)
        self.visualizer_window.setGeometry(1000, 1000, 1000, 1000)
         
        self.point_1 = np.array([0, 2, 0])
        self.point_2 = np.array([0, -2, 0])

        self.origin = (self.point_1 + self.point_2) / 2
        self.radius = np.linalg.norm(self.point_2 - self.point_1) / 2

        self.sphere = gl.MeshData.sphere(rows = 25, cols = 25, radius = self.radius)

        self.sphere_visualizer = gl.GLMeshItem(meshdata = self.sphere, smooth = True, color = (10, 10, 10, 0.2), shader = "balloon", glOptions = "additive",)

        self.sphere_visualizer.translate(*self.origin) #ability to pan around origin with mouse

        self.visualizer_window.addItem(self.sphere_visualizer)

        self.CHUNK = 1024 #data chunking size for audio stream, use power of 2 for optimal fft speed, higher = better peak resolution, lower = better time resolution
        self.FS = 44100 #sampling rate, denoted by audio input/hardware (Hz)
        self.ALPHA = 0.3 #smoothing factor

        audio_input = pyaudio.PyAudio()
        self.audio_stream = audio_input.open(format = pyaudio.paFloat32, channels = 2, rate = self.FS, input = True, output = True, frames_per_buffer = self.CHUNK) #initiates audio stream

        self.fft_analysis = FFT()
    
    #updates visualizer in real time based on audio input
    def update(self):
        live_data = np.frombuffer(self.audio_stream.read(self.CHUNK, exception_on_overflow = False), dtype = np.float32)
        fft_data = self.fft_analysis.calculate_fft(live_data, self.ALPHA)
        
        #takes the average of the fft of both channels and calculates a new radius
        point_1 = np.array([0, 2 * (np.average(fft_data[0])), 0])
        point_2 = np.array([0, 2 * (-1 * np.average(fft_data[1])), 0])
        radius = np.linalg.norm(point_2 - point_1) / 2
        new_sphere = gl.MeshData.sphere(rows = 25, cols = 25, radius = self.radius + radius)
        
        #updates sphere with new radius
        self.visualizer_window.removeItem(self.sphere_visualizer)
        self.sphere_visualizer = gl.GLMeshItem(meshdata = new_sphere, smooth = True, color = (10, 10, 10, 0.2), shader = "balloon", glOptions = "additive",)
        self.visualizer_window.addItem(self.sphere_visualizer)

    #keeps visualizer running in loop
    def real_time(self):
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(20)
        self.start()

    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
            QtGui.QApplication.instance().exec_()

if __name__ == "__main__":

    visualizer = Blob_Visualizer()
    visualizer.real_time()