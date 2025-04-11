#!/usr/bin/env python

import numpy as np

from pamiq_recorder import AudioRecorder

# Create an audio recorder for stereo audio
recorder = AudioRecorder(file_path="output.wav", sample_rate=44100, channels=2)

# Create a simple stereo sine wave
duration = 5.0  # seconds
t = np.linspace(0, duration, int(duration * 44100), endpoint=False)

# Left channel: 440 Hz, Right channel: 880 Hz
left = np.sin(2 * np.pi * 440 * t)
right = np.sin(2 * np.pi * 880 * t)

# Combine into stereo data with shape (samples, channels)
data = np.column_stack((left, right)).astype(np.float32)

# Write the audio data
recorder.write(data)

# Close when done
recorder.close()
