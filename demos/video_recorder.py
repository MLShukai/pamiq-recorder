#!/usr/bin/env python
import numpy as np

from pamiq_recorder import VideoRecorder

# Create a video recorder for RGB video
recorder = VideoRecorder(
    file_path="output.mp4",
    fps=30.0,
    height=480,
    width=640,
    channels=3,  # RGB format
)

# Create a sample frame (RGB gradient)
frame = np.zeros((480, 640, 3), dtype=np.uint8)
for i in range(480):
    for j in range(640):
        frame[i, j, 0] = i * 255 // 480  # Red gradient
        frame[i, j, 1] = j * 255 // 640  # Green gradient
        frame[i, j, 2] = (i + j) * 255 // 1120  # Blue gradient

# Write the frame to the video
recorder.write(frame)

# Close when done
recorder.close()
