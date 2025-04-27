# 🎬 pamiq-recorder

[![Formatter & Linter / Tests / Type Check](https://github.com/MLShukai/pamiq-recorder/actions/workflows/main.yml/badge.svg)](https://github.com/MLShukai/pamiq-recorder/actions/workflows/main.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Document Style](https://img.shields.io/badge/%20docstyle-google-3666d6.svg)](https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings)

**pamiq-recorder** is a simple, modern, and type-safe recording library for P-AMI\<Q>, providing easy and consistent interfaces for video, audio, and CSV data recording.

## ✨ Features

- 📹 Video recording via OpenCV with support for multiple formats
- 🎵 Audio recording via SoundFile with various audio formats
- 📊 CSV recording with automatic timestamping
- 📦 JSONL (JSON Lines): Line-delimited JSON format for streaming data records
- 🔗 Seamless integration with [**PAMIQ-Core**](https://github.com/MLShukai/pamiq-core) framework using `to_wrapper` factory method.
- 🔍 Type-safe interfaces with complete type annotations
- 🧪 Comprehensive test coverage

## 📦 Installation

```bash
# Install with pip
pip install pamiq-recorder

# For development setup
git clone https://github.com/MLShukai/pamiq-recorder.git
cd pamiq-recorder
make venv  # Sets up virtual environment with all dependencies
```

## 🧰 Requirements

- Python 3.12+
- OpenCV (for video recording, `libopencv-dev` for Ubuntu.)
- SoundFile (for audio recording, `libsndfile1` for Ubuntu.)

## 📝 Usage

### PAMIQ-Core Integration

All recorders provide seamless integration with the PAMIQ-Core framework through wrapper functionality, allowing you to easily record data from sensors and observations:

```python
from pamiq_recorder import VideoRecorder
from pamiq_core.interaction.wrappers import SensorWrapper

# Create a camera sensor
camera_sensor = YourCameraSensor()  # Your implementation of a sensor class

# Wrap the camera sensor to record all observations
wrapped_sensor = SensorWrapper(
    camera_sensor,
    VideoRecorder.to_wrapper(
        save_dir="./camera_recordings",
        fps=30.0,
        height=480,
        width=640,
        channels=3
    )
)
```

Recording wrappers automatically:

- Create new recordings with timestamped filenames
- Close recordings when the system pauses
- Create fresh recordings when the system resumes
- Clean up resources properly during teardown

### Video Recording

```python
from pamiq_recorder import VideoRecorder
import numpy as np

# Create a video recorder for RGB video
recorder = VideoRecorder(
    file_path="output.mp4",
    fps=30.0,
    height=480,
    width=640,
    channels=3  # RGB format
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
```

### Audio Recording

```python
from pamiq_recorder import AudioRecorder
import numpy as np

# Create an audio recorder for stereo audio
recorder = AudioRecorder(
    file_path="output.wav",
    sample_rate=44100,
    channels=2
)

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
```

### CSV Recording

```python
from pamiq_recorder import CsvRecorder
import time

# Create a CSV recorder with custom headers
recorder = CsvRecorder(
    file_path="sensor_data.csv",
    headers=["temperature", "humidity", "pressure"],
    timestamp_header="time"
)

# Write some sample data rows
recorder.write([25.4, 60.2, 1013.25])
time.sleep(1)
recorder.write([25.5, 60.0, 1013.20])
time.sleep(1)
recorder.write([25.6, 59.8, 1013.15])

# Close when done
recorder.close()
```

### JSONL Recording

```python
from pamiq_recorder import JsonlRecorder
import time

# Create a JSONL recorder
recorder = JsonlRecorder(
    file_path="sensor_data.jsonl",
    ensure_ascii=False  # Optional: Allow direct Unicode characters
)

# Write various data types
recorder.write("Simple string")
recorder.write(42)
recorder.write({"temperature": 25.4, "humidity": 60.2, "pressure": 1013.25})

# Complex nested structures
recorder.write({
    "device_info": {
        "id": "sensor-12345",
        "location": "room-1",
        "firmware": "v2.1.3"
    },
    "readings": [
        {"type": "temperature", "value": 25.4},
        {"type": "humidity", "value": 60.2}
    ],
    "status": "active"
})

# Close when done
recorder.close()
```

The JsonlRecorder stores data in the JSON Lines format (also known as JSONL or Line-delimited JSON), where each line is a valid, independent JSON object.

Each line is saved in the format `{"timestamp": 1619855734.123, "data": ...}` where the timestamp is automatically added and the data can be any JSON-serializable value.

### Using Context Managers

All recorders support the context manager protocol for automatic resource cleanup:

```python
import numpy as np
from pamiq_recorder import AudioRecorder

# Audio data to write
data = np.random.rand(44100, 2).astype(np.float32)  # 1 second of random stereo audio

# Use with statement for automatic closing
with AudioRecorder("output.wav", sample_rate=44100, channels=2) as recorder:
    recorder.write(data)
    # No need to call close() - it happens automatically
```

## 🔧 Supported Formats

### Video Formats

- MP4 (mp4v codec)
- AVI (XVID codec)
- MOV (mp4v codec)
- MKV (X264 codec)

### Audio Formats

- WAV
- FLAC
- OGG/Vorbis
- OGG/Opus
- MP3
- M4A/ALAC
- MOV/ALAC

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Run all workflow (`make run`)
4. Commit your changes (`git commit -m 'Add some amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request
