#!/usr/bin/env python
import time

from pamiq_recorder import CsvRecorder

# Create a CSV recorder with custom headers
recorder = CsvRecorder(
    file_path="sensor_data.csv",
    headers=["temperature", "humidity", "pressure"],
    timestamp_header="time",
)

# Write some sample data rows
recorder.write([25.4, 60.2, 1013.25])
time.sleep(1)
recorder.write([25.5, 60.0, 1013.20])
time.sleep(1)
recorder.write([25.6, 59.8, 1013.15])

# Close when done
recorder.close()
