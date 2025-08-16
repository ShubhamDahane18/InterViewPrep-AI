
import streamlit as st
import cv2
import torch
import numpy as np
import tempfile
import os
from edge_tts import Communicate
import asyncio
import threading
import time

# Streamlit UI Setup
st.set_page_config(page_title="Real-Time Object Detection with Voice", layout="centered")
st.title("YOLO Person Detection with Edge TTS")

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Async TTS function
async def speak(text):
    communicate = Communicate(text, voice="en-IN-NeerjaNeural")
    await communicate.save("output.mp3")
    os.system("start output.mp3" if os.name == "nt" else "afplay output.mp3")

# Run TTS in a separate thread
def tts_thread(text):
    asyncio.run(speak(text))

# Start and Stop buttons
start = st.button("Start Camera")
stop = st.button("Stop Camera")

# Track camera status
if "camera_running" not in st.session_state:
    st.session_state.camera_running = False

if start:
    st.session_state.camera_running = True
if stop:
    st.session_state.camera_running = False

# Run detection if camera is on
if st.session_state.camera_running:
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()

    person_present = True  # Track if person was present in previous frame
    last_prompt_time = 0  # Track when last "step into view" was said
    
    try:
        while st.session_state.camera_running:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to grab frame")
                break

            # YOLOv5 inference
            results = model(frame)
            df = results.pandas().xyxy[0]
            persons = df[df['name'] == 'person']

            current_time = time.time()
            
            # Handle TTS prompts when no person is detected
            if len(persons) == 0:
                if person_present:  # First time person disappears
                    threading.Thread(target=tts_thread, args=("Please step into view",), daemon=True).start()
                    person_present = False
                    last_prompt_time = current_time
                elif current_time - last_prompt_time >= 3.0:  # Repeat every 3 seconds if still no person
                    threading.Thread(target=tts_thread, args=("Please step into view",), daemon=True).start()
                    last_prompt_time = current_time
            else:
                person_present = True
                last_prompt_time = 0  # Reset timer when person is detected

            # Draw bounding boxes (without text labels)
            for index, row in persons.iterrows():
                x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Removed the cv2.putText line to eliminate text labels

            # Display in Streamlit
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame, channels="RGB")

    except Exception as e:
        st.error(f"Error: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        st.session_state.camera_running = False


# import streamlit as st
# import cv2
# import torch
# import numpy as np
# import asyncio
# import threading
# import tempfile
# import os
# from edge_tts import Communicate
# from pydub import AudioSegment
# from pydub.playback import play
# from io import BytesIO

# # Streamlit UI Setup
# st.set_page_config(page_title="Real-Time Object Detection with Voice", layout="centered")
# st.title("YOLO Person Detection with Edge TTS (No File Save)")

# # Load YOLO model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# # Async TTS with in-memory playback using pydub
# async def speak(text):
#     communicate = Communicate(text, voice="en-IN-NeerjaNeural")
#     stream = BytesIO()
#     async for chunk in communicate.stream():
#         if chunk["type"] == "audio":
#             stream.write(chunk["data"])
#     stream.seek(0)
#     audio = AudioSegment.from_file(stream, format="mp3")
#     play(audio)

# # Run TTS in a separate thread
# def tts_thread(text):
#     asyncio.run(speak(text))

# # Start and Stop buttons
# start = st.button("Start Camera")
# stop = st.button("Stop Camera")

# # Track camera status
# if "camera_running" not in st.session_state:
#     st.session_state.camera_running = False

# if start:
#     st.session_state.camera_running = True
# if stop:
#     st.session_state.camera_running = False

# # Run detection if camera is on
# if st.session_state.camera_running:
#     cap = cv2.VideoCapture(0)
#     frame_placeholder = st.empty()
#     last_state = ""

#     try:
#         while st.session_state.camera_running:
#             ret, frame = cap.read()
#             if not ret:
#                 st.warning("Failed to grab frame")
#                 break

#             # YOLOv5 inference
#             results = model(frame)
#             df = results.pandas().xyxy[0]
#             persons = df[df['name'] == 'person']

#             if len(persons) > 0:
#                 label = "Person detected"
#             else:
#                 label = "Please step into view"

#             # Speak only on state change
#             if label != last_state:
#                 threading.Thread(target=tts_thread, args=(label,), daemon=True).start()
#                 last_state = label

#             # Draw bounding boxes
#             for index, row in persons.iterrows():
#                 x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame, label, (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#             # Display in Streamlit
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frame_placeholder.image(frame, channels="RGB")

#     except Exception as e:
#         st.error(f"Error: {e}")

#     finally:
#         cap.release()
#         cv2.destroyAllWindows()
#         st.session_state.camera_running = False

# import streamlit as st
# import cv2
# import torch
# import numpy as np
# import asyncio
# import threading
# from edge_tts import Communicate
# from pydub import AudioSegment
# import simpleaudio as sa
# from io import BytesIO

# # Streamlit UI Setup
# st.set_page_config(page_title="Real-Time Object Detection with Voice", layout="centered")
# st.title("YOLO Person Detection with Edge TTS (In-Memory Only)")

# # Load YOLO model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

# # Async TTS with in-memory playback using simpleaudio
# async def speak(text):
#     communicate = Communicate(text, voice="en-IN-NeerjaNeural")
#     stream = BytesIO()
#     async for chunk in communicate.stream():
#         if chunk["type"] == "audio":
#             stream.write(chunk["data"])
#     stream.seek(0)
#     audio = AudioSegment.from_file(stream, format="mp3")
#     playback = sa.play_buffer(
#         audio.raw_data,
#         num_channels=audio.channels,
#         bytes_per_sample=audio.sample_width,
#         sample_rate=audio.frame_rate
#     )
#     playback.wait_done()

# # Run TTS in a separate thread
# def tts_thread(text):
#     asyncio.run(speak(text))

# # Start and Stop buttons
# start = st.button("Start Camera")
# stop = st.button("Stop Camera")

# # Track camera status
# if "camera_running" not in st.session_state:
#     st.session_state.camera_running = False

# if start:
#     st.session_state.camera_running = True
# if stop:
#     st.session_state.camera_running = False

# # Run detection if camera is on
# if st.session_state.camera_running:
#     cap = cv2.VideoCapture(0)
#     frame_placeholder = st.empty()
#     last_state = ""

#     try:
#         while st.session_state.camera_running:
#             ret, frame = cap.read()
#             if not ret:
#                 st.warning("Failed to grab frame")
#                 break

#             # YOLOv5 inference
#             results = model(frame)
#             df = results.pandas().xyxy[0]
#             persons = df[df['name'] == 'person']

#             if len(persons) > 0:
#                 label = "Person detected"
#             else:
#                 label = "Please step into view"

#             # Speak only on state change
#             if label != last_state:
#                 threading.Thread(target=tts_thread, args=(label,), daemon=True).start()
#                 last_state = label

#             # Draw bounding boxes
#             for _, row in persons.iterrows():
#                 x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame, label, (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#             # Display in Streamlit
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frame_placeholder.image(frame, channels="RGB")

#     except Exception as e:
#         st.error(f"Error: {e}")

#     finally:
#         cap.release()
#         cv2.destroyAllWindows()
#         st.session_state.camera_running = False
