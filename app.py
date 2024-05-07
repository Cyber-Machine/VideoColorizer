import os
from time import sleep

import streamlit as st

# from deoldify import device
# from deoldify.device_id import DeviceId

# device.set(device=DeviceId.GPU0)
# import torch

# from deoldify.visualize import *

# if not torch.cuda.is_available():
#     print("GPU not available.")

RESULT = False
with st.sidebar:
    st.title("Video Uploader")
    render_factor = st.slider("render_factor", min_value=10, max_value=50)

    link = st.text_input("Add Video Link")

    # Box to upload video
    video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    # colorizer = get_video_colorizer()
    # Submit button
    if st.button("Submit"):
        RESULT = True
        if link:
            # try:
            #     video_path = colorizer.colorize_from_url(
            #         link,
            #         "video.mp4",
            #         render_factor=render_factor,
            #         watermarked=False,
            #     )
            #     show_video_in_notebook(video_path)
            # except ffmpeg.Error as e:
            #     print("stderr:", e.stderr)
            pass

        if video_file is not None:
            # Process uploaded video
            # Save the video or do something wit
            pass

if os.path.exists("video.mp4") and RESULT:
    sleep(3)
    video_file = open("video.mp4", "rb")
    video_bytes = video_file.read()

    st.video(video_bytes)
