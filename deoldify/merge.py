import ffmpeg


def get_duration(filename):
    probe = ffmpeg.probe(filename)
    return float(probe["format"]["duration"])


def get_frame_rate(filename):
    r_frame_rate = ffmpeg.probe(filename)["streams"][0]["r_frame_rate"]
    num, den = map(int, r_frame_rate.split("/"))
    return num / den


def add_audio(video_path, audio_path, output_path):
    video_duration = get_duration(video_path)
    audio_duration = get_duration(audio_path)

    input_video = ffmpeg.input(video_path)
    input_audio = ffmpeg.input(audio_path)

    # If video is longer, trim video
    if video_duration > audio_duration:
        num_frames = int(audio_duration * get_frame_rate(video_path))
        input_video = input_video.trim(start_frame=0, end_frame=num_frames)

    # If audio is longer, trim audio
    if audio_duration > video_duration:
        input_audio = input_audio.filter_("atrim", start=0, end=video_duration)

    ffmpeg.concat(input_video, input_audio, v=1, a=1).output(output_path).run()


def convert_aac_to_mp3(input_file, output_file):
    stream = ffmpeg.input(input_file)
    stream = ffmpeg.output(
        stream, output_file, format="mp3", acodec="libmp3lame", ab="192k"
    )
    ffmpeg.run(stream)
    return output_file


if __name__ == "__main__":
    AUDIO_PATH = "/home/eyepatch/Desktop/DeOldify/video/source/video.aac"
    VIDEO_PATH = "/home/eyepatch/Desktop/DeOldify/video/result/video_no_audio.mp4"
    OUTPUT_PATH = "/home/eyepatch/Desktop/DeOldify/video/result/result_with_audio.mp4"
    # AUDIO_PATH = convert_aac_to_mp3(
    #     AUDIO_PATH, "/home/eyepatch/Desktop/DeOldify/video/source/video.mp3"
    # )
    AUDIO_PATH = "/home/eyepatch/Desktop/DeOldify/video/source/video.mp3"
    add_audio(VIDEO_PATH, AUDIO_PATH, OUTPUT_PATH)
