from moviepy.editor import VideoFileClip
import os

def extract_audio(video_path, output_audio_path="temp_audio.wav"):
    try:
        video = VideoFileClip(video_path)

        # 🔴 Check if audio exists
        if video.audio is None:
            print("No audio track found in video.")
            return None

        video.audio.write_audiofile(output_audio_path, codec='pcm_s16le', verbose=False, logger=None)

        # 🔴 Check if file actually created
        if not os.path.exists(output_audio_path) or os.path.getsize(output_audio_path) == 0:
            return None

        return output_audio_path

    except Exception as e:
        print("Audio extraction error:", e)
        return None
