import base64
import gc
import logging
import subprocess
from io import BytesIO

import cv2
import ffmpeg
import requests
import yt_dlp as youtube_dl
from IPython import display as ipythondisplay
from IPython.display import HTML
from IPython.display import Image as ipythonimage
from matplotlib.axes import Axes
from PIL import Image

from fastai.core import *
from fastai.vision import *

from .filters import ColorizerFilter, IFilter, MasterFilter
from .generators import gen_inference_deep, gen_inference_wide


# adapted from https://www.pyimagesearch.com/2016/04/25/watermarking-images-with-opencv-and-python/
def get_watermarked(pil_image: Image) -> Image:
    try:
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        (h, w) = image.shape[:2]
        image = np.dstack([image, np.ones((h, w), dtype="uint8") * 255])
        pct = 0.05
        full_watermark = cv2.imread(
            "./resource_images/watermark.png", cv2.IMREAD_UNCHANGED
        )
        (fwH, fwW) = full_watermark.shape[:2]
        wH = int(pct * h)
        wW = int((pct * h / fwH) * fwW)
        watermark = cv2.resize(full_watermark, (wH, wW), interpolation=cv2.INTER_AREA)
        overlay = np.zeros((h, w, 4), dtype="uint8")
        (wH, wW) = watermark.shape[:2]
        overlay[h - wH - 10 : h - 10, 10 : 10 + wW] = watermark
        # blend the two images together using transparent overlays
        output = image.copy()
        cv2.addWeighted(overlay, 0.5, output, 1.0, 0, output)
        rgb_image = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        final_image = Image.fromarray(rgb_image)
        return final_image
    except:
        # Don't want this to crash everything, so let's just not watermark the image for now.
        return pil_image


class ModelImageVisualizer:
    def __init__(self, filter: IFilter, results_dir: str = None):
        self.filter = filter
        self.results_dir = None if results_dir is None else Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _clean_mem(self):
        torch.cuda.empty_cache()
        # gc.collect()

    def _open_pil_image(self, path: Path) -> Image:
        return PIL.Image.open(path).convert("RGB")

    def _get_image_from_url(self, url: str) -> Image:
        response = requests.get(
            url,
            timeout=30,
            headers={
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36"
            },
        )
        img = PIL.Image.open(BytesIO(response.content)).convert("RGB")
        return img

    def plot_transformed_image_from_url(
        self,
        url: str,
        path: str = "test_images/image.png",
        results_dir: Path = None,
        figsize: Tuple[int, int] = (20, 20),
        render_factor: int = None,
        display_render_factor: bool = False,
        compare: bool = False,
        post_process: bool = True,
        watermarked: bool = True,
    ) -> Path:
        img = self._get_image_from_url(url)
        img.save(path)
        return self.plot_transformed_image(
            path=path,
            results_dir=results_dir,
            figsize=figsize,
            render_factor=render_factor,
            display_render_factor=display_render_factor,
            compare=compare,
            post_process=post_process,
            watermarked=watermarked,
        )

    def plot_transformed_image(
        self,
        path: str,
        results_dir: Path = None,
        figsize: Tuple[int, int] = (20, 20),
        render_factor: int = None,
        display_render_factor: bool = False,
        compare: bool = False,
        post_process: bool = True,
        watermarked: bool = True,
    ) -> Path:
        path = Path(path)
        if results_dir is None:
            results_dir = Path(self.results_dir)
        result = self.get_transformed_image(
            path, render_factor, post_process=post_process, watermarked=watermarked
        )
        orig = self._open_pil_image(path)
        if compare:
            self._plot_comparison(
                figsize, render_factor, display_render_factor, orig, result
            )
        else:
            self._plot_solo(figsize, render_factor, display_render_factor, result)

        orig.close()
        result_path = self._save_result_image(path, result, results_dir=results_dir)
        result.close()
        return result_path

    def _plot_comparison(
        self,
        figsize: Tuple[int, int],
        render_factor: int,
        display_render_factor: bool,
        orig: Image,
        result: Image,
    ):
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        self._plot_image(
            orig,
            axes=axes[0],
            figsize=figsize,
            render_factor=render_factor,
            display_render_factor=False,
        )
        self._plot_image(
            result,
            axes=axes[1],
            figsize=figsize,
            render_factor=render_factor,
            display_render_factor=display_render_factor,
        )

    def _plot_solo(
        self,
        figsize: Tuple[int, int],
        render_factor: int,
        display_render_factor: bool,
        result: Image,
    ):
        fig, axes = plt.subplots(1, 1, figsize=figsize)
        self._plot_image(
            result,
            axes=axes,
            figsize=figsize,
            render_factor=render_factor,
            display_render_factor=display_render_factor,
        )

    def _save_result_image(
        self, source_path: Path, image: Image, results_dir=None
    ) -> Path:
        if results_dir is None:
            results_dir = Path(self.results_dir)
        result_path = results_dir / source_path.name
        image.save(result_path)
        return result_path

    def get_transformed_image(
        self,
        path: Path,
        render_factor: int = None,
        post_process: bool = True,
        watermarked: bool = True,
    ) -> Image:
        self._clean_mem()
        orig_image = self._open_pil_image(path)
        filtered_image = self.filter.filter(
            orig_image,
            orig_image,
            render_factor=render_factor,
            post_process=post_process,
        )

        if watermarked:
            return get_watermarked(filtered_image)

        return filtered_image

    def _plot_image(
        self,
        image: Image,
        render_factor: int,
        axes: Axes = None,
        figsize=(20, 20),
        display_render_factor=False,
    ):
        if axes is None:
            _, axes = plt.subplots(figsize=figsize)
        axes.imshow(np.asarray(image) / 255)
        axes.axis("off")
        if render_factor is not None and display_render_factor:
            plt.text(
                10,
                10,
                "render_factor: " + str(render_factor),
                color="white",
                backgroundcolor="black",
            )

    def _get_num_rows_columns(
        self, num_images: int, max_columns: int
    ) -> Tuple[int, int]:
        columns = min(num_images, max_columns)
        rows = num_images // columns
        rows = rows if rows * columns == num_images else rows + 1
        return rows, columns


class VideoColorizer:
    def __init__(self, vis: ModelImageVisualizer):
        self.vis = vis
        workfolder = Path("./video")
        self.source_folder = workfolder / "source"
        self.bwframes_root = workfolder / "bwframes"
        self.audio_root = workfolder / "audio"
        self.colorframes_root = workfolder / "colorframes"
        self.result_folder = workfolder / "result"

    def _purge_images(self, dir):
        for f in os.listdir(dir):
            if re.search(".*?\.jpg", f):
                os.remove(os.path.join(dir, f))

    def _get_ffmpeg_probe(self, path: Path):
        try:
            probe = ffmpeg.probe(str(path))
            return probe
        except ffmpeg.Error as e:
            logging.error("ffmpeg error: {0}".format(e), exc_info=True)
            logging.error("stdout:" + e.stdout.decode("UTF-8"))
            logging.error("stderr:" + e.stderr.decode("UTF-8"))
            raise e
        except Exception as e:
            logging.error(
                "Failed to instantiate ffmpeg.probe.  Details: {0}".format(e),
                exc_info=True,
            )
            raise e

    def _get_fps(self, source_path: Path) -> str:
        probe = self._get_ffmpeg_probe(source_path)
        stream_data = next(
            (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
            None,
        )
        return stream_data["avg_frame_rate"]

    def _download_video_from_url(self, source_url, source_path: Path):
        if source_path.exists():
            source_path.unlink()

        ydl_opts = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
            "outtmpl": str(source_path),
            "retries": 30,
            "fragment-retries": 30,
        }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([source_url])

    def _extract_raw_frames(self, source_path: Path):
        bwframes_folder = self.bwframes_root / (source_path.stem)
        bwframe_path_template = str(bwframes_folder / "%5d.jpg")
        bwframes_folder.mkdir(parents=True, exist_ok=True)
        self._purge_images(bwframes_folder)

        process = (
            ffmpeg.input(str(source_path))
            .output(
                str(bwframe_path_template),
                format="image2",
                vcodec="mjpeg",
                **{"q:v": "0"},
            )
            .global_args("-hide_banner")
            .global_args("-nostats")
            .global_args("-loglevel", "error")
        )

        try:
            process.run()
        except ffmpeg.Error as e:
            logging.error("ffmpeg error: {0}".format(e), exc_info=True)
            logging.error("stdout:" + e.stdout.decode("UTF-8"))
            logging.error("stderr:" + e.stderr.decode("UTF-8"))
            raise e
        except Exception as e:
            logging.error(
                "Errror while extracting raw frames from source video.  Details: {0}".format(
                    e
                ),
                exc_info=True,
            )
            raise e

    def _extract_audio(self, source_path: Path) -> Path:
        audio_path = self.result_folder / (source_path.stem + ".wav")
        command = [
            "ffmpeg",
            "-i",
            str(source_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "44100",
            "-ac",
            "2",
            str(audio_path),
        ]
        subprocess.run(command, check=True)
        return audio_path

    def _colorize_raw_frames(
        self,
        source_path: Path,
        render_factor: int = None,
        post_process: bool = True,
        watermarked: bool = True,
    ):
        colorframes_folder = self.colorframes_root / (source_path.stem)
        colorframes_folder.mkdir(parents=True, exist_ok=True)
        self._purge_images(colorframes_folder)
        bwframes_folder = self.bwframes_root / (source_path.stem)

        for img in progress_bar(os.listdir(str(bwframes_folder))):
            img_path = bwframes_folder / img

            if os.path.isfile(str(img_path)):
                color_image = self.vis.get_transformed_image(
                    str(img_path),
                    render_factor=render_factor,
                    post_process=post_process,
                    watermarked=watermarked,
                )
                color_image.save(str(colorframes_folder / img))

    def _build_video(self, source_path: Path) -> Path:
        colorized_path = self.result_folder / (
            source_path.name.replace(".mp4", "_no_audio.mp4")
        )
        print("COLORIZED PATH :: ", colorized_path)
        colorframes_folder = self.colorframes_root / (source_path.stem)
        colorframes_path_template = str(colorframes_folder / "%5d.jpg")
        colorized_path.parent.mkdir(parents=True, exist_ok=True)
        if colorized_path.exists():
            colorized_path.unlink()
        fps = self._get_fps(source_path)

        process = (
            ffmpeg.input(
                str(colorframes_path_template),
                format="image2",
                vcodec="mjpeg",
                framerate=fps,
            )
            .output(str(colorized_path), crf=17, vcodec="libx264")
            .global_args("-hide_banner")
            .global_args("-nostats")
            .global_args("-loglevel", "error")
        )

        try:
            process.run()
        except ffmpeg.Error as e:
            logging.error("ffmpeg error: {0}".format(e), exc_info=True)
            logging.error("stdout:" + e.stdout.decode("UTF-8"))
            logging.error("stderr:" + e.stderr.decode("UTF-8"))
            raise e
        except Exception as e:
            logging.error(
                "Errror while building output video.  Details: {0}".format(e),
                exc_info=True,
            )
            raise e

        result_path = self.result_folder / source_path.name
        if result_path.exists():
            result_path.unlink()
        # making copy of non-audio version in case adding back audio doesn't apply or fails.
        shutil.copyfile(str(colorized_path), str(result_path))

        # adding back sound here
        audio_file = Path(str(source_path).replace(".mp4", ".aac"))
        # if audio_file.exists():
        #     audio_file.unlink()
        aac_audio_path = audio_file.name
        mp3_file_path = str(source_path).replace(".acc", ".mp3")
        print("MP3 FILE PATH :: ", mp3_file_path)
        mp3_audio_path = self.convert_aac_to_mp3(aac_audio_path, mp3_file_path)
        print("CONVERSION SUCCESSFUL :: ", mp3_audio_path)
        self.translate(mp3_audio_path)
        self.add_audio(source_path, mp3_audio_path, result_path)
        # os.system(
        #     'ffmpeg -y -i "'
        #     + str(source_path)
        #     + '" -vn -acodec copy "'
        #     + str(audio_file)
        #     + '"'
        #     + " -hide_banner"
        #     + " -nostats"
        #     + " -loglevel error"
        # )

        # if audio_file.exists():
        #     os.system(
        #         'ffmpeg -y -i "'
        #         + str(colorized_path)
        #         + '" -i "'
        #         + str(audio_file)
        #         + '" -shortest -c:v copy -c:a aac -b:a 256k "'
        #         + str(result_path)
        #         + '"'
        #         + " -hide_banner"
        #         + " -nostats"
        #         + " -loglevel error"
        #     )
        # logging.info("Video created here: " + str(result_path))

        return result_path

    def get_duration(self, filename):
        probe = ffmpeg.probe(filename)
        return float(probe["format"]["duration"])

    def get_frame_rate(self, filename):
        r_frame_rate = ffmpeg.probe(filename)["streams"][0]["r_frame_rate"]
        num, den = map(int, r_frame_rate.split("/"))
        return num / den

    def add_audio(self, video_path, audio_path, output_path):
        video_duration = self.get_duration(video_path)
        audio_duration = self.get_duration(audio_path)

        input_video = ffmpeg.input(video_path)
        input_audio = ffmpeg.input(audio_path)

        # If video is longer, trim video
        if video_duration > audio_duration:
            num_frames = int(audio_duration * self.get_frame_rate(video_path))
            input_video = input_video.trim(start_frame=0, end_frame=num_frames)

        # If audio is longer, trim audio
        if audio_duration > video_duration:
            input_audio = input_audio.filter_("atrim", start=0, end=video_duration)

        ffmpeg.concat(input_video, input_audio, v=1, a=1).output(output_path).run()

    def convert_aac_to_mp3(self, input_file, output_file):
        stream = ffmpeg.input(input_file)
        stream = ffmpeg.output(
            stream, output_file, format="mp3", acodec="libmp3lame", ab="192k"
        )
        ffmpeg.run(stream)
        return output_file

    def translate(self, input_file):
        import os
        import time

        import googletrans
        import speech_recognition
        import speech_recognition as sr
        from gtts import gTTS
        from pydub import AudioSegment

        ip_lang = "en"
        op_lang = "hi"

        recognizer = speech_recognition.Recognizer()

        audio = AudioSegment.from_mp3(input_file)
        audio.export("temp.wav", format="wav")

        # Initialize the recognizer
        recognizer = sr.Recognizer()

        # Load the WAV file
        with sr.AudioFile("temp.wav") as source:
            # Adjust for ambient noise, if necessary
            recognizer.adjust_for_ambient_noise(source)

            # Listen to the audio file
            audio = recognizer.record(source)

            # Use Google Web Speech API to transcribe the audio
            try:
                text = recognizer.recognize_google(audio, language="en-US")
                print("Transcription: ", text)
            except sr.UnknownValueError:
                print("Google Web Speech API could not understand the audio.")
            except sr.RequestError as e:
                print(f"Could not request results from Google Web Speech API; {e}")

        # Cleanup: Delete temporary WAV file
        os.remove("temp.wav")

        transator = googletrans.Translator()
        translation = transator.translate(text, dest=op_lang)
        conv_audio = gTTS(translation.text, lang=op_lang)
        conv_audio.save(input_file)

    def colorize_from_url(
        self,
        source_url,
        file_name: str,
        render_factor: int = None,
        post_process: bool = True,
        watermarked: bool = True,
    ) -> Path:
        source_path = self.source_folder / file_name
        self._download_video_from_url(source_url, source_path)
        return self._colorize_from_path(
            source_path,
            render_factor=render_factor,
            post_process=post_process,
            watermarked=watermarked,
        )

    def colorize_from_file_name(
        self,
        file_name: str,
        render_factor: int = None,
        watermarked: bool = True,
        post_process: bool = True,
    ) -> Path:
        source_path = self.source_folder / file_name
        return self._colorize_from_path(
            source_path,
            render_factor=render_factor,
            post_process=post_process,
            watermarked=watermarked,
        )

    def _colorize_from_path(
        self,
        source_path: Path,
        render_factor: int = None,
        watermarked: bool = True,
        post_process: bool = True,
    ) -> Path:
        if not source_path.exists():
            raise Exception(
                "Video at path specfied, " + str(source_path) + " could not be found."
            )
        self._extract_raw_frames(source_path)
        self._colorize_raw_frames(
            source_path,
            render_factor=render_factor,
            post_process=post_process,
            watermarked=watermarked,
        )
        return self._build_video(source_path)


def get_video_colorizer(render_factor: int = 21) -> VideoColorizer:
    return get_stable_video_colorizer(render_factor=render_factor)


def get_artistic_video_colorizer(
    root_folder: Path = Path("./"),
    weights_name: str = "ColorizeArtistic_gen",
    results_dir="result_images",
    render_factor: int = 35,
) -> VideoColorizer:
    learn = gen_inference_deep(root_folder=root_folder, weights_name=weights_name)
    filtr = MasterFilter([ColorizerFilter(learn=learn)], render_factor=render_factor)
    vis = ModelImageVisualizer(filtr, results_dir=results_dir)
    return VideoColorizer(vis)


def get_stable_video_colorizer(
    root_folder: Path = Path("./"),
    weights_name: str = "ColorizeVideo_gen",
    results_dir="result_images",
    render_factor: int = 21,
) -> VideoColorizer:
    learn = gen_inference_wide(root_folder=root_folder, weights_name=weights_name)
    filtr = MasterFilter([ColorizerFilter(learn=learn)], render_factor=render_factor)
    vis = ModelImageVisualizer(filtr, results_dir=results_dir)
    return VideoColorizer(vis)


def get_image_colorizer(
    root_folder: Path = Path("./"), render_factor: int = 35, artistic: bool = True
) -> ModelImageVisualizer:
    if artistic:
        return get_artistic_image_colorizer(
            root_folder=root_folder, render_factor=render_factor
        )
    else:
        return get_stable_image_colorizer(
            root_folder=root_folder, render_factor=render_factor
        )


def get_stable_image_colorizer(
    root_folder: Path = Path("./"),
    weights_name: str = "ColorizeStable_gen",
    results_dir="result_images",
    render_factor: int = 35,
) -> ModelImageVisualizer:
    learn = gen_inference_wide(root_folder=root_folder, weights_name=weights_name)
    filtr = MasterFilter([ColorizerFilter(learn=learn)], render_factor=render_factor)
    vis = ModelImageVisualizer(filtr, results_dir=results_dir)
    return vis


def get_artistic_image_colorizer(
    root_folder: Path = Path("./"),
    weights_name: str = "ColorizeArtistic_gen",
    results_dir="result_images",
    render_factor: int = 35,
) -> ModelImageVisualizer:
    learn = gen_inference_deep(root_folder=root_folder, weights_name=weights_name)
    filtr = MasterFilter([ColorizerFilter(learn=learn)], render_factor=render_factor)
    vis = ModelImageVisualizer(filtr, results_dir=results_dir)
    return vis


def show_image_in_notebook(image_path: Path):
    ipythondisplay.display(ipythonimage(str(image_path)))


def show_video_in_notebook(video_path: Path):
    video = io.open(video_path, "r+b").read()
    encoded = base64.b64encode(video)
    ipythondisplay.display(
        HTML(
            data="""<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>""".format(
                encoded.decode("ascii")
            )
        )
    )
