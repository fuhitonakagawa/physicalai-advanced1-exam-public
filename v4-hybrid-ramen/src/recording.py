"""カメラ観測と試行番号を含む動画。"""

import subprocess
from pathlib import Path
from types import TracebackType

from PIL import Image, ImageDraw, ImageFont

from src.experiment.types import Observation


class VideoEncoder:
    """別プロセスでH.264動画を出力する。"""

    def __init__(self, path: Path, fps: int, executable: Path) -> None:
        """RGBフレームの入力先を用意する。"""
        self.process = subprocess.Popen(
            [
                str(executable),
                "-hide_banner",
                "-loglevel",
                "error",
                "-n",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                "768x448",
                "-r",
                str(fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def append(self, image: Image.Image) -> None:
        """カメラフレームをエンコーダへ渡す。"""
        if self.process.stdin is None:
            raise RuntimeError("動画ストリームがありません")
        self.process.stdin.write(image.tobytes())

    def close(self) -> None:
        """ストリームを閉じ、エンコーダの終了を確認する。"""
        if self.process.stdin is not None:
            self.process.stdin.close()
            self.process.stdin = None
        try:
            self.process.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.communicate()
            raise RuntimeError("動画の出力がタイムアウトしました") from None
        if self.process.returncode != 0:
            raise RuntimeError("動画の出力に失敗しました")


class EpisodeRecorder:
    """成功・失敗によらずカメラ画像を動画へ保存する。"""

    def __init__(
        self, path: Path, fps: int, instruction: str, executable: Path
    ) -> None:
        """1試行分の記録を開始する。"""
        self.path = path
        self.fps = fps
        self.instruction = instruction
        self.encoder = VideoEncoder(path, fps, executable)
        self.font = ImageFont.load_default(size=18)

    def __enter__(self) -> "EpisodeRecorder":
        """記録先を返す。"""
        return self

    def record(
        self, observation: Observation, step: int, phase: str, food: str
    ) -> None:
        """モデルが観測した2台のカメラ画像を記録する。"""
        frame = Image.new("RGB", (768, 448), "#142033")
        for index, camera in enumerate(
            (observation.main_image, observation.wrist_image)
        ):
            frame.paste(Image.fromarray(camera).resize((384, 384)), (index * 384, 64))
        draw = ImageDraw.Draw(frame)
        draw.text(
            (16, 8),
            f"{phase} | {food} | {step / self.fps:.1f} s",
            font=self.font,
            fill="white",
        )
        draw.text(
            (16, 35),
            self.instruction[:88],
            font=ImageFont.load_default(size=12),
            fill="#b9cedc",
        )
        self.encoder.append(frame)
        if step == 0:
            frame.save(self.path.with_suffix(".png"))

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """動画ファイルを確定する。"""
        self.encoder.close()
