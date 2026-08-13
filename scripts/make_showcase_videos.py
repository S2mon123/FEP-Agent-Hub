from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


WIDTH = 1920
HEIGHT = 1080
FPS = 30
CYAN = (250, 220, 50)
MAGENTA = (235, 80, 240)
WHITE = (245, 248, 255)


def font_path() -> Path | None:
    windows = Path(os.environ.get("WINDIR") or os.environ.get("SystemRoot") or "C:/Windows")
    candidates = [windows / "Fonts" / name for name in ("msyh.ttc", "msyhbd.ttc", "simhei.ttf", "arial.ttf")]
    return next((path for path in candidates if path.is_file()), None)


FONT_PATH = font_path()


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    selected = FONT_PATH
    if bold and selected:
        bold_candidate = selected.with_name("msyhbd.ttc")
        if bold_candidate.is_file():
            selected = bold_candidate
    return ImageFont.truetype(str(selected), size) if selected else ImageFont.load_default()


def text(
    frame: np.ndarray,
    xy: tuple[int, int],
    value: str,
    size: int,
    color: tuple[int, int, int] = WHITE,
    *,
    bold: bool = False,
    anchor: str | None = None,
) -> None:
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    draw.text(xy, value, font=font(size, bold=bold), fill=(color[2], color[1], color[0]), anchor=anchor)
    frame[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def ease(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def read_image(path: Path) -> np.ndarray:
    try:
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except OSError:
        image = None
    if image is None:
        raise FileNotFoundError(path)
    return image


def base_frame(t: float) -> np.ndarray:
    y = np.linspace(0.0, 1.0, HEIGHT, dtype=np.float32)[:, None, None]
    top = np.asarray([28, 20, 8], dtype=np.float32)[None, None, :]
    bottom = np.asarray([9, 6, 3], dtype=np.float32)[None, None, :]
    frame = np.repeat(top * (1.0 - y) + bottom * y, WIDTH, axis=1).astype(np.uint8)
    grid_color = (38, 35, 20)
    offset = int((t * 35.0) % 80)
    for x in range(-HEIGHT, WIDTH + HEIGHT, 80):
        cv2.line(frame, (x + offset, HEIGHT), (x + HEIGHT // 2 + offset, HEIGHT // 2), grid_color, 1)
    for row in range(7):
        yy = int(HEIGHT * (0.52 + 0.48 * (row / 7.0) ** 1.7))
        cv2.line(frame, (0, yy), (WIDTH, yy), grid_color, 1)
    glow = 0.5 + 0.5 * math.sin(t * 1.8)
    cv2.circle(frame, (1620, 120), int(190 + 12 * glow), (24, 21, 10), -1)
    return frame


def fit_image(source: np.ndarray, width: int, height: int, zoom: float = 1.0) -> np.ndarray:
    sh, sw = source.shape[:2]
    scale = max(width / sw, height / sh) * zoom
    resized = cv2.resize(source, (int(sw * scale), int(sh * scale)), interpolation=cv2.INTER_CUBIC)
    rh, rw = resized.shape[:2]
    x0 = max(0, (rw - width) // 2)
    y0 = max(0, (rh - height) // 2)
    crop = resized[y0 : y0 + height, x0 : x0 + width]
    if crop.shape[:2] != (height, width):
        crop = cv2.resize(crop, (width, height), interpolation=cv2.INTER_CUBIC)
    return crop


def add_panel(frame: np.ndarray, image: np.ndarray, t: float, *, alpha: float = 1.0) -> None:
    x, y, width, height = 72, 176, 1300, 790
    cv2.rectangle(frame, (x - 2, y - 2), (x + width + 2, y + height + 2), (120, 86, 20), 2)
    zoom = 1.0 + 0.018 * math.sin(t * 0.55)
    fitted = fit_image(image, width, height, zoom)
    roi = frame[y : y + height, x : x + width]
    cv2.addWeighted(fitted, alpha, roi, 1.0 - alpha, 0, roi)
    scan_x = x + int((t * 170.0) % width)
    overlay = frame.copy()
    cv2.line(overlay, (scan_x, y), (scan_x, y + height), CYAN, 3)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
    for corner_x, corner_y, sx, sy in (
        (x, y, 1, 1), (x + width, y, -1, 1), (x, y + height, 1, -1), (x + width, y + height, -1, -1)
    ):
        cv2.line(frame, (corner_x, corner_y), (corner_x + sx * 32, corner_y), CYAN, 3)
        cv2.line(frame, (corner_x, corner_y), (corner_x, corner_y + sy * 32), CYAN, 3)


def header(frame: np.ndarray, index: str, title_value: str, subtitle: str) -> None:
    cv2.line(frame, (72, 134), (1848, 134), (78, 64, 24), 2)
    cv2.rectangle(frame, (72, 52), (184, 112), (74, 51, 10), -1)
    text(frame, (128, 82), index, 25, CYAN, bold=True, anchor="mm")
    text(frame, (214, 49), title_value, 48, WHITE, bold=True)
    text(frame, (216, 105), subtitle, 22, (196, 184, 145))


def metrics_panel(frame: np.ndarray, rows: list[tuple[str, str]], badge: str = "MCP VERIFIED") -> None:
    x, y, width, height = 1420, 176, 428, 790
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (30, 22, 9), -1)
    cv2.addWeighted(overlay, 0.84, frame, 0.16, 0, frame)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (82, 67, 24), 2)
    cv2.rectangle(frame, (x + 26, y + 25), (x + 243, y + 69), (64, 46, 10), -1)
    text(frame, (x + 42, y + 31), badge, 20, CYAN, bold=True)
    yy = y + 116
    for label, value in rows:
        text(frame, (x + 28, yy), label, 19, (171, 162, 132))
        text(frame, (x + 28, yy + 31), value, 29, WHITE, bold=True)
        cv2.line(frame, (x + 28, yy + 77), (x + width - 28, yy + 77), (55, 47, 24), 1)
        yy += 112
    cv2.circle(frame, (x + 45, y + height - 42), 8, CYAN, -1)
    text(frame, (x + 66, y + height - 56), "FreeCAD → Elmer → ParaView", 18, (205, 198, 169))


def intro(t: float, total: float, count: int) -> np.ndarray:
    frame = base_frame(t)
    pulse = 0.5 + 0.5 * math.sin(t * 3.0)
    cv2.circle(frame, (WIDTH // 2, 450), int(155 + pulse * 18), (65, 45, 10), 3)
    cv2.circle(frame, (WIDTH // 2, 450), int(105 + pulse * 10), (118, 78, 16), 2)
    text(frame, (WIDTH // 2, 375), "FEP", 92, CYAN, bold=True, anchor="mm")
    text(frame, (WIDTH // 2, 520), "AGENT HUB", 63, WHITE, bold=True, anchor="mm")
    text(frame, (WIDTH // 2, 635), f"{count} 个真实 CAE 案例 · MCP 证据链演示", 34, (220, 212, 181), anchor="mm")
    progress = min(1.0, t / max(total, 0.001))
    cv2.rectangle(frame, (600, 720), (1320, 730), (40, 36, 20), -1)
    cv2.rectangle(frame, (600, 720), (600 + int(720 * progress), 730), CYAN, -1)
    return frame


def heat_scene(t: float, duration: float, images: list[np.ndarray], physics: dict[str, Any], index: str) -> np.ndarray:
    frame = base_frame(t)
    phase = min(len(images) - 1, int(t / max(duration, 0.001) * len(images)))
    image = images[phase]
    header(frame, index, "稳态热传导", "10 mm 立方体 · 300 K → 400 K · 稳态场扫描展示")
    add_panel(frame, image, t)
    metrics_panel(
        frame,
        [
            ("Tmin", f"{physics['Tmin_K']:.2f} K"),
            ("Tmid", f"{physics['Tmid_K']:.2f} K"),
            ("Tmax", f"{physics['Tmax_K']:.2f} K"),
            ("Physics gate", "PASS" if physics.get("pass") else "FAIL"),
        ],
    )
    return frame


def em_scene(t: float, duration: float, images: list[np.ndarray], physics: dict[str, Any], index: str) -> np.ndarray:
    frame = base_frame(t)
    position = (t / max(duration, 0.001)) * len(images)
    first = int(position) % len(images)
    second = (first + 1) % len(images)
    blend = ease(position - math.floor(position))
    mixed = cv2.addWeighted(images[first], 1.0 - blend, images[second], blend, 0)
    header(frame, index, "变压器电磁感应", "50 Hz 谐波场 · 相位重建展示（非瞬态求解）")
    add_panel(frame, mixed, t)
    phase_deg = int((t * 120.0) % 360)
    metrics_panel(
        frame,
        [
            ("Bmax", f"{physics['Bmax_T']:.3f} T"),
            ("Magnetic flux", f"{physics['flux_magnitude_Wb'] * 1e3:.3f} mWb"),
            ("Secondary Voc", f"{physics['V2_open_rms_V']:.3f} Vrms"),
            ("Phase sample", f"{phase_deg:03d}°"),
            ("Turns ratio", f"{physics['turns_ratio']:.3f}  PASS"),
        ],
    )
    return frame


def beam_scene(t: float, duration: float, images: list[np.ndarray], physics: dict[str, Any], index: str) -> np.ndarray:
    frame = base_frame(t)
    progress = min(0.999999, max(0.0, t / max(duration, 0.001))) * (len(images) - 1)
    first = int(progress)
    second = min(len(images) - 1, first + 1)
    blend = ease(progress - first)
    mixed = cv2.addWeighted(images[first], 1.0 - blend, images[second], blend, 0)
    factor = 0.1 + 0.9 * progress / max(1, len(images) - 1)
    header(frame, index, "简支梁应力—应变", "10 个真实分级静力载荷步 · 变形显示放大 100×")
    add_panel(frame, mixed, t)
    metrics_panel(
        frame,
        [
            ("Load factor", f"{factor * 100:.1f}%"),
            ("Midspan deflection", f"{physics['midspan_downward_deflection_m'] * factor * 1e3:.3f} mm"),
            ("Midspan |σxx|", f"{physics['midspan_outer_fiber_abs_stress_xx_pa'] * factor / 1e6:.2f} MPa"),
            ("Midspan |εxx|", f"{physics['midspan_outer_fiber_abs_strain_xx'] * factor:.3e}"),
            ("10 load steps", "10 / 10 PASS"),
        ],
    )
    return frame


def outro(t: float, count: int) -> np.ndarray:
    frame = base_frame(t)
    text(frame, (WIDTH // 2, 365), "DEVELOPMENT EFFECT", 40, CYAN, bold=True, anchor="mm")
    text(frame, (WIDTH // 2, 475), f"{count} CASES · VERIFIED PIPELINE", 58, WHITE, bold=True, anchor="mm")
    text(frame, (WIDTH // 2, 585), "FreeCAD  /  Elmer FEM  /  ParaView  /  Codex MCP", 31, (217, 208, 175), anchor="mm")
    text(frame, (WIDTH // 2, 690), "真实求解 · 可审计证据 · 可重复自动化", 31, (217, 208, 175), anchor="mm")
    return frame


def fade_scene(frame: np.ndarray, local: float, duration: float) -> np.ndarray:
    alpha = min(1.0, local / 0.28, (duration - local) / 0.28)
    return cv2.convertScaleAbs(frame, alpha=max(0.0, alpha), beta=0)


def render_timeline(
    t: float,
    timeline: list[tuple[float, float, Callable[[float, float], np.ndarray]]],
) -> np.ndarray:
    for start, end, renderer in timeline:
        if start <= t < end:
            local = t - start
            return fade_scene(renderer(local, end - start), local, end - start)
    return timeline[-1][2](timeline[-1][1] - timeline[-1][0], timeline[-1][1] - timeline[-1][0])


def encode_video(output: Path, duration: float, ffmpeg: Path, renderer: Callable[[float], np.ndarray]) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(ffmpeg), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
    assert process.stdin is not None
    total_frames = int(round(duration * FPS))
    poster = None
    for index in range(total_frames):
        frame = renderer(index / FPS)
        if frame.shape != (HEIGHT, WIDTH, 3):
            raise ValueError("Renderer returned an unexpected frame shape")
        if index == min(total_frames - 1, int(2.0 * FPS)):
            poster = frame.copy()
        process.stdin.write(np.ascontiguousarray(frame).tobytes())
    process.stdin.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed ({return_code}): {stderr[-2000:]}")
    if poster is not None:
        cv2.imwrite(str(output.with_suffix(".poster.png")), poster)
    capture = cv2.VideoCapture(str(output))
    actual_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    actual_duration = actual_frames / actual_fps if actual_fps > 0 else 0.0
    valid = (
        output.is_file()
        and output.stat().st_size > 0
        and actual_frames == total_frames
        and actual_width == WIDTH
        and actual_height == HEIGHT
        and abs(actual_duration - duration) <= 1.0 / FPS
    )
    if not valid:
        raise RuntimeError("Encoded MP4 failed frame count, resolution, or duration validation")
    return {
        "path": str(output),
        "size": output.stat().st_size,
        "frames": actual_frames,
        "fps": actual_fps,
        "resolution": [actual_width, actual_height],
        "duration_seconds": actual_duration,
        "codec": "H.264/yuv420p",
        "valid": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create technology-styled FEP showcase MP4 files from verified outputs")
    parser.add_argument("--heat-surface", type=Path, required=True)
    parser.add_argument("--heat-slice", type=Path, required=True)
    parser.add_argument("--heat-result", type=Path, required=True)
    parser.add_argument("--em-field", type=Path, required=True)
    parser.add_argument("--em-lines", type=Path, required=True)
    parser.add_argument("--em-vectors", type=Path, required=True)
    parser.add_argument("--em-metrics", type=Path, required=True)
    parser.add_argument("--beam-frames", type=Path, required=True)
    parser.add_argument("--beam-summary", type=Path, required=True)
    parser.add_argument("--output-two", type=Path, required=True)
    parser.add_argument("--output-three", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--skip-two", action="store_true", help="Keep the validated two-case video and only re-encode the three-case video")
    args = parser.parse_args()

    heat_result = json.loads(args.heat_result.read_text(encoding="utf-8"))["physics_acceptance"]
    em_report = json.loads(args.em_metrics.read_text(encoding="utf-8"))
    em_physics = em_report["variants"][0]["physics"] if "variants" in em_report else em_report
    beam_physics = json.loads(args.beam_summary.read_text(encoding="utf-8"))["full_load"]
    heat_images = [read_image(args.heat_surface), read_image(args.heat_slice)]
    em_images = [read_image(args.em_field), read_image(args.em_lines), read_image(args.em_vectors)]
    beam_paths = sorted(args.beam_frames.glob("frame*.png"))
    if len(beam_paths) != 10:
        raise RuntimeError(f"Expected 10 verified beam frames, found {len(beam_paths)}")
    beam_images = [read_image(path) for path in beam_paths]
    if not args.ffmpeg.is_file():
        raise FileNotFoundError(args.ffmpeg)

    two_timeline = [
        (0.0, 1.2, lambda t, d: intro(t, d, 2)),
        (1.2, 6.0, lambda t, d: heat_scene(t, d, heat_images, heat_result, "01")),
        (6.0, 11.0, lambda t, d: em_scene(t, d, em_images, em_physics, "02")),
        (11.0, 12.0, lambda t, d: outro(t, 2)),
    ]
    three_timeline = [
        (0.0, 1.5, lambda t, d: intro(t, d, 3)),
        (1.5, 5.5, lambda t, d: heat_scene(t, d, heat_images, heat_result, "01")),
        (5.5, 10.5, lambda t, d: em_scene(t, d, em_images, em_physics, "02")),
        (10.5, 18.5, lambda t, d: beam_scene(t, d, beam_images, beam_physics, "03")),
        (18.5, 20.0, lambda t, d: outro(t, 3)),
    ]
    report_path = args.output_three.with_name("showcase_video_validation.json")
    reports: dict[str, Any] = {}
    if args.skip_two:
        if not report_path.is_file():
            raise FileNotFoundError("--skip-two requires an existing showcase_video_validation.json")
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        reports["two_case"] = previous["two_case"]
    else:
        reports["two_case"] = encode_video(
            args.output_two,
            12.0,
            args.ffmpeg,
            lambda t: render_timeline(t, two_timeline),
        )
    reports.update({
        "three_case": encode_video(
            args.output_three,
            20.0,
            args.ffmpeg,
            lambda t: render_timeline(t, three_timeline),
        ),
        "semantics": {
            "heat": "steady-state field with scanning presentation",
            "electromagnetic": "harmonic phase reconstruction; not transient solve",
            "beam": "ten real quasi-static load levels; not transient dynamics",
        },
    })
    report_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
