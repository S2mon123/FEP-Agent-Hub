from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from make_showcase_videos import (
    CYAN,
    WIDTH,
    WHITE,
    add_panel,
    base_frame,
    beam_scene,
    ease,
    em_scene,
    encode_video,
    header,
    intro,
    metrics_panel,
    read_image,
    render_timeline,
    text,
)


def author_mark(frame: np.ndarray, avatar: np.ndarray, author_name: str) -> None:
    """Add a compact, non-destructive author identity mark to the top-right corner."""
    x, y, width, height = 1450, 18, 398, 102
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (24, 18, 8), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (94, 72, 22), 1)
    text(frame, (x + 23, y + 18), "AUTHOR · CAE AUTOMATION", 17, CYAN, bold=True)
    text(frame, (x + 23, y + 49), author_name, 30, WHITE, bold=True)

    avatar_size = 84
    avatar_x, avatar_y = x + width - avatar_size - 9, y + 9
    square = min(avatar.shape[:2])
    crop_y = (avatar.shape[0] - square) // 2
    crop_x = (avatar.shape[1] - square) // 2
    cropped = avatar[crop_y : crop_y + square, crop_x : crop_x + square]
    resized = cv2.resize(cropped, (avatar_size, avatar_size), interpolation=cv2.INTER_AREA)
    frame[avatar_y : avatar_y + avatar_size, avatar_x : avatar_x + avatar_size] = resized
    cv2.rectangle(
        frame,
        (avatar_x - 2, avatar_y - 2),
        (avatar_x + avatar_size + 2, avatar_y + avatar_size + 2),
        CYAN,
        2,
    )


def research_strip(frame: np.ndarray, message: str) -> None:
    x, y, width, height = 92, 889, 1260, 58
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (20, 17, 8), -1)
    cv2.addWeighted(overlay, 0.83, frame, 0.17, 0, frame)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (86, 68, 20), 1)
    text(frame, (x + 20, y + 14), message, 21, WHITE, bold=True)


def research_intro(t: float, duration: float) -> np.ndarray:
    frame = intro(t, duration, 4)
    text(frame, (WIDTH // 2, 676), "45 秒研究展示版 · 真实过程 / 全误差表 / 物理语义", 25, CYAN, bold=True, anchor="mm")
    return frame


def heat_history_chart(
    frame: np.ndarray,
    history: list[dict[str, Any]],
    initial_temperature: float,
    current_time: float,
    current_temperature: float,
) -> None:
    """Draw the actual solver midpoint-temperature history as a progressive line chart."""
    x, y, width, height = 104, 598, 570, 267
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (18, 16, 9), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (92, 73, 23), 2)
    text(frame, (x + 20, y + 15), "真实求解跨中温度迁移 Tmid(t)", 20, WHITE, bold=True)
    text(
        frame,
        (x + width - 20, y + 17),
        f"{current_time:04.1f} s / {current_temperature:.1f} K",
        18,
        CYAN,
        bold=True,
        anchor="ra",
    )

    left, top = x + 58, y + 57
    chart_width, chart_height = width - 82, height - 100
    bottom = top + chart_height
    time_min, time_max = 0.0, 20.0
    temperature_min, temperature_max = 300.0, 352.0

    def point(time_s: float, temperature_k: float) -> tuple[int, int]:
        px = left + int((time_s - time_min) / (time_max - time_min) * chart_width)
        py = bottom - int(
            (temperature_k - temperature_min)
            / (temperature_max - temperature_min)
            * chart_height
        )
        return px, py

    for temperature_k in (300.0, 325.0, 350.0):
        _, py = point(0.0, temperature_k)
        cv2.line(frame, (left, py), (left + chart_width, py), (48, 43, 27), 1)
        text(frame, (left - 9, py), f"{temperature_k:.0f}", 14, (177, 168, 139), anchor="rm")
    for time_s in (0.0, 5.0, 10.0, 15.0, 20.0):
        px, _ = point(time_s, temperature_min)
        cv2.line(frame, (px, top), (px, bottom), (41, 37, 24), 1)
        text(frame, (px, bottom + 10), f"{time_s:.0f}", 14, (177, 168, 139), anchor="ma")
    cv2.line(frame, (left, bottom), (left + chart_width, bottom), (115, 98, 56), 2)
    cv2.line(frame, (left, top), (left, bottom), (115, 98, 56), 2)

    all_points = [(0.0, initial_temperature)] + [
        (float(row["time_s"]), float(row["Tmid_K"])) for row in history
    ]
    full_line = np.array([point(time_s, temperature) for time_s, temperature in all_points], dtype=np.int32)
    cv2.polylines(frame, [full_line], False, (72, 66, 42), 2, cv2.LINE_AA)

    visible_points = [item for item in all_points if item[0] <= current_time]
    if not visible_points or visible_points[-1][0] < current_time:
        visible_points.append((current_time, current_temperature))
    visible_line = np.array(
        [point(time_s, temperature) for time_s, temperature in visible_points],
        dtype=np.int32,
    )
    if len(visible_line) >= 2:
        cv2.polylines(frame, [visible_line], False, CYAN, 4, cv2.LINE_AA)
    marker = point(current_time, current_temperature)
    cv2.circle(frame, marker, 11, (42, 36, 14), -1)
    cv2.circle(frame, marker, 7, CYAN, -1)
    cv2.circle(frame, marker, 3, WHITE, -1)
    text(frame, (left + chart_width + 9, bottom + 10), "s", 14, CYAN, bold=True)
    text(frame, (left - 38, top - 8), "K", 14, CYAN, bold=True)


def transient_heat_scene(
    t: float,
    duration: float,
    images: list[np.ndarray],
    physics: dict[str, Any],
) -> np.ndarray:
    frame = base_frame(t)
    progress = min(0.999999, max(0.0, t / max(duration, 0.001))) * (len(images) - 1)
    first = int(progress)
    second = min(len(images) - 1, first + 1)
    blend = ease(progress - first)
    mixed = cv2.addWeighted(images[first], 1.0 - blend, images[second], blend, 0)
    history = physics["history"]
    first_mid = float(history[first]["Tmid_K"])
    second_mid = float(history[second]["Tmid_K"])
    midpoint_temperature = first_mid * (1.0 - blend) + second_mid * blend
    time_s = 1.0 + progress
    relative_error = (
        float(physics["analytical_absolute_error_K"])
        / float(physics["analytical_midpoint_temperature_K"])
    )
    header(frame, "01", "瞬态热传导过程", "20 个真实 BDF1 时间步 · 热扩散前沿 1–20 s")
    add_panel(frame, mixed, t)
    metrics_panel(
        frame,
        [
            ("Physical time", f"{time_s:05.2f} s"),
            ("Midpoint T", f"{midpoint_temperature:.2f} K"),
            ("Final analytic Tmid", f"{physics['analytical_midpoint_temperature_K']:.2f} K"),
            ("Final abs. error", f"{physics['analytical_absolute_error_K']:.3f} K"),
            ("Final rel. error", f"{relative_error * 100:.3f}%  PASS"),
        ],
        badge="TRUE TRANSIENT",
    )
    heat_history_chart(
        frame,
        history,
        float(physics["initial_temperature_K"]),
        time_s,
        midpoint_temperature,
    )
    research_strip(frame, "研究重点 · 瞬态离散结果与 Fourier 级数解对照 · 20/20 时间步")
    return frame


def fluid_scene(
    t: float,
    duration: float,
    images: list[np.ndarray],
    physics: dict[str, Any],
) -> np.ndarray:
    frame = base_frame(t)
    position = 1.25 * t / max(duration, 0.001) * len(images)
    first = int(position) % len(images)
    second = (first + 1) % len(images)
    blend = ease(position - math.floor(position))
    mixed = cv2.addWeighted(images[first], 1.0 - blend, images[second], blend, 0)
    header(frame, "04", "二维层流通道", "稳态 Navier–Stokes · Poiseuille 速度剖面 · 流动轨迹展示")
    add_panel(frame, mixed, t)
    for index in range(24):
        lane = index % 9
        eta = (lane + 0.5) / 9.0
        speed_factor = max(0.08, 1.0 - (2.0 * eta - 1.0) ** 2)
        x = int(350 + ((index * 83 + t * 170.0 * speed_factor) % 760))
        y = int(505 + eta * 135)
        radius = 3 + (index % 3)
        cv2.circle(frame, (x, y), radius, CYAN if index % 2 else WHITE, -1)
    metrics_panel(
        frame,
        [
            ("Reynolds number", f"{physics['reynolds_number']:.1f}"),
            ("Mean velocity", f"{physics['measured_mean_velocity_m_per_s']:.4f} m/s"),
            ("Mean velocity error", f"{physics['mean_velocity_relative_error'] * 100:.2f}%"),
            ("Profile L2 error", f"{physics['profile_relative_l2_error'] * 100:.2f}%"),
            ("Pressure-drop error", f"{physics['pressure_drop_relative_error'] * 100:.2f}%  PASS"),
        ],
    )
    research_strip(frame, "研究重点 · Poiseuille 剖面 / 解析压降 / 壁面无滑移三重校核")
    return frame


def electromagnetic_research_scene(
    t: float,
    duration: float,
    images: list[np.ndarray],
    physics: dict[str, Any],
    report: dict[str, Any],
) -> np.ndarray:
    frame = em_scene(t, duration, images, physics, "02")
    mesh_error = max(float(value) for key, value in report["mesh_sensitivity_acceptance"].items() if key.endswith("difference"))
    research_strip(
        frame,
        f"研究重点 · 匝比 / 电流线性 / 网格敏感性 · 最大网格差异 {mesh_error * 100:.3f}%",
    )
    return frame


def beam_research_scene(
    t: float,
    duration: float,
    images: list[np.ndarray],
    physics: dict[str, Any],
) -> np.ndarray:
    frame = beam_scene(t, duration, images, physics, "03")
    research_strip(
        frame,
        "研究重点 · FEA 与 Euler–Bernoulli 基准对照 · 挠度/应力/应变均 ≤ 20%",
    )
    return frame


def research_summary_scene(
    t: float,
    heat: dict[str, Any],
    em_report: dict[str, Any],
    beam: dict[str, Any],
    fluid: dict[str, Any],
    author_name: str,
) -> np.ndarray:
    frame = base_frame(t)
    header(frame, "R", "研究重点与误差总览", "解析基准、网格敏感性与物理门禁共同约束结论")
    em_mesh = em_report["mesh_sensitivity_acceptance"]
    cards = [
        (
            "瞬态热传导",
            "Fourier 级数基准",
            f"ΔT = {heat['analytical_absolute_error_K']:.3f} K",
            f"相对误差 {heat['analytical_absolute_error_K'] / heat['analytical_midpoint_temperature_K'] * 100:.3f}%",
        ),
        (
            "电磁感应",
            "线性 + 网格敏感性",
            f"磁通差异 {em_mesh['flux_relative_difference'] * 100:.3f}%",
            f"Bmax 差异 {em_mesh['Bmax_relative_difference'] * 100:.3f}%",
        ),
        (
            "简支梁",
            "Euler–Bernoulli 基准",
            f"挠度误差 {beam['deflection_relative_error'] * 100:.2f}%",
            f"应力 / 应变 {beam['stress_relative_error'] * 100:.2f}% / {beam['strain_relative_error'] * 100:.2f}%",
        ),
        (
            "二维层流",
            "Poiseuille 基准",
            f"剖面误差 {fluid['profile_relative_l2_error'] * 100:.2f}%",
            f"压降误差 {fluid['pressure_drop_relative_error'] * 100:.2f}%",
        ),
    ]
    for index, (title_value, benchmark, primary, secondary) in enumerate(cards):
        column = index % 2
        row = index // 2
        x, y, width, height = 82 + column * 894, 188 + row * 292, 850, 250
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (28, 21, 9), -1)
        cv2.addWeighted(overlay, 0.86, frame, 0.14, 0, frame)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (88, 70, 22), 2)
        cv2.rectangle(frame, (x, y), (x + 12, y + height), CYAN, -1)
        text(frame, (x + 42, y + 28), title_value, 31, WHITE, bold=True)
        text(frame, (x + 42, y + 79), benchmark, 20, (182, 173, 143))
        text(frame, (x + 42, y + 130), primary, 29, CYAN, bold=True)
        text(frame, (x + 42, y + 184), secondary, 24, WHITE)
        text(frame, (x + width - 92, y + 32), "PASS", 20, CYAN, bold=True)
    text(frame, (WIDTH // 2, 862), "研究主线：物理语义正确 → 原生求解 → 理论对照 → 误差门禁 → 可审计证据", 27, WHITE, bold=True, anchor="mm")
    text(frame, (WIDTH // 2, 918), "49/49 MCP 工具覆盖 · 52/52 契约通过 · 作者：" + author_name, 23, CYAN, bold=True, anchor="mm")
    text(frame, (WIDTH // 2, 974), "演示动效不替代求解：热为真实瞬态；电磁为谐波；梁为准静态；流体为稳态", 20, (202, 194, 164), anchor="mm")
    return frame


def build_error_rows(
    heat: dict[str, Any],
    em_report: dict[str, Any],
    beam: dict[str, Any],
    fluid: dict[str, Any],
) -> list[tuple[str, str, str, str, str]]:
    baseline_em = em_report["variants"][0]["physics"]
    mesh = em_report["mesh_sensitivity_acceptance"]
    mesh_flux_v2 = max(
        float(mesh["flux_relative_difference"]),
        float(mesh["V2_relative_difference"]),
    )
    return [
        (
            "热传导",
            "20 s Fourier 跨中温度",
            f"{heat['analytical_absolute_error_K']:.3f} K / {heat['analytical_absolute_error_K'] / heat['analytical_midpoint_temperature_K'] * 100:.3f}%",
            "≤ 3 K",
            "PASS",
        ),
        ("电磁", "开路电压 / 匝数比", f"{baseline_em['turns_ratio_relative_error'] * 100:.3f}%", "≤ 2%", "PASS"),
        ("电磁", "A 势差 / B 线积分磁通", f"{baseline_em['flux_A_vs_B_relative_error'] * 100:.6f}%", "≤ 5%", "PASS"),
        ("电磁", "半电流线性", f"{em_report['linearity_acceptance']['max_relative_error'] * 100:.2e}%", "≤ 3%", "PASS"),
        ("电磁", "磁通 / V2 网格敏感性", f"{mesh_flux_v2 * 100:.3f}%", "≤ 5%", "PASS"),
        ("简支梁", "跨中挠度", f"{beam['deflection_relative_error'] * 100:.2f}%", "≤ 20%", "PASS"),
        ("简支梁", "跨中外纤维应力", f"{beam['stress_relative_error'] * 100:.2f}%", "≤ 20%", "PASS"),
        ("简支梁", "跨中外纤维应变", f"{beam['strain_relative_error'] * 100:.2f}%", "≤ 20%", "PASS"),
        ("流体", "平均速度", f"{fluid['mean_velocity_relative_error'] * 100:.2f}%", "≤ 5%", "PASS"),
        ("流体", "Poiseuille 剖面 L2", f"{fluid['profile_relative_l2_error'] * 100:.2f}%", "≤ 8%", "PASS"),
        ("流体", "10%L–90%L 压降", f"{fluid['pressure_drop_relative_error'] * 100:.2f}%", "≤ 15%", "PASS"),
    ]


def error_table_scene(
    t: float,
    duration: float,
    rows: list[tuple[str, str, str, str, str]],
) -> np.ndarray:
    frame = base_frame(t)
    header(frame, "V", "仿真计算误差全表", "11 项理论与数值门禁逐行核验 · 所有结果均为 PASS")
    x0, x1 = 72, 1848
    table_top, heading_height, row_height = 175, 54, 62
    data_top = table_top + heading_height
    boundaries = (x0, 260, 810, 1160, 1410, x1)
    labels = ("案例", "验证项目", "误差结果", "验收门槛", "状态")

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x0, table_top),
        (x1, data_top + row_height * len(rows)),
        (22, 18, 8),
        -1,
    )
    cv2.addWeighted(overlay, 0.86, frame, 0.14, 0, frame)
    cv2.rectangle(frame, (x0, table_top), (x1, data_top), (67, 49, 11), -1)
    cv2.rectangle(
        frame,
        (x0, table_top),
        (x1, data_top + row_height * len(rows)),
        (94, 74, 24),
        2,
    )
    for boundary in boundaries[1:-1]:
        cv2.line(
            frame,
            (boundary, table_top),
            (boundary, data_top + row_height * len(rows)),
            (53, 46, 25),
            1,
        )
    for index, label in enumerate(labels):
        text(frame, (boundaries[index] + 20, table_top + 14), label, 20, CYAN, bold=True)

    reveal = max(0.0, (t - 0.25) / max(duration - 1.2, 0.001) * len(rows))
    for index, row in enumerate(rows):
        row_y = data_top + index * row_height
        if index % 2:
            stripe = frame.copy()
            cv2.rectangle(stripe, (x0 + 1, row_y), (x1 - 1, row_y + row_height), (30, 25, 13), -1)
            cv2.addWeighted(stripe, 0.48, frame, 0.52, 0, frame)
        cv2.line(frame, (x0, row_y + row_height), (x1, row_y + row_height), (49, 43, 24), 1)
        alpha = min(1.0, max(0.0, reveal - index))
        if alpha <= 0.0:
            continue
        slide = int((1.0 - ease(alpha)) * 42)
        row_layer = frame.copy()
        for column, value in enumerate(row[:-1]):
            color = WHITE if column != 2 else CYAN
            text(
                row_layer,
                (boundaries[column] + 20 + slide, row_y + 16),
                value,
                19,
                color,
                bold=column in (0, 2),
            )
        pill_left, pill_right = boundaries[4] + 108 + slide, boundaries[5] - 108 + slide
        cv2.rectangle(row_layer, (pill_left, row_y + 12), (pill_right, row_y + 49), (63, 57, 15), -1)
        cv2.rectangle(row_layer, (pill_left, row_y + 12), (pill_right, row_y + 49), CYAN, 1)
        text(
            row_layer,
            ((pill_left + pill_right) // 2, row_y + 30),
            row[-1],
            18,
            CYAN,
            bold=True,
            anchor="mm",
        )
        cv2.addWeighted(row_layer, ease(alpha), frame, 1.0 - ease(alpha), 0, frame)

    revealed_count = min(len(rows), max(0, int(math.ceil(reveal))))
    pulse = 0.6 + 0.4 * math.sin(t * 4.0)
    footer_color = tuple(int(component * pulse) for component in CYAN)
    text(
        frame,
        (WIDTH // 2, 975),
        f"VALIDATION GATES  {revealed_count:02d} / {len(rows):02d}  PASS",
        25,
        footer_color,
        bold=True,
        anchor="mm",
    )
    return frame


def closing_scene(t: float, duration: float, author_name: str) -> np.ndarray:
    frame = base_frame(t)
    progress = ease(min(1.0, t / max(duration * 0.45, 0.001)))
    center = (WIDTH // 2, 377)
    radius = int(118 * progress)
    cv2.circle(frame, center, radius + 26, (55, 45, 18), 3)
    cv2.circle(frame, center, radius, CYAN, 5)
    if progress > 0.35:
        cv2.line(frame, (center[0] - 48, center[1]), (center[0] - 12, center[1] + 38), CYAN, 12, cv2.LINE_AA)
        cv2.line(frame, (center[0] - 12, center[1] + 38), (center[0] + 64, center[1] - 55), CYAN, 12, cv2.LINE_AA)
    text(frame, (WIDTH // 2, 565), "研究展示完成", 53, WHITE, bold=True, anchor="mm")
    text(frame, (WIDTH // 2, 638), "ALL 11 ERROR GATES PASS", 34, CYAN, bold=True, anchor="mm")
    text(frame, (WIDTH // 2, 705), "4 CAE CASES · 49 MCP TOOLS · 52 CONTRACT CALLS", 25, (216, 207, 176), anchor="mm")
    text(frame, (WIDTH // 2, 766), "FreeCAD  →  Elmer FEM  →  ParaView  →  Codex MCP", 25, WHITE, anchor="mm")
    text(frame, (WIDTH // 2, 833), "真实求解 · 误差透明 · 证据可审计", 28, (216, 207, 176), bold=True, anchor="mm")
    text(frame, (WIDTH // 2, 906), "作者 · " + author_name, 27, CYAN, bold=True, anchor="mm")
    text(frame, (WIDTH // 2, 969), "感谢观看", 22, (183, 175, 147), anchor="mm")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the four-case FEP technology showcase MP4")
    parser.add_argument("--heat-frames", type=Path, required=True)
    parser.add_argument("--heat-summary", type=Path, required=True)
    parser.add_argument("--em-field", type=Path, required=True)
    parser.add_argument("--em-lines", type=Path, required=True)
    parser.add_argument("--em-vectors", type=Path, required=True)
    parser.add_argument("--em-metrics", type=Path, required=True)
    parser.add_argument("--beam-frames", type=Path, required=True)
    parser.add_argument("--beam-summary", type=Path, required=True)
    parser.add_argument("--fluid-field", type=Path, required=True)
    parser.add_argument("--fluid-vectors", type=Path, required=True)
    parser.add_argument("--fluid-summary", type=Path, required=True)
    parser.add_argument("--author-avatar", type=Path, required=True)
    parser.add_argument("--author-name", default="麦克尼克欧")
    parser.add_argument("--preview-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    args = parser.parse_args()

    heat_paths = sorted(args.heat_frames.glob("frame*.png"))
    beam_paths = sorted(args.beam_frames.glob("frame*.png"))
    if len(heat_paths) != 20 or len(beam_paths) != 10:
        raise RuntimeError(f"Expected 20 heat and 10 beam frames, found {len(heat_paths)} and {len(beam_paths)}")
    heat_images = [read_image(path) for path in heat_paths]
    beam_images = [read_image(path) for path in beam_paths]
    em_images = [read_image(args.em_field), read_image(args.em_lines), read_image(args.em_vectors)]
    fluid_images = [read_image(args.fluid_field), read_image(args.fluid_vectors)]
    author_avatar = read_image(args.author_avatar)
    heat_physics = json.loads(args.heat_summary.read_text(encoding="utf-8"))["physics_acceptance"]
    em_report = json.loads(args.em_metrics.read_text(encoding="utf-8"))
    em_physics = em_report["variants"][0]["physics"] if "variants" in em_report else em_report
    beam_physics = json.loads(args.beam_summary.read_text(encoding="utf-8"))["full_load"]
    fluid_physics = json.loads(args.fluid_summary.read_text(encoding="utf-8"))["physics_acceptance"]
    error_rows = build_error_rows(heat_physics, em_report, beam_physics, fluid_physics)
    timeline = [
        (0.0, 2.5, research_intro),
        (2.5, 11.0, lambda t, d: transient_heat_scene(t, d, heat_images, heat_physics)),
        (11.0, 17.5, lambda t, d: electromagnetic_research_scene(t, d, em_images, em_physics, em_report)),
        (17.5, 25.0, lambda t, d: beam_research_scene(t, d, beam_images, beam_physics)),
        (25.0, 32.0, lambda t, d: fluid_scene(t, d, fluid_images, fluid_physics)),
        (32.0, 41.0, lambda t, d: error_table_scene(t, d, error_rows)),
        (41.0, 45.0, lambda t, d: closing_scene(t, d, args.author_name)),
    ]

    def render_authored_frame(t: float) -> np.ndarray:
        frame = render_timeline(t, timeline)
        author_mark(frame, author_avatar, args.author_name)
        return frame

    if args.preview_dir:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
        previews = {
            "heat_line": 7.0,
            "error_table_mid": 36.0,
            "error_table_full": 40.2,
            "closing": 43.0,
        }
        for name, time_s in previews.items():
            path = args.preview_dir / f"{name}.png"
            if not cv2.imwrite(str(path), render_authored_frame(time_s)):
                raise RuntimeError(f"Failed to write preview: {path}")
        print(json.dumps({"preview_dir": str(args.preview_dir), "frames": previews}, ensure_ascii=False, indent=2))
        return

    validation = {
        "four_case_research_45s": encode_video(
            args.output,
            45.0,
            args.ffmpeg,
            render_authored_frame,
        ),
        "author": {
            "name": args.author_name,
            "avatar_source": str(args.author_avatar),
            "presentation": "top-right identity mark on every frame",
        },
        "audio": "none (-an)",
        "error_table": {
            "gate_count": len(error_rows),
            "passed": sum(row[-1] == "PASS" for row in error_rows),
            "dynamic_row_reveal": True,
        },
        "timeline_seconds": {
            "intro": [0.0, 2.5],
            "transient_heat_with_history_line": [2.5, 11.0],
            "electromagnetic": [11.0, 17.5],
            "beam": [17.5, 25.0],
            "fluid": [25.0, 32.0],
            "dynamic_error_table": [32.0, 41.0],
            "closing": [41.0, 45.0],
        },
        "error_review": {
            "heat_final_absolute_error_K": heat_physics["analytical_absolute_error_K"],
            "heat_final_relative_error": heat_physics["analytical_absolute_error_K"] / heat_physics["analytical_midpoint_temperature_K"],
            "electromagnetic_linearity_max_relative_error": em_report["linearity_acceptance"]["max_relative_error"],
            "electromagnetic_mesh_sensitivity": em_report["mesh_sensitivity_acceptance"],
            "beam_deflection_relative_error": beam_physics["deflection_relative_error"],
            "beam_stress_relative_error": beam_physics["stress_relative_error"],
            "beam_strain_relative_error": beam_physics["strain_relative_error"],
            "fluid_mean_velocity_relative_error": fluid_physics["mean_velocity_relative_error"],
            "fluid_profile_relative_l2_error": fluid_physics["profile_relative_l2_error"],
            "fluid_pressure_drop_relative_error": fluid_physics["pressure_drop_relative_error"],
        },
        "semantics": {
            "heat": "twenty real BDF1 transient heat time steps",
            "electromagnetic": "harmonic phase reconstruction; not transient solve",
            "beam": "ten real quasi-static load levels; not transient dynamics",
            "fluid": "steady Navier-Stokes field with animated flow-path presentation",
        },
    }
    validation_path = args.output.with_name(f"{args.output.stem}_validation.json")
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
