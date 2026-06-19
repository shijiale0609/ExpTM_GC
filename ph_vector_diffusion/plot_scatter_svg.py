#!/usr/bin/env python3
"""Render generated two-feature samples as a dependency-free SVG scatter plot."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


PALETTE = (
    "#D55E00",
    "#CC79A7",
    "#009E73",
    "#E69F00",
    "#0072B2",
    "#56B4E9",
    "#000000",
    "#999999",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-csv", type=Path, help="Generated CSV containing an LP1 column")
    source.add_argument(
        "--condition",
        action="append",
        nargs=2,
        metavar=("LP1", "CSV"),
        help="Raw training CSV for an LP1 value; repeat for each condition",
    )
    parser.add_argument("--output-svg", type=Path, required=True)
    parser.add_argument("--control-name", default="LP1")
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=700)
    return parser.parse_args()


def padded_extent(values: list[float]) -> tuple[float, float]:
    low, high = min(values), max(values)
    padding = max((high - low) * 0.06, 1e-6)
    return low - padding, high + padding


def main() -> None:
    args = parse_args()
    points: dict[float, list[tuple[float, float]]] = defaultdict(list)
    control_name = args.control_name
    if args.input_csv:
        with args.input_csv.open(newline="") as handle:
            reader = csv.DictReader(handle)
            required = {control_name, "x_top10mean", "y_rank450mean_ascending"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError(f"{args.input_csv} must contain {sorted(required)}")
            for row in reader:
                points[float(row[control_name])].append(
                    (float(row["x_top10mean"]), float(row["y_rank450mean_ascending"]))
                )
    else:
        for control_value_text, path_text in args.condition:
            control_value, path = float(control_value_text), Path(path_text)
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                required = {"x_top10mean", "y_rank450mean_ascending"}
                if not reader.fieldnames or not required.issubset(reader.fieldnames):
                    raise ValueError(f"{path} must contain {sorted(required)}")
                for row in reader:
                    points[control_value].append(
                        (float(row["x_top10mean"]), float(row["y_rank450mean_ascending"]))
                    )
    if not points:
        raise ValueError("No points found")

    all_x = [x for group in points.values() for x, _ in group]
    all_y = [y for group in points.values() for _, y in group]
    x_min, x_max = padded_extent(all_x)
    y_min, y_max = padded_extent(all_y)
    margin_left, margin_right, margin_top, margin_bottom = 105, 35, 45, 90
    plot_width = args.width - margin_left - margin_right
    plot_height = args.height - margin_top - margin_bottom

    def screen_x(value: float) -> float:
        return margin_left + (value - x_min) / (x_max - x_min) * plot_width

    def screen_y(value: float) -> float:
        return margin_top + (y_max - value) / (y_max - y_min) * plot_height

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{args.width}" height="{args.height}" viewBox="0 0 {args.width} {args.height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{args.width / 2}" y="28" text-anchor="middle" font-family="Arial, sans-serif" font-size="19">Conditional diffusion samples by {control_name}</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#222"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#222"/>',
    ]

    for index in range(6):
        fraction = index / 5
        x_value = x_min + fraction * (x_max - x_min)
        x_position = screen_x(x_value)
        y_value = y_min + fraction * (y_max - y_min)
        y_position = screen_y(y_value)
        elements.extend(
            [
                f'<line x1="{x_position:.2f}" y1="{margin_top}" x2="{x_position:.2f}" y2="{margin_top + plot_height}" stroke="#dddddd"/>',
                f'<text x="{x_position:.2f}" y="{margin_top + plot_height + 23}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12">{x_value:.1f}</text>',
                f'<line x1="{margin_left}" y1="{y_position:.2f}" x2="{margin_left + plot_width}" y2="{y_position:.2f}" stroke="#dddddd"/>',
                f'<text x="{margin_left - 10}" y="{y_position + 4:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12">{y_value:.1f}</text>',
            ]
        )

    ordered_controls = sorted(points)
    colors = {value: PALETTE[index % len(PALETTE)] for index, value in enumerate(ordered_controls)}
    for control_value in ordered_controls:
        color = colors[control_value]
        for x_value, y_value in points[control_value]:
            elements.append(
                f'<circle cx="{screen_x(x_value):.2f}" cy="{screen_y(y_value):.2f}" r="2.1" fill="{color}" fill-opacity="0.34"/>'
            )

    elements.extend(
        [
            f'<text x="{margin_left + plot_width / 2}" y="{args.height - 22}" text-anchor="middle" font-family="Arial, sans-serif" font-size="15">x_top10mean</text>',
            f'<text x="24" y="{margin_top + plot_height / 2}" transform="rotate(-90 24 {margin_top + plot_height / 2})" text-anchor="middle" font-family="Arial, sans-serif" font-size="15">y_rank450mean_ascending</text>',
        ]
    )
    legend_x, legend_y = margin_left + 15, margin_top + 22
    for index, control_value in enumerate(ordered_controls):
        color = colors[control_value]
        y_position = legend_y + index * 23
        elements.extend(
            [
                f'<circle cx="{legend_x}" cy="{y_position}" r="5" fill="{color}"/>',
                f'<text x="{legend_x + 12}" y="{y_position + 4}" font-family="Arial, sans-serif" font-size="13">{control_name} {control_value:g} (n={len(points[control_value])})</text>',
            ]
        )
    elements.append("</svg>")
    args.output_svg.parent.mkdir(parents=True, exist_ok=True)
    args.output_svg.write_text("\n".join(elements) + "\n")
    print(f"wrote {sum(map(len, points.values()))} points to {args.output_svg}")


if __name__ == "__main__":
    main()
