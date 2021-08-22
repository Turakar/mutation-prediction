# adapted from https://github.com/kevinkovalchik/Plotly-Logo
import collections
import os
import pathlib
from random import uniform as r_uniform
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import plotly.colors
import plotly.graph_objects as go
from PIL import ImageColor
from scipy.stats import entropy

from mutation_prediction import data

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class Point:
    """
    x,y point
    """

    def __init__(self, x: Union[int, float], y: Union[int, float]):
        self.x: int = x
        self.y: int = y


class PathSegment:
    """
    A segment of an SVG path. Consists of the kind of path (i.e. M, L, Q, C, Z) and corresponding x,y points. "A" is not
    supported by Plotly and thus is not a valid kind of path.
    """

    def __init__(self, kind: str, coords: Union[List[Point], None]):

        if kind == "M":
            if not isinstance(coords, list):
                raise TypeError(
                    "For path segments of type 'M', the coordinate argument must be a list of 1 point"
                )
            if len(coords) != 1:
                raise ValueError(
                    "For path segments of type 'M', the coordinate argument must be a list of 1 point"
                )
            if not isinstance(coords[0], Point):
                raise TypeError(
                    "For path segments of type 'M', the coordinate argument must be a list of 1 point"
                )

        elif kind == "L":
            if not isinstance(coords, list):
                raise TypeError(
                    "For path segments of type 'L', the coordinate argument must be a list of 1 point"
                )
            if len(coords) != 1:
                raise ValueError(
                    "For path segments of type 'L', the coordinate argument must be a list of 1 point"
                )
            if not isinstance(coords[0], Point):
                raise TypeError(
                    "For path segments of type 'L', the coordinate argument must be a list of 1 point"
                )

        elif kind == "Q":
            if not isinstance(coords, list):
                raise TypeError(
                    "For path segments of type 'Q', the coordinate argument must be a list of 2 points"
                )
            if len(coords) != 2:
                raise ValueError(
                    "For path segments of type 'Q', the coordinate argument must be a list of 2 points"
                )
            if False in [isinstance(p, Point) for p in coords]:
                raise TypeError(
                    "For path segments of type 'Q', the coordinate argument must be a list of 2 points"
                )

        elif kind == "C":
            if not isinstance(coords, list):
                raise TypeError(
                    "For path segments of type 'C', the coordinate argument must be a list of 3 points"
                )
            if len(coords) != 3:
                raise ValueError(
                    "For path segments of type 'C', the coordinate argument must be a list of 3 points"
                )
            if False in [isinstance(p, Point) for p in coords]:
                raise TypeError(
                    "For path segments of type 'C', the coordinate argument must be a list of 3 points"
                )
        elif kind == "Z":
            if coords != []:
                raise TypeError(
                    "For path segments of type 'Z', the coordinate argument must an empty list"
                )
        elif kind == "A":
            raise ValueError(
                "Sorry. Plotly only works with absolute paths, so 'A' is not allowed in the SVG path."
            )

        self.kind: str = kind
        self.coords: List[Point] = coords


class Path:
    """
    An SVG path.
    """

    def __init__(self, path: str):
        path = path.strip()
        if ", " in path:
            path = path.replace(", ", " ").split(" ")
        elif "," in path:
            path = path.replace(",", " ").split(" ")
        elif " " in path:
            path = path.split(" ")
        else:
            raise ValueError("Path is of incorrect format. Details to be added later, hopefully.")

        self.path = []

        path_index = 0
        while path_index < len(path):
            if path[path_index].isalpha():
                kind = path[path_index].upper()
                if kind == "Z":
                    self.path.append(PathSegment(kind, []))
                    path_index += 1
                    continue
                path_index += 1
                points = []
                while (
                    path[path_index].replace(".", "").replace("-", "").replace("+", "").isnumeric()
                ):
                    points.append(Point(float(path[path_index]), float(path[path_index + 1])))
                    path_index = path_index + 2
                    if path_index >= len(path):
                        break
                self.path.append(PathSegment(kind, points))

        self.normalize()

        self.left = 0
        self.right = 1
        self.top = 1
        self.bottom = 0

    def normalize(self):
        """
        Normalize the path such that all x,y points lie between 0 and 1.
        :return:
        """
        max_x = -1e12
        min_x = 1e12
        max_y = -1e12
        min_y = 1e12

        for segment in self.path:
            if segment.kind == "Z":
                break
            for point in segment.coords:
                if point.x > max_x:
                    max_x = point.x
                if point.y > max_y:
                    max_y = point.y
                if point.x < min_x:
                    min_x = point.x
                if point.y < min_y:
                    min_y = point.y

        for segment in self.path:
            if segment.kind == "Z":
                continue
            for point in segment.coords:
                point.x = (point.x - min_x) / (max_x - min_x)
                point.y = (point.y - min_y) / (max_y - min_y)

    def invert(self, axis: str):
        """
        Invert the x or y components of a path, or both.
        :param axis: Which axis to invert: [x, y, both]
        :return:
        """
        for segment in self.path:
            if segment.kind == "Z":
                continue
            for point in segment.coords:
                if axis in ["x" or "both"]:
                    point.x = 1 - point.x
                if axis in ["y" or "both"]:
                    point.y = 1 - point.y

    def reposition(self, left: float, right: float, top: float, bottom: float):
        """
        Reposition/resize the path.
        :param left:
        :param right:
        :param top:
        :param bottom:
        :return:
        """
        x_ratio = (right - left) / (self.right - self.left)
        y_ratio = (top - bottom) / (self.top - self.bottom)
        for segment in self.path:
            if segment.kind == "Z":
                continue
            for point in segment.coords:
                point.x = (point.x - self.left) * x_ratio + left
                point.y = (point.y - self.bottom) * y_ratio + bottom
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top

    def path_string(self):
        """
        Return the path as a string.
        :return:
        """
        string = []
        for segment in self.path:
            string.append(segment.kind)
            if segment.kind == "Z":
                continue
            for point in segment.coords:
                string.append(str(round(point.x, 3)))
                string.append(str(round(point.y, 3)))
        return " ".join(string)


AAs = {
    "A": "M 10.72 14 L 8.92 14 L 7.84 10.56 L 2.9 10.56 L 1.84 14 L 0 14 L 4.62 0 L 6.08 0 L 10.72 14 Z M 4.98 3.9 L 3.38 9.02 L 7.34 9.02 L 5.72 3.9 Q 5.48 3.12 5.34 2.54 Q 5.26 3.02 4.98 3.9 Z",
    "I": "M 0.95 0.0 Q 1.05 0.0565 0.95 0.113 L 0.606 0.113 L 0.606 0.887 L .95 0.887 Q 1.05 0.9435 .95 1.0 L 0.05 1.0 Q -0.05 0.9435 .05 0.887 L 0.394 0.887 L 0.394 0.113 L 0.05 0.113 Q -0.05 0.0565 0.05 0.0 L 0.95 0.0 Z",
    "L": "M 8.38 14 L 0 14 L 0 0 L 1.72 0 L 1.72 12.34 L 8.38 12.34 L 8.38 14 Z",
    "M": "M 1.58 14 L 0 14 L 0 0 L 1.86 0 L 3.96 8.38 Q 4.18 9.26 4.26 9.84 Q 4.32 9.26 4.56 8.38 L 6.68 0 L 8.48 0 L 8.48 14 L 6.9 14 L 6.9 5.22 Q 6.9 4.38 6.94 3.94 Q 6.9 4.16 6.64 5.22 L 4.38 14 L 4.1 14 L 1.84 5.22 Q 1.6 4.16 1.54 3.94 Q 1.58 4.82 1.58 5.22 L 1.58 14 Z",
    "V": "M 6.06 14 L 4.62 14 L 0 0 L 1.86 0 L 5.08 10.38 Q 5.2 10.76 5.34 11.5 Q 5.48 10.76 5.6 10.38 L 8.84 0 L 10.68 0 L 6.06 14 Z",
    "F": "M 1.72 14 L 0 14 L 0 0 L 8.78 0 L 8.78 1.64 L 1.72 1.64 L 1.72 5.94 L 6.66 5.94 L 6.66 7.58 L 1.72 7.58 L 1.72 14 Z",
    "W": "M 3.6 14 L 2.2 14 L 0 0 L 1.64 0 L 2.94 9.24 Q 2.96 9.4 2.99 9.71 Q 3.02 10.02 3.04 10.14 Q 3.1 9.68 3.2 9.24 L 4.8 0 L 6 0 L 7.6 9.14 Q 7.66 9.46 7.74 9.96 Q 7.76 9.56 7.84 9.16 L 9.1 0 L 10.76 0 L 8.56 14 L 7.16 14 L 5.52 4.82 Q 5.5 4.68 5.45 4.37 Q 5.4 4.06 5.38 3.88 Q 5.3 4.52 5.24 4.78 L 3.6 14 Z",
    "Y": "M 4.9 6.78 L 7.92 0 L 9.76 0 L 5.76 8.48 L 5.76 14 L 4.04 14 L 4.04 8.48 L 0 0 L 1.92 0 L 4.9 6.78 Z",
    "N": "M 1.68 14 L 0 14 L 0 0 L 1.7 0 L 6.48 9.82 Q 6.8 10.44 6.84 10.56 Q 6.82 10.18 6.82 9.8 L 6.82 0 L 8.48 0 L 8.48 14 L 6.82 14 L 2.02 4.3 Q 1.88 4.06 1.64 3.54 Q 1.68 3.84 1.68 4.3 L 1.68 14 Z",
    "C": "M 4.54 1.242 L 3.78 1.652 Q 3.56 1.242 3.3 1.022 Q 3.04 0.802 2.63 0.802 Q 2.17 0.802 1.82 1.057 Q 1.47 1.312 1.275 1.732 Q 1.08 2.152 0.985 2.627 Q 0.89 3.102 0.89 3.622 Q 0.89 4.372 1.065 4.987 Q 1.24 5.602 1.645 6.022 Q 2.05 6.442 2.63 6.442 Q 3.43 6.442 3.85 5.552 L 4.62 5.842 Q 4.36 6.462 3.865 6.852 Q 3.37 7.242 2.63 7.242 Q 2.08 7.242 1.625 7.037 Q 1.17 6.832 0.875 6.487 Q 0.58 6.142 0.38 5.672 Q 0.18 5.202 0.09 4.692 Q 0 4.182 0 3.622 Q 0 2.662 0.26 1.867 Q 0.52 1.072 1.13 0.537 Q 1.74 0.002 2.63 0.002 Q 3.96 0.002 4.54 1.242 Z",
    "Q": "M 7.58 13.381 L 8.68 15.041 L 7.46 15.901 L 6.32 14.201 Q 5.6 14.481 4.76 14.481 Q 3.5 14.481 2.55 13.861 Q 1.6 13.241 1.06 12.181 Q 0.52 11.121 0.26 9.881 Q 0 8.641 0 7.241 Q 0 5.841 0.26 4.601 Q 0.52 3.361 1.06 2.301 Q 1.6 1.241 2.55 0.621 Q 3.5 0.001 4.76 0.001 Q 6.02 0.001 6.97 0.621 Q 7.92 1.241 8.47 2.301 Q 9.02 3.361 9.28 4.601 Q 9.54 5.841 9.54 7.241 Q 9.54 11.481 7.58 13.381 Z M 5.38 12.801 L 4.42 11.381 L 5.68 10.521 L 6.58 11.881 Q 7.74 10.361 7.74 7.241 Q 7.74 4.821 7.01 3.211 Q 6.28 1.601 4.76 1.601 Q 3.72 1.601 3.03 2.471 Q 2.34 3.341 2.06 4.561 Q 1.78 5.781 1.78 7.241 Q 1.78 9.661 2.51 11.271 Q 3.24 12.881 4.76 12.881 Q 5.12 12.881 5.38 12.801 Z",
    "S": "M 0 5.692 L 0.8 5.402 Q 1 5.882 1.39 6.167 Q 1.78 6.452 2.27 6.452 Q 2.82 6.452 3.14 6.147 Q 3.46 5.842 3.46 5.282 Q 3.46 4.372 2.05 3.762 Q 1.65 3.592 1.38 3.437 Q 1.11 3.282 0.815 3.037 Q 0.52 2.792 0.365 2.457 Q 0.21 2.122 0.21 1.702 Q 0.21 0.982 0.74 0.492 Q 1.27 0.002 2.15 0.002 Q 2.86 0.002 3.365 0.347 Q 3.87 0.692 4.02 1.242 L 3.23 1.502 Q 3.1 1.192 2.81 0.992 Q 2.52 0.792 2.12 0.792 Q 1.65 0.792 1.37 1.037 Q 1.09 1.282 1.09 1.702 Q 1.09 1.942 1.195 2.132 Q 1.3 2.322 1.53 2.482 Q 1.76 2.642 1.945 2.742 Q 2.13 2.842 2.48 2.992 Q 2.86 3.162 3.135 3.332 Q 3.41 3.502 3.715 3.772 Q 4.02 4.042 4.18 4.417 Q 4.34 4.792 4.34 5.252 Q 4.34 6.182 3.75 6.712 Q 3.16 7.242 2.23 7.242 Q 1.43 7.242 0.825 6.802 Q 0.22 6.362 0 5.692 Z",
    "T": "M 5.74 14 L 4.02 14 L 4.02 1.66 L 0 1.66 L 0 0 L 9.76 0 L 9.76 1.66 L 5.74 1.66 L 5.74 14 Z",
    "D": "M 3.22 14.001 L 0 14.001 L 0 0.001 L 3.3 0.001 Q 4.82 0.001 5.97 0.601 Q 7.12 1.201 7.77 2.221 Q 8.42 3.241 8.73 4.441 Q 9.04 5.641 9.04 7.001 Q 9.04 8.301 8.75 9.471 Q 8.46 10.641 7.83 11.691 Q 7.2 12.741 6.01 13.371 Q 4.82 14.001 3.22 14.001 Z M 1.74 1.581 L 1.74 12.421 L 3.28 12.421 Q 4.38 12.421 5.18 11.951 Q 5.98 11.481 6.41 10.661 Q 6.84 9.841 7.04 8.941 Q 7.24 8.041 7.24 7.001 Q 7.24 4.721 6.31 3.151 Q 5.38 1.581 3.5 1.581 L 1.74 1.581 Z",
    "E": "M 8.62 14 L 0 14 L 0 0 L 8.2 0 L 8.2 1.64 L 1.72 1.64 L 1.72 5.94 L 6.14 5.94 L 6.14 7.58 L 1.72 7.58 L 1.72 12.36 L 8.62 12.36 L 8.62 14 Z",
    "R": "M 5.64 7.88 L 8.64 14.02 L 6.74 14.02 L 3.82 8.02 L 1.72 8.02 L 1.72 14.02 L 0 14.02 L 0 0.02 L 4.3 0.02 Q 6.64 0.02 7.67 1.15 Q 8.7 2.28 8.7 4 Q 8.7 5.36 7.95 6.45 Q 7.2 7.54 5.64 7.88 Z M 4.34 1.6 L 1.72 1.6 L 1.72 6.44 L 4.34 6.44 Q 5.74 6.44 6.37 5.75 Q 7 5.06 7 4.04 Q 7 3.08 6.38 2.34 Q 5.76 1.6 4.34 1.6 Z",
    "H": "M 1.72 14 L 0 14 L 0 0 L 1.72 0 L 1.72 5.92 L 6.76 5.92 L 6.76 0 L 8.48 0 L 8.48 14 L 6.76 14 L 6.76 7.56 L 1.72 7.56 L 1.72 14 Z",
    "K": "M 9.32 14 L 7.34 14 L 3.32 6.38 L 1.72 8.4 L 1.72 14 L 0 14 L 0 0 L 1.72 0 L 1.72 6.02 L 6.4 0 L 8.38 0 L 4.5 4.9 L 9.32 14 Z",
    "G": "M 8.925 2.446 L 7.405 3.326 Q 6.965 2.526 6.425 2.066 Q 5.885 1.606 5.045 1.606 Q 3.905 1.606 3.145 2.506 Q 2.385 3.406 2.095 4.616 Q 1.805 5.826 1.805 7.246 Q 1.805 9.626 2.625 11.256 Q 3.445 12.886 5.045 12.886 Q 6.225 12.886 6.905 12.016 Q 7.585 11.146 7.585 9.706 L 7.585 9.046 L 5.065 9.046 L 5.065 7.466 L 9.385 7.466 L 9.385 9.146 Q 9.385 11.666 8.165 13.076 Q 6.945 14.486 5.045 14.486 Q 3.345 14.486 2.175 13.416 Q 1.005 12.346 0.505 10.756 Q 0.005 9.166 0.005 7.246 Q 0.005 4.226 1.305 2.116 Q 2.605 0.006 5.045 0.006 Q 6.565 0.006 7.445 0.666 Q 8.325 1.326 8.925 2.446 Z",
    "P": "M 1.72 14.001 L 0 14.001 L 0 0.001 L 4.32 0.001 Q 5.54 0.001 6.45 0.361 Q 7.36 0.721 7.86 1.331 Q 8.36 1.941 8.6 2.661 Q 8.84 3.381 8.84 4.201 Q 8.84 5.861 7.71 7.121 Q 6.58 8.381 4.32 8.381 L 1.72 8.381 L 1.72 14.001 Z M 1.72 1.581 L 1.72 6.781 L 4.44 6.781 Q 5.82 6.781 6.48 6.021 Q 7.14 5.261 7.14 4.201 Q 7.14 3.161 6.51 2.371 Q 5.88 1.581 4.44 1.581 L 1.72 1.581 Z",
    "X": "M 42.432 70.02 L 35.645 70.02 L 21.24 40.918 L 6.836 70.02 L 0 70.02 L 17.822 34.033 L 1.025 0 L 7.813 0 L 21.24 27.1 L 34.619 0 L 41.406 0 L 24.609 34.033 L 42.432 70.02 Z",
}


# adapted from https://stackoverflow.com/a/67912302
def get_color(colorscale_name, loc):
    from _plotly_utils.basevalidators import ColorscaleValidator

    # first parameter: Name of the property being validated
    # second parameter: a string, doesn't really matter in our use case
    cv = ColorscaleValidator("colorscale", "")
    # colorscale will be a list of lists: [[loc1, "rgb1"], [loc2, "rgb2"], ...]
    colorscale = cv.validate_coerce(colorscale_name)

    if hasattr(loc, "__iter__"):
        return [get_continuous_color(colorscale, x) for x in loc]
    return get_continuous_color(colorscale, loc)


def get_continuous_color(colorscale, intermed):
    """
    Plotly continuous colorscales assign colors to the range [0, 1]. This function computes the intermediate
    color for any value in that range.

    Plotly doesn't make the colorscales directly accessible in a common format.
    Some are ready to use:

        colorscale = plotly.colors.PLOTLY_SCALES["Greens"]

    Others are just swatches that need to be constructed into a colorscale:

        viridis_colors, scale = plotly.colors.convert_colors_to_same_type(plotly.colors.sequential.Viridis)
        colorscale = plotly.colors.make_colorscale(viridis_colors, scale=scale)

    :param colorscale: A plotly continuous colorscale defined with RGB string colors.
    :param intermed: value in the range [0, 1]
    :return: color in rgb string format
    :rtype: str
    """
    if len(colorscale) < 1:
        raise ValueError("colorscale must have at least one color")

    hex_to_rgb = lambda c: "rgb" + str(ImageColor.getcolor(c, "RGB"))

    if intermed <= 0 or len(colorscale) == 1:
        c = colorscale[0][1]
        return c if c[0] != "#" else hex_to_rgb(c)
    if intermed >= 1:
        c = colorscale[-1][1]
        return c if c[0] != "#" else hex_to_rgb(c)

    for cutoff, color in colorscale:
        if intermed > cutoff:
            low_cutoff, low_color = cutoff, color
        else:
            high_cutoff, high_color = cutoff, color
            break

    if (low_color[0] == "#") or (high_color[0] == "#"):
        # some color scale names (such as cividis) returns:
        # [[loc1, "hex1"], [loc2, "hex2"], ...]
        low_color = hex_to_rgb(low_color)
        high_color = hex_to_rgb(high_color)

    return plotly.colors.find_intermediate_color(
        lowcolor=low_color,
        highcolor=high_color,
        intermed=((intermed - low_cutoff) / (high_cutoff - low_cutoff)),
        colortype="rgb",
    )


def add_logo(
    fig: go.Figure,
    matrix: np.ndarray,
    colors: np.ndarray,
    row: int = 1,
    col: int = 1,
    colorscale: str = "Plasma",
    alphabet: Optional[List[str]] = None,
    threshold: float = 0.1,
):
    maximum = np.max(np.sum(matrix, axis=1))
    sequence_length = matrix.shape[0]
    fig.add_trace(
        go.Scatter(
            x=[0, sequence_length],
            y=[0, maximum],
            visible=False,
        ),
        row=row,
        col=col,
    )
    x_name = next(fig.select_xaxes(row=row, col=col)).plotly_name.replace("axis", "")
    y_name = next(fig.select_yaxes(row=row, col=col)).plotly_name.replace("axis", "")

    if alphabet is None:
        alphabet = [data.index_to_acid(i) for i in range(0, data.num_acids() + 1)]

    shapes = []
    for i, column in enumerate(matrix):
        height = 0
        for j, weight in enumerate(column):
            if weight > threshold:
                path = Path(AAs[alphabet[j]])
                path.invert("y")
                path.reposition(i, i + 1, height + weight, height)
                height += weight
                color = colors[i][j]
                shapes.append(
                    dict(
                        type="path",
                        path=path.path_string(),
                        xref=x_name,
                        yref=y_name,
                        ysizemode="scaled",
                        fillcolor=color,
                        line_color=color,
                        line_width=1,
                    )
                )
    fig.layout.shapes = shapes


def make_shannon_matrix(alignment: np.ndarray, reverse: bool = False):
    # https://en.wikipedia.org/wiki/Sequence_logo
    frequencies = np.count_nonzero(alignment[:, :, None] == np.arange(21)[None, None, :], axis=0)
    mask = np.sum(frequencies, axis=1) > 0
    frequencies = frequencies[mask]
    frequencies = frequencies / np.sum(frequencies, axis=1, keepdims=True)
    h = entropy(frequencies, base=2, axis=1)
    e = (21 - 1) / (np.log(2) * 2 * alignment.shape[0])
    if not reverse:
        r = np.log2(21) - (h + e)
    else:
        r = h + e
    matrix = frequencies * r[:, None]
    full_matrix = np.zeros((alignment.shape[1], 21))
    full_matrix[mask] = matrix
    return full_matrix


def make_colors_from_linear(
    linear: np.ndarray, size: int = 21, colorscale: str = "Plasma"
) -> np.ndarray:
    linear = linear / np.max(linear)
    colors = np.zeros((len(linear), size), dtype=np.object)
    for i, x in enumerate(linear):
        colors[i, :] = get_color(colorscale, x)
    return colors


def make_colors_from_matrix(matrix: np.ndarray, colorscale: str = "Plasma") -> np.ndarray:
    matrix = matrix / np.max(matrix)
    colors = np.zeros(matrix.shape, dtype=np.object)
    for i, xs in enumerate(matrix):
        for j, x in enumerate(xs):
            colors[i, j] = get_color(colorscale, x)
    return colors
