#!/usr/bin/env python3
"""Bake the console's fallback coastline into a compact GeoJSON.

The console draws on Leaflet, so projection is Leaflet's job and this script has
none: it emits latitude and longitude, simplified, and the map decides where
that lands on screen. That is the whole reason to do it this way. An earlier
version of this map projected here and projected again, separately, in the
page's JavaScript, and the two drifted -- the continents ended up drawn in a
projection that does not exist while every marker still sat in exactly the right
place relative to them, which is the hardest kind of wrong to notice.

What this produces is only the fallback. The console's basemap is tiles; this is
what it draws instead when they fail to arrive, so a bad network costs detail
rather than the entire map.

    scripts/build_worldmap.py [source.geojson] > x/cluster/console/land.geojson

With no argument it fetches Natural Earth 110m land, which is public domain.
"""

import json
import math
import sys
import urllib.request

SOURCE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_land.geojson"

# Simplification tolerance in degrees, and the smallest ring worth keeping.
# At the zoom levels this fallback is ever seen at, a quarter of a degree is
# well under a pixel, and dropping specks costs nothing visible.
TOLERANCE = 0.22
MIN_AREA = 1.4

# Antarctica spans the whole bottom edge and an operations map gains nothing
# from it.
LAT_BOTTOM = -60.0


def perpendicular_distance(p, a, b):
    (px, py), (ax, ay), (bx, by) = p, a, b
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def simplify(points, tolerance):
    """Douglas-Peucker, iteratively so a long coastline cannot blow the stack."""
    if len(points) < 3:
        return points

    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]

    while stack:
        first, last = stack.pop()
        if last <= first + 1:
            continue
        worst, index = tolerance, -1
        for i in range(first + 1, last):
            d = perpendicular_distance(points[i], points[first], points[last])
            if d > worst:
                worst, index = d, i
        if index != -1:
            keep[index] = True
            stack.append((first, index))
            stack.append((index, last))

    return [p for p, k in zip(points, keep) if k]


def area(points):
    """Twice the enclosed area, unsigned. Used only to drop specks."""
    total = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def rings(geometry):
    """Yield every linear ring, whatever the geometry type."""
    kind, coords = geometry["type"], geometry["coordinates"]
    if kind == "Polygon":
        yield from coords
    elif kind == "MultiPolygon":
        for polygon in coords:
            yield from polygon


def clip_south(ring):
    """Keep the runs of the ring above the southern bound."""
    runs, current = [], []
    for lon, lat in ring:
        if lat >= LAT_BOTTOM:
            current.append((lon, lat))
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return [r for r in runs if len(r) >= 3]


def main():
    if len(sys.argv) > 1:
        data = json.load(open(sys.argv[1]))
    else:
        with urllib.request.urlopen(SOURCE, timeout=60) as response:
            data = json.load(response)

    polygons = []
    for feature in data["features"]:
        for ring in rings(feature["geometry"]):
            for run in clip_south(ring):
                points = simplify(run, TOLERANCE)
                if len(points) < 3 or area(points) < MIN_AREA:
                    continue
                polygons.append([[[round(lon, 2), round(lat, 2)] for lon, lat in points]])

    out = {
        "type": "Feature",
        "properties": {"source": "Natural Earth 110m land, public domain"},
        "geometry": {"type": "MultiPolygon", "coordinates": polygons},
    }

    sys.stderr.write(f"rings {len(polygons)}  tolerance {TOLERANCE} deg\n")
    json.dump(out, sys.stdout, separators=(",", ":"))


if __name__ == "__main__":
    main()
