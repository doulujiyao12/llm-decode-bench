"""Dependency-free SVG summary. Invalid runs are listed, never plotted as zero."""
import html
import json
from pathlib import Path


def plot(paths, output):
    points, invalid = [], []
    for raw in paths:
        path = Path(raw)
        cfg = json.loads((path / "config.json").read_text())
        result = json.loads((path / "analysis.json").read_text())
        if result["status"] == "valid":
            points.append((path.name, cfg["batch_size"], result["sse_gap_p50_ms"], result["batch_verification"]))
        else:
            invalid.append(path.name)
    if not points:
        raise ValueError("No valid runs to plot")
    width, height = 1000, 180 + 56 * len(points)
    maximum = max(p[2] for p in points) or 1
    chunks = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
              '<rect width="100%" height="100%" fill="white"/>',
              '<g font-family="sans-serif" font-size="14" fill="#17233a">',
              '<text x="24" y="30" font-size="22">Steady SSE gap (ms / progress event)</text>',
              '<text x="24" y="56">Not per-token TPOT. No MTP acceptance-length division.</text>']
    for i, (name, bs, value, verification) in enumerate(points):
        y = 92 + i * 56
        label = html.escape(f"{name} | target BS {bs}")
        chunks.append(f'<text x="24" y="{y}">{label}</text>')
        chunks.append(f'<rect x="400" y="{y-16}" width="{value/maximum*430:.2f}" height="22" fill="#3378bb"/>')
        chunks.append(f'<text x="850" y="{y}">{value:.3f} ms</text>')
        chunks.append(f'<text x="400" y="{y+22}" fill="#65738a">{html.escape(verification)}</text>')
    chunks.append(f'<text x="24" y="{height-30}">Invalid runs excluded: {html.escape(", ".join(invalid) or "none")}</text></g></svg>')
    path = Path(output)
    with path.open("x") as f:
        f.write("\n".join(chunks))
    return str(path)
