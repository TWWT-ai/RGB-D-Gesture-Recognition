"""
Dump the CODE of a notebook (no outputs, no images) to a small text file.

Why: yolo-gesture.ipynb is ~25 MB because every plot and log line is stored
inside it. That is far too big to move around. This writes a few-KB file with
just the cell sources, which is all anyone needs in order to review or patch
the notebook.

Reads nothing but the notebook, writes nothing but the dump. Safe to re-run.

    cd "D:\\School\\Personal\\UCL\\COMP0248\\CW1\\Object Detection"
    python tools/nb_dump.py yolo-gesture.ipynb
"""
import json
import sys
import pathlib


def main():
    nb_path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "yolo-gesture.ipynb")
    out_path = pathlib.Path("tools") / (nb_path.stem + ".cells.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])

    lines = [f"# notebook : {nb_path.name}",
             f"# cells    : {len(cells)}",
             f"# nbformat : {nb.get('nbformat')}.{nb.get('nbformat_minor')}",
             ""]
    for i, cell in enumerate(cells):
        src = "".join(cell.get("source", []))
        n_out = len(cell.get("outputs", []))
        lines.append(f"===== CELL {i:03d} [{cell.get('cell_type')}] "
                     f"({n_out} output block(s)) =====")
        lines.append(src.rstrip("\n"))
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    size_mb = nb_path.stat().st_size / 1e6
    print(f"{nb_path.name} is {size_mb:.1f} MB, {len(cells)} cells")
    print(f"wrote {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
