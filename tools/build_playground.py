"""Build the browser playground: inlines the wasm module (base64) into the page.

    cargo build --release --target wasm32-unknown-unknown -p swadesh-wasm
    python tools/build_playground.py            # -> docs/playground/index.html
    python tools/build_playground.py --fragment OUT.html   # body-only fragment
"""
import base64
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
wasm = ROOT / "target/wasm32-unknown-unknown/release/swadesh_wasm.wasm"
tpl = (ROOT / "docs/playground/template.html").read_text(encoding="utf-8")
page = tpl.replace("__WASM_B64__", base64.b64encode(wasm.read_bytes()).decode("ascii"))
if len(sys.argv) > 2 and sys.argv[1] == "--fragment":
    Path(sys.argv[2]).write_text(page, encoding="utf-8")
    print("wrote", sys.argv[2])
else:
    full = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
            "</head>\n<body>\n" + page + "\n</body>\n</html>\n")
    out = ROOT / "docs/playground/index.html"
    out.write_text(full, encoding="utf-8")
    print("wrote", out, f"{len(full) // 1024} KB")
