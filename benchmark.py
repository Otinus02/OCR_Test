"""
OCR Benchmark Tool - GLM-OCR vs pymupdf4llm vs markitdown
Usage:
    python benchmark.py --pdf test_pdfs/sample.pdf --server http://localhost:8080
    python benchmark.py --pdf test_pdfs/sample.pdf --methods glm_ocr,pymupdf4llm
    python benchmark.py --pdf test_pdfs/sample.pdf --pages 0,1,2 --glm-mode table
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pymupdf
import requests
from PIL import Image

from ocr_utils import image_to_base64, get_pdf_page_count


# ── Method implementations ──────────────────────────────────────────────────

class GlmOcrMethod:
    name = "glm_ocr"

    def __init__(self, server_url: str, mode: str = "text", dpi: int = 200):
        self.server_url = server_url.rstrip("/")
        self.mode = mode
        self.dpi = dpi
        self.prompts = {
            "text": "Text Recognition:",
            "table": "Table Recognition:",
            "formula": "Formula Recognition:",
        }

    def is_available(self) -> tuple[bool, str]:
        try:
            resp = requests.get(f"{self.server_url}/v1/models", timeout=10)
            resp.raise_for_status()
            return True, "OK"
        except Exception as e:
            return False, f"vLLM server not reachable: {e}"

    def run_page(self, pdf_path: str, page_idx: int) -> str:
        """Run OCR on a single page"""
        zoom = self.dpi / 72
        matrix = pymupdf.Matrix(zoom, zoom)

        with pymupdf.open(pdf_path) as doc:
            page = doc[page_idx]
            pix = page.get_pixmap(matrix=matrix)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        b64 = image_to_base64(img)
        prompt_text = self.prompts.get(self.mode, self.mode)

        payload = {
            "model": "zai-org/GLM-OCR",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64}"
                            },
                        },
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ],
            "max_tokens": 8192,
        }

        resp = requests.post(
            f"{self.server_url}/v1/chat/completions",
            json=payload,
            timeout=300,
        )
        resp.raise_for_status()

        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            raise ValueError(f"서버 응답에 choices가 없습니다: {data}")
        return choices[0]["message"]["content"]

    def supports_per_page(self) -> bool:
        return True


class Pymupdf4llmMethod:
    name = "pymupdf4llm"

    def __init__(self):
        self._mod = None

    def is_available(self) -> tuple[bool, str]:
        try:
            import pymupdf4llm as mod
            self._mod = mod
            return True, "OK"
        except ImportError:
            return False, "pymupdf4llm not installed (pip install pymupdf4llm)"

    def run_page(self, pdf_path: str, page_idx: int) -> str:
        chunks = self._mod.to_markdown(pdf_path, page_chunks=True, pages=[page_idx])
        if chunks:
            return chunks[0].get("text", "")
        return ""

    def supports_per_page(self) -> bool:
        return True


class MarkitdownMethod:
    name = "markitdown"

    def __init__(self):
        self._cls = None

    def is_available(self) -> tuple[bool, str]:
        try:
            from markitdown import MarkItDown
            self._cls = MarkItDown
            return True, "OK"
        except ImportError:
            return False, "markitdown not installed (pip install markitdown)"

    def run_whole(self, pdf_path: str) -> str:
        converter = self._cls()
        result = converter.convert(pdf_path)
        return result.text_content

    def supports_per_page(self) -> bool:
        return False


# ── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(text: str) -> dict:
    chars = len(text)
    words = len(text.split())
    lines = text.count("\n") + (1 if text else 0)
    empty = chars == 0
    return {"chars": chars, "words": words, "lines": lines, "empty": empty}


# ── Benchmark runner ────────────────────────────────────────────────────────

def run_benchmark(
    pdf_path: str,
    methods: list,
    pages: list[int] | None,
    total_pages: int,
) -> dict:
    """Run all methods and return structured results."""
    if pages is None:
        pages = list(range(total_pages))

    results = {}

    for method in methods:
        method_name = method.name
        print(f"\n{'─' * 50}")
        print(f"  Running: {method_name}")
        print(f"{'─' * 50}")

        ok, msg = method.is_available()
        if not ok:
            print(f"  SKIPPED: {msg}")
            results[method_name] = {
                "status": "skipped",
                "reason": msg,
                "pages": {},
                "total_time": 0,
            }
            continue

        page_results = {}

        if method.supports_per_page():
            total_time = 0.0
            for pg in pages:
                print(f"  Page {pg}...", end=" ", flush=True)
                t0 = time.perf_counter()
                try:
                    text = method.run_page(pdf_path, pg)
                except Exception as e:
                    text = ""
                    print(f"ERROR: {e}")
                    page_results[pg] = {
                        "text": "",
                        "time": 0,
                        "metrics": compute_metrics(""),
                        "error": str(e),
                    }
                    continue
                elapsed = time.perf_counter() - t0
                total_time += elapsed
                metrics = compute_metrics(text)
                page_results[pg] = {
                    "text": text,
                    "time": elapsed,
                    "metrics": metrics,
                }
                status = "EMPTY" if metrics["empty"] else f"{metrics['chars']} chars"
                print(f"{elapsed:.2f}s  ({status})")

            results[method_name] = {
                "status": "ok",
                "pages": page_results,
                "total_time": total_time,
            }
        else:
            # Whole-document method (markitdown)
            print(f"  Processing entire document...", end=" ", flush=True)
            t0 = time.perf_counter()
            try:
                full_text = method.run_whole(pdf_path)
            except Exception as e:
                print(f"ERROR: {e}")
                results[method_name] = {
                    "status": "error",
                    "reason": str(e),
                    "pages": {},
                    "total_time": 0,
                }
                continue
            elapsed = time.perf_counter() - t0
            print(f"{elapsed:.2f}s")

            # Split by form-feed into pages
            raw_pages = full_text.split("\f")
            for pg in pages:
                if pg < len(raw_pages):
                    text = raw_pages[pg].strip()
                else:
                    text = ""
                metrics = compute_metrics(text)
                page_results[pg] = {
                    "text": text,
                    "time": None,  # per-page timing not available
                    "metrics": metrics,
                }

            results[method_name] = {
                "status": "ok",
                "pages": page_results,
                "total_time": elapsed,
            }

    return results


# ── Report generation ───────────────────────────────────────────────────────

def build_summary_table(results: dict, pages: list[int]) -> str:
    header = (
        f"{'Method':<16}| {'Total Time':>10} | {'Avg/Page':>8} | "
        f"{'Chars':>7} | {'Words':>6} | {'Empty Pages'}"
    )
    sep = "─" * len(header)
    lines = [header, sep]

    for method_name, data in results.items():
        if data["status"] == "skipped":
            lines.append(f"{method_name:<16}|  {'SKIPPED':>9} | {'':>8} | {'':>7} | {'':>6} | {data['reason']}")
            continue
        if data["status"] == "error":
            lines.append(f"{method_name:<16}|  {'ERROR':>9} | {'':>8} | {'':>7} | {'':>6} | {data['reason']}")
            continue

        total_time = data["total_time"]
        page_data = data["pages"]
        total_chars = sum(p["metrics"]["chars"] for p in page_data.values())
        total_words = sum(p["metrics"]["words"] for p in page_data.values())
        empty_count = sum(1 for p in page_data.values() if p["metrics"]["empty"])
        num_pages = len(page_data)

        avg_str = f"{total_time / num_pages:.2f}s" if num_pages > 0 else "N/A"
        # markitdown doesn't support per-page timing
        has_per_page = any(p.get("time") is not None for p in page_data.values())
        if not has_per_page:
            avg_str = "N/A"

        lines.append(
            f"{method_name:<16}| {total_time:>9.2f}s | {avg_str:>8} | "
            f"{total_chars:>7,} | {total_words:>6,} | "
            f"{empty_count}/{num_pages}"
        )

    return "\n".join(lines)


def save_results(results: dict, pages: list[int], pdf_path: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Per-method page outputs
    for method_name, data in results.items():
        if data["status"] != "ok":
            continue
        method_dir = output_dir / method_name
        method_dir.mkdir(exist_ok=True)
        for pg, pg_data in data["pages"].items():
            out_file = method_dir / f"page_{int(pg):03d}.md"
            out_file.write_text(pg_data["text"], encoding="utf-8")

    # 2. Side-by-side comparison
    side_dir = output_dir / "side_by_side"
    side_dir.mkdir(exist_ok=True)
    method_names = [m for m, d in results.items() if d["status"] == "ok"]

    for pg in pages:
        parts = []
        for method_name in method_names:
            pg_data = results[method_name]["pages"].get(pg)
            text = pg_data["text"] if pg_data else "(no output)"
            parts.append(f"{'=' * 60}\n  {method_name}  |  Page {pg}\n{'=' * 60}\n{text}")
        side_file = side_dir / f"page_{pg:03d}.txt"
        side_file.write_text("\n\n".join(parts), encoding="utf-8")

    # 3. Summary table
    summary = build_summary_table(results, pages)
    summary_header = (
        f"OCR Benchmark Summary\n"
        f"PDF: {pdf_path}\n"
        f"Pages: {pages}\n"
        f"Date: {datetime.now().isoformat()}\n\n"
    )
    (output_dir / "summary.txt").write_text(summary_header + summary, encoding="utf-8")
    print(f"\n{summary}")

    # 4. report.json (without full text to keep it small)
    report = {
        "pdf": pdf_path,
        "pages": pages,
        "timestamp": datetime.now().isoformat(),
        "methods": {},
    }
    for method_name, data in results.items():
        method_report = {
            "status": data["status"],
            "total_time": data["total_time"],
        }
        if data["status"] == "ok":
            method_report["page_metrics"] = {
                str(pg): {
                    "time": pg_data["time"],
                    "chars": pg_data["metrics"]["chars"],
                    "words": pg_data["metrics"]["words"],
                    "lines": pg_data["metrics"]["lines"],
                    "empty": pg_data["metrics"]["empty"],
                }
                for pg, pg_data in data["pages"].items()
            }
            total_chars = sum(p["metrics"]["chars"] for p in data["pages"].values())
            total_words = sum(p["metrics"]["words"] for p in data["pages"].values())
            empty_count = sum(1 for p in data["pages"].values() if p["metrics"]["empty"])
            method_report["totals"] = {
                "chars": total_chars,
                "words": total_words,
                "empty_pages": empty_count,
                "total_pages": len(data["pages"]),
            }
        elif "reason" in data:
            method_report["reason"] = data["reason"]
        report["methods"][method_name] = method_report

    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nResults saved to: {output_dir}")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="OCR Benchmark - GLM-OCR vs pymupdf4llm vs markitdown"
    )
    parser.add_argument(
        "--pdf", required=True, help="Input PDF file path"
    )
    parser.add_argument(
        "--server", default="http://localhost:8080",
        help="vLLM server URL for GLM-OCR (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--methods", default="glm_ocr,pymupdf4llm,markitdown",
        help="Comma-separated methods to benchmark (default: all three)",
    )
    parser.add_argument(
        "--pages", default=None,
        help="Comma-separated page indices to test (default: all pages)",
    )
    parser.add_argument(
        "--glm-mode", default="text", choices=["text", "table", "formula"],
        help="GLM-OCR recognition mode (default: text)",
    )
    parser.add_argument(
        "--dpi", type=int, default=200,
        help="DPI for PDF-to-image conversion in GLM-OCR (default: 200)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: results/<timestamp>)",
    )

    args = parser.parse_args()

    # Validate PDF
    pdf_path = args.pdf
    if not Path(pdf_path).exists():
        print(f"ERROR: PDF not found: {pdf_path}")
        sys.exit(1)

    total_pages = get_pdf_page_count(pdf_path)
    print(f"PDF: {pdf_path} ({total_pages} pages)")

    # Parse pages
    pages = None
    if args.pages:
        pages = [int(p.strip()) for p in args.pages.split(",")]
        for p in pages:
            if p < 0 or p >= total_pages:
                print(f"ERROR: Page {p} out of range (0-{total_pages - 1})")
                sys.exit(1)

    if pages is None:
        pages = list(range(total_pages))

    # Build method list
    requested = [m.strip() for m in args.methods.split(",")]
    method_map = {
        "glm_ocr": lambda: GlmOcrMethod(args.server, mode=args.glm_mode, dpi=args.dpi),
        "pymupdf4llm": lambda: Pymupdf4llmMethod(),
        "markitdown": lambda: MarkitdownMethod(),
    }
    methods = []
    for name in requested:
        if name not in method_map:
            print(f"WARNING: Unknown method '{name}', skipping")
            continue
        methods.append(method_map[name]())

    if not methods:
        print("ERROR: No valid methods specified")
        sys.exit(1)

    print(f"Methods: {', '.join(m.name for m in methods)}")
    print(f"Pages: {pages}")

    # Run benchmark
    results = run_benchmark(pdf_path, methods, pages, total_pages)

    # Save results
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results") / timestamp

    save_results(results, pages, pdf_path, output_dir)


if __name__ == "__main__":
    main()
