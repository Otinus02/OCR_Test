"""
GLM-OCR Client - PDF/Image OCR via vLLM Server
Usage:
    python main.py --server http://<GPU_SERVER_IP>:8080 --input document.pdf
    python main.py --server http://<GPU_SERVER_IP>:8080 --input image.png --mode table
"""

import argparse
import json
import sys
from pathlib import Path

import requests
from PIL import Image

from ocr_utils import pdf_to_images, image_to_base64


def ocr_request(server_url: str, image_b64: str, mode: str = "text",
                max_tokens: int = 8192) -> str:
    """vLLM 서버에 OCR 요청 (OpenAI-compatible API)"""
    prompts = {
        "text": "Text Recognition:",
        "table": "Table Recognition:",
        "formula": "Formula Recognition:",
    }
    prompt_text = prompts.get(mode, mode)  # 커스텀 프롬프트도 지원

    payload = {
        "model": "zai-org/GLM-OCR",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt_text
                    }
                ]
            }
        ],
        "max_tokens": max_tokens,
    }

    api_url = f"{server_url.rstrip('/')}/v1/chat/completions"
    resp = requests.post(api_url, json=payload, timeout=300)
    resp.raise_for_status()

    data = resp.json()
    choices = data.get("choices", [])
    if not choices:
        raise ValueError(f"서버 응답에 choices가 없습니다: {data}")
    return choices[0]["message"]["content"]


def process_file(server_url: str, input_path: str, mode: str,
                 output_path: str | None, dpi: int) -> None:
    """파일(PDF/이미지) OCR 처리"""
    path = Path(input_path)

    if not path.exists():
        print(f"파일을 찾을 수 없습니다: {input_path}")
        sys.exit(1)

    # PDF인 경우 이미지로 변환
    if path.suffix.lower() == ".pdf":
        print(f"PDF 변환 중: {path.name} (DPI: {dpi})")
        images = pdf_to_images(input_path, dpi=dpi)
    else:
        # 이미지 파일 직접 로드
        print(f"이미지 로드: {path.name}")
        images = [Image.open(input_path).convert("RGB")]

    # 각 페이지/이미지에 대해 OCR 수행
    all_results = []
    total = len(images)

    for i, img in enumerate(images):
        print(f"\nOCR 처리 중 [{i + 1}/{total}] (모드: {mode})...")
        b64 = image_to_base64(img)

        try:
            text = ocr_request(server_url, b64, mode=mode)
            all_results.append(text)
            # 미리보기 출력 (처음 200자)
            preview = text[:200] + "..." if len(text) > 200 else text
            print(f"  결과 미리보기: {preview}")
        except requests.exceptions.ConnectionError:
            print(f"  서버 연결 실패: {server_url}")
            print("  서버가 실행 중인지 확인하세요.")
            sys.exit(1)
        except requests.exceptions.HTTPError as e:
            print(f"  서버 오류: {e}")
            print(f"  응답: {e.response.text[:500]}")
            all_results.append(f"[ERROR] {e}")
        except json.JSONDecodeError as e:
            print(f"  JSON 파싱 오류: {e}")
            all_results.append(f"[ERROR] JSON 파싱 실패: {e}")
        except requests.exceptions.RequestException as e:
            print(f"  요청 오류: {e}")
            all_results.append(f"[ERROR] {e}")

    # 결과 합치기
    separator = f"\n\n{'=' * 60}\n\n"
    full_result = separator.join(
        f"[페이지 {i + 1}]\n{text}" if total > 1 else text
        for i, text in enumerate(all_results)
    )

    # 출력
    if output_path:
        out = Path(output_path)
        out.write_text(full_result, encoding="utf-8")
        print(f"\n결과 저장됨: {out.absolute()}")
    else:
        print(f"\n{'=' * 60}")
        print("OCR 결과:")
        print(f"{'=' * 60}")
        print(full_result)

    print(f"\n완료! 총 {total}개 페이지 처리됨.")


def main():
    parser = argparse.ArgumentParser(
        description="GLM-OCR Client - PDF/Image OCR via vLLM Server"
    )
    parser.add_argument(
        "--server", "-s",
        required=True,
        help="vLLM 서버 URL (예: http://123.456.789.0:8080)"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="입력 파일 경로 (PDF 또는 이미지)"
    )
    parser.add_argument(
        "--mode", "-m",
        default="text",
        choices=["text", "table", "formula"],
        help="OCR 모드 (기본: text)"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="결과 저장 파일 경로 (미지정 시 콘솔 출력)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="PDF 렌더링 DPI (기본: 200)"
    )

    args = parser.parse_args()
    process_file(args.server, args.input, args.mode, args.output, args.dpi)


if __name__ == "__main__":
    main()
