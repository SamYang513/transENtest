
import argparse
import json
import time
from pathlib import Path

from asr_translate import (
    load_asr_model,
    transcribe_audio,
    merge_asr_segments,
    OUTPUT_DIR
)


def process_asr(audio_path, device="cpu", merge=True):
    audio_path = Path(audio_path).resolve()

    if not audio_path.is_file():
        raise FileNotFoundError(f"找不到音檔：{audio_path}")

    start_time = time.perf_counter()

    print("========== ASR Only ==========")
    print(f"音檔：{audio_path.name}")

    print("\n[1/2] 載入 Whisper 模型...")
    asr_model = load_asr_model(device)

    print("[2/2] 開始英文語音辨識...")
    raw_segments, info = transcribe_audio(
        audio_path,
        asr_model
    )

    if merge:
        segments = merge_asr_segments(raw_segments)
    else:
        segments = [
            {
                **item,
                "source_segment_ids": [item["id"]]
            }
            for item in raw_segments
        ]

    for index, item in enumerate(segments, start=1):
        item["id"] = index

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    name = audio_path.stem

    raw_path = OUTPUT_DIR / f"{name}_raw_en.txt"
    english_path = OUTPUT_DIR / f"{name}_en.txt"
    json_path = OUTPUT_DIR / f"{name}_asr.json"

    with open(raw_path, "w", encoding="utf-8") as f:
        for item in raw_segments:
            f.write(
                f"[{item['start']:.2f} - {item['end']:.2f}] "
                f"{item['english_text']}\n"
            )

    with open(english_path, "w", encoding="utf-8") as f:
        for item in segments:
            f.write(
                f"[{item['start']:.2f} - {item['end']:.2f}] "
                f"{item['english_text']}\n"
            )

    elapsed = time.perf_counter() - start_time

    result = {
        "audio_file": audio_path.name,
        "language": "en",
        "device": device,
        "asr_model": "faster-whisper-medium",
        "raw_segment_count": len(raw_segments),
        "segment_count": len(segments),
        "processing_seconds": round(elapsed, 2),
        "segments": segments
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n========== ASR 完成 ==========")
    print(f"原始英文：{raw_path}")
    print(f"合併英文：{english_path}")
    print(f"ASR JSON：{json_path}")
    print(f"處理時間：{elapsed:.2f} 秒")


def main():
    parser = argparse.ArgumentParser(
        description="音檔轉英文逐字稿"
    )

    parser.add_argument("audio", type=str)

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu"
    )

    parser.add_argument(
        "--no-merge",
        action="store_true"
    )

    args = parser.parse_args()

    process_asr(
        args.audio,
        device=args.device,
        merge=not args.no_merge
    )


if __name__ == "__main__":
    main()
