
import argparse
import json
import re
import time
from pathlib import Path

from asr_translate import (
    load_translation_model,
    translate_text,
    OUTPUT_DIR,
    TRANSLATE_NUM_BEAMS,
    TRANSLATE_LENGTH_PENALTY,
    TRANSLATE_REPETITION_PENALTY,
    TRANSLATE_MAX_NEW_TOKENS
)


def read_transcript(path):
    path = Path(path).resolve()

    if not path.is_file():
        raise FileNotFoundError(f"找不到逐字稿：{path}")

    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError("JSON 最外層必須是物件")

        segments = data.get("segments")

        if not isinstance(segments, list):
            raise ValueError("JSON 缺少 segments 清單")

        for item in segments:
            if not isinstance(item, dict):
                raise ValueError("segments 內容必須是物件")
            if "english_text" not in item:
                raise ValueError("片段缺少 english_text")

        return segments, data.get("audio_file")

    if path.suffix.lower() != ".txt":
        raise ValueError("只支援 .json 或 .txt")

    segments = []

    pattern = re.compile(
        r"^\[(\d+(?:\.\d+)?)\s*-\s*"
        r"(\d+(?:\.\d+)?)\]\s*(.*)$"
    )

    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            match = pattern.match(line)

            if match:
                start = float(match.group(1))
                end = float(match.group(2))
                english = match.group(3).strip()
            else:
                start = None
                end = None
                english = line

            if not english:
                continue

            segments.append({
                "id": len(segments) + 1,
                "start": start,
                "end": end,
                "english_text": english
            })

    return segments, None


def format_transcript_line(item, text):
    start = item.get("start")
    end = item.get("end")

    if start is not None and end is not None:
        return f"[{start:.2f} - {end:.2f}] {text}\n"

    return text + "\n"


def get_output_stem(path):
    stem = Path(path).stem

    for suffix in ("_asr", "_raw_en", "_en", "_result"):
        if stem.endswith(suffix):
            return stem[:-len(suffix)]

    return stem


def process_translation(input_path, device="cpu"):
    input_path = Path(input_path).resolve()

    start_time = time.perf_counter()

    print("========== Translation Only ==========")
    print(f"英文逐字稿：{input_path.name}")

    segments, audio_file = read_transcript(input_path)

    print(f"翻譯片段數：{len(segments)}")

    print("\n[1/2] 載入翻譯模型...")
    tokenizer, translator = load_translation_model(device)

    print("[2/2] 開始繁體中文翻譯...")

    translated_segments = []

    for index, item in enumerate(segments, start=1):
        english = item["english_text"].strip()

        chinese = translate_text(
            english,
            tokenizer,
            translator
        )

        translated_segments.append({
            **item,
            "translated_text": chinese
        })

        print(f"\n[{index}/{len(segments)}]")
        print(f"EN: {english}")
        print(f"ZH: {chinese}")

    elapsed = time.perf_counter() - start_time

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    name = get_output_stem(input_path)

    chinese_path = OUTPUT_DIR / f"{name}_zh_tw.txt"
    json_path = OUTPUT_DIR / f"{name}_result.json"

    with open(chinese_path, "w", encoding="utf-8") as f:
        for item in translated_segments:
            f.write(
                format_transcript_line(
                    item,
                    item["translated_text"]
                )
            )

    result = {
        "audio_file": audio_file,
        "language": "en",
        "target_language": "zh-TW",
        "device": device,
        "asr_model": (
            "faster-whisper-medium"
            if audio_file else None
        ),
        "translation_model": "agentlans/en-zhtw",
        "generation_config": {
            "num_beams": TRANSLATE_NUM_BEAMS,
            "length_penalty": TRANSLATE_LENGTH_PENALTY,
            "repetition_penalty": TRANSLATE_REPETITION_PENALTY,
            "max_new_tokens": TRANSLATE_MAX_NEW_TOKENS,
            "do_sample": False
        },
        "segment_count": len(translated_segments),
        "processing_seconds": round(elapsed, 2),
        "segments": translated_segments
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n========== 翻譯完成 ==========")
    print(f"中文逐字稿：{chinese_path}")
    print(f"翻譯 JSON：{json_path}")
    print(f"處理時間：{elapsed:.2f} 秒")


def main():
    parser = argparse.ArgumentParser(
        description="英文逐字稿翻譯成繁體中文"
    )

    parser.add_argument("input", type=str)

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu"
    )

    args = parser.parse_args()

    process_translation(
        args.input,
        device=args.device
    )


if __name__ == "__main__":
    main()
