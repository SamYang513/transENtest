
import argparse
import json
import re
import time
from pathlib import Path

import torch
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


BASE_DIR = Path(__file__).resolve().parent

ASR_MODEL_PATH = BASE_DIR / "models" / "faster-whisper-medium"
TRANSLATE_MODEL_PATH = BASE_DIR / "models" / "en-zhtw"
OUTPUT_DIR = BASE_DIR / "output"

# Baseline A
ASR_BEAM_SIZE = 5
TRANSLATE_NUM_BEAMS = 4
TRANSLATE_LENGTH_PENALTY = 1.0
TRANSLATE_REPETITION_PENALTY = 1.0
TRANSLATE_MAX_NEW_TOKENS = 256

MERGE_MAX_CHARS = 300
MERGE_MAX_GAP = 1.5


def load_asr_model(device="cpu"):
    if not ASR_MODEL_PATH.is_dir():
        raise FileNotFoundError(
            f"找不到 ASR 模型：{ASR_MODEL_PATH}"
        )

    return WhisperModel(
        str(ASR_MODEL_PATH),
        device=device,
        compute_type="int8"
    )


def load_translation_model(device="cpu"):
    if not TRANSLATE_MODEL_PATH.is_dir():
        raise FileNotFoundError(
            f"找不到翻譯模型：{TRANSLATE_MODEL_PATH}"
        )

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PyTorch 無法使用 CUDA")

    tokenizer = AutoTokenizer.from_pretrained(
        str(TRANSLATE_MODEL_PATH),
        local_files_only=True
    )

    translator = AutoModelForSeq2SeqLM.from_pretrained(
        str(TRANSLATE_MODEL_PATH),
        local_files_only=True
    )

    translator = translator.to(device)
    translator.eval()

    return tokenizer, translator


def transcribe_audio(audio_path, asr_model):
    segments, info = asr_model.transcribe(
        str(audio_path),
        language="en",
        beam_size=ASR_BEAM_SIZE,
        vad_filter=False
    )

    results = []

    for index, segment in enumerate(segments, start=1):
        text = segment.text.strip()

        if not text:
            continue

        results.append({
            "id": index,
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "english_text": text
        })

        print(
            f"[{segment.start:.2f} - {segment.end:.2f}] "
            f"{text}"
        )

    return results, info


def merge_asr_segments(
    segments,
    max_chars=MERGE_MAX_CHARS,
    max_gap=MERGE_MAX_GAP
):
    if not segments:
        return []

    merged = []
    current = None

    for segment in segments:
        text = segment["english_text"].strip()

        if current is None:
            current = {
                "start": segment["start"],
                "end": segment["end"],
                "english_text": text,
                "source_segment_ids": [segment["id"]]
            }
            continue

        gap = segment["start"] - current["end"]

        previous_ends_sentence = bool(
            re.search(r'[.!?]["\']?$', current["english_text"])
        )

        would_exceed_length = (
            len(current["english_text"]) + len(text) + 1
            > max_chars
        )

        if (
            previous_ends_sentence
            or gap > max_gap
            or would_exceed_length
        ):
            merged.append(current)

            current = {
                "start": segment["start"],
                "end": segment["end"],
                "english_text": text,
                "source_segment_ids": [segment["id"]]
            }
        else:
            current["english_text"] += " " + text
            current["end"] = segment["end"]
            current["source_segment_ids"].append(segment["id"])

    if current is not None:
        merged.append(current)

    for index, item in enumerate(merged, start=1):
        item["id"] = index

    return merged


def translate_text(text, tokenizer, translator):
    if not text.strip():
        return ""

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    device = next(translator.parameters()).device

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # 保留 Baseline A 原始 generate() 行為。
    # 不修改 generation_config。
    # Transformers 的長度設定警告可以保留。
    with torch.inference_mode():
        outputs = translator.generate(
            **inputs,
            num_beams=TRANSLATE_NUM_BEAMS,
            length_penalty=TRANSLATE_LENGTH_PENALTY,
            repetition_penalty=TRANSLATE_REPETITION_PENALTY,
            max_new_tokens=TRANSLATE_MAX_NEW_TOKENS,
            do_sample=False
        )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    ).strip()


def save_results(
    audio_path,
    raw_segments,
    translated_segments,
    elapsed_seconds,
    device
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    name = audio_path.stem

    raw_path = OUTPUT_DIR / f"{name}_raw_en.txt"
    english_path = OUTPUT_DIR / f"{name}_en.txt"
    chinese_path = OUTPUT_DIR / f"{name}_zh_tw.txt"
    json_path = OUTPUT_DIR / f"{name}_result.json"

    with open(raw_path, "w", encoding="utf-8") as f:
        for item in raw_segments:
            f.write(
                f"[{item['start']:.2f} - {item['end']:.2f}] "
                f"{item['english_text']}\n"
            )

    with open(english_path, "w", encoding="utf-8") as f:
        for item in translated_segments:
            f.write(
                f"[{item['start']:.2f} - {item['end']:.2f}] "
                f"{item['english_text']}\n"
            )

    with open(chinese_path, "w", encoding="utf-8") as f:
        for item in translated_segments:
            f.write(
                f"[{item['start']:.2f} - {item['end']:.2f}] "
                f"{item['translated_text']}\n"
            )

    result = {
        "audio_file": audio_path.name,
        "language": "en",
        "target_language": "zh-TW",
        "device": device,
        "asr_model": "faster-whisper-medium",
        "translation_model": "agentlans/en-zhtw",
        "generation_config": {
            "num_beams": TRANSLATE_NUM_BEAMS,
            "length_penalty": TRANSLATE_LENGTH_PENALTY,
            "repetition_penalty": TRANSLATE_REPETITION_PENALTY,
            "max_new_tokens": TRANSLATE_MAX_NEW_TOKENS,
            "do_sample": False
        },
        "raw_segment_count": len(raw_segments),
        "segment_count": len(translated_segments),
        "processing_seconds": round(elapsed_seconds, 2),
        "segments": translated_segments
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n========== 輸出完成 ==========")
    print(f"原始英文：{raw_path}")
    print(f"合併英文：{english_path}")
    print(f"繁體中文：{chinese_path}")
    print(f"JSON：{json_path}")


def process_audio(audio_path, device="cpu", merge=True):
    audio_path = Path(audio_path).resolve()

    if not audio_path.is_file():
        raise FileNotFoundError(f"找不到音檔：{audio_path}")

    start_time = time.perf_counter()

    print("========== ASR + Translation ==========")
    print(f"音檔：{audio_path.name}")
    print(f"裝置：{device}")

    print("\n[1/4] 載入 ASR 模型...")
    asr_model = load_asr_model(device)

    print("[2/4] 載入翻譯模型...")
    tokenizer, translator = load_translation_model(device)

    print("[3/4] 開始英文語音辨識...")
    raw_segments, info = transcribe_audio(audio_path, asr_model)

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

    print("\n[4/4] 開始繁體中文翻譯...")

    translated_segments = []

    for index, item in enumerate(segments, start=1):
        english = item["english_text"]

        chinese = translate_text(
            english,
            tokenizer,
            translator
        )

        result = {
            **item,
            "translated_text": chinese
        }

        translated_segments.append(result)

        print(f"\n[{index}/{len(segments)}]")
        print(f"EN: {english}")
        print(f"ZH: {chinese}")

    elapsed_seconds = time.perf_counter() - start_time

    save_results(
        audio_path,
        raw_segments,
        translated_segments,
        elapsed_seconds,
        device
    )

    print(f"\n總執行時間：{elapsed_seconds:.2f} 秒")

    return translated_segments


def main():
    parser = argparse.ArgumentParser(
        description="英文 ASR + 繁體中文翻譯"
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

    process_audio(
        audio_path=args.audio,
        device=args.device,
        merge=not args.no_merge
    )


if __name__ == "__main__":
    main()
