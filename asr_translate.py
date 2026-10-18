
import argparse
import json
import time
from pathlib import Path

import torch
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ==========================================
# 1. 專案路徑設定
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

ASR_MODEL_PATH = BASE_DIR / "models" / "faster-whisper-medium"
TRANSLATE_MODEL_PATH = BASE_DIR / "models" / "en-zhtw"
OUTPUT_DIR = BASE_DIR / "output"


# ==========================================
# 2. 載入模型
# ==========================================

def load_models(device="cpu"):

    if not ASR_MODEL_PATH.is_dir():
        raise FileNotFoundError(
            f"找不到 ASR 模型：{ASR_MODEL_PATH}"
        )

    if not TRANSLATE_MODEL_PATH.is_dir():
        raise FileNotFoundError(
            f"找不到翻譯模型：{TRANSLATE_MODEL_PATH}"
        )

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch 無法使用 CUDA，請改用 --device cpu"
        )

    print("\n========== 載入模型 ==========")

    start = time.perf_counter()

    print("[1/2] 載入 faster-whisper-medium...")

    asr_model = WhisperModel(
        str(ASR_MODEL_PATH),
        device=device,
        compute_type="float16" if device == "cuda" else "int8",
        local_files_only=True
    )

    print("[2/2] 載入 en-zhtw 翻譯模型...")

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

    elapsed = time.perf_counter() - start

    print(f"模型載入完成，耗時：{elapsed:.2f} 秒")

    return asr_model, tokenizer, translator


# ==========================================
# 3. 英文翻譯繁體中文
# ==========================================

def translate_text(text, tokenizer, translator):

    if not text.strip():
        return ""

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    # 將輸入移至翻譯模型所在裝置
    device = next(translator.parameters()).device
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = translator.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=4
        )

    translated = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return translated


# ==========================================
# 4. ASR 語音辨識與翻譯
# ==========================================

def process_audio(
    audio_path,
    asr_model,
    tokenizer,
    translator
):

    print("\n========== 開始處理音訊 ==========")
    print(f"音訊檔案：{audio_path.name}")

    start_time = time.perf_counter()

    segments, info = asr_model.transcribe(
        str(audio_path),
        language="en",
        task="transcribe",
        beam_size=5,
        vad_filter=True
    )

    print("辨識語言：英文")

    results = []

    # 逐段辨識與翻譯
    for index, segment in enumerate(segments, start=1):

        english = segment.text.strip()

        if not english:
            continue

        chinese = translate_text(
            english,
            tokenizer,
            translator
        )

        item = {
            "id": index,
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "english_text": english,
            "translated_text": chinese
        }

        results.append(item)

        print(f"\n========== Segment {index} ==========")
        print(
            f"時間：{segment.start:.2f}s "
            f"-> {segment.end:.2f}s"
        )
        print(f"英文：{english}")
        print(f"繁中：{chinese}")

    elapsed = time.perf_counter() - start_time

    print("\n========== 處理完成 ==========")
    print(f"總處理時間：{elapsed:.2f} 秒")
    print(f"總片段數：{len(results)}")

    return results, elapsed


# ==========================================
# 5. 儲存 TXT 與 JSON
# ==========================================

def save_results(audio_path, results, elapsed):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    english_file = (
        OUTPUT_DIR / f"{audio_path.stem}_en.txt"
    )

    chinese_file = (
        OUTPUT_DIR / f"{audio_path.stem}_zhtw.txt"
    )

    json_file = (
        OUTPUT_DIR / f"{audio_path.stem}_result.json"
    )

    english_lines = []
    chinese_lines = []

    for item in results:

        timestamp = (
            f"[{item['start']:.2f}s "
            f"-> {item['end']:.2f}s]"
        )

        english_lines.append(
            f"{timestamp} {item['english_text']}"
        )

        chinese_lines.append(
            f"{timestamp} {item['translated_text']}"
        )

    english_file.write_text(
        "\n".join(english_lines),
        encoding="utf-8-sig"
    )

    chinese_file.write_text(
        "\n".join(chinese_lines),
        encoding="utf-8-sig"
    )

    json_data = {
        "audio_file": audio_path.name,
        "asr_model": "Systran/faster-whisper-medium",
        "translation_model": "agentlans/en-zhtw",
        "source_language": "en",
        "target_language": "zh-TW",
        "processing_time_seconds": round(elapsed, 2),
        "segments": results
    }

    json_file.write_text(
        json.dumps(
            json_data,
            ensure_ascii=False,
            indent=4
        ),
        encoding="utf-8"
    )

    print("\n========== 輸出檔案 ==========")
    print(f"英文逐字稿：{english_file}")
    print(f"繁中翻譯：{chinese_file}")
    print(f"JSON 結果：{json_file}")


# ==========================================
# 6. 主程式
# ==========================================

def main():

    parser = argparse.ArgumentParser(
        description="英文 ASR + 繁體中文翻譯系統"
    )

    parser.add_argument(
        "audio",
        help="音訊檔案路徑，例如 test.wav"
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="執行裝置，預設 cpu"
    )

    args = parser.parse_args()

    audio_path = Path(args.audio).resolve()

    if not audio_path.is_file():
        print(f"錯誤：找不到音訊檔案 {audio_path}")
        return

    asr_model, tokenizer, translator = load_models(
        device=args.device
    )

    results, elapsed = process_audio(
        audio_path,
        asr_model,
        tokenizer,
        translator
    )

    save_results(
        audio_path,
        results,
        elapsed
    )


if __name__ == "__main__":
    main()
