
import argparse
import copy
import csv
import json
import time
from pathlib import Path

import torch
from sacrebleu.metrics import BLEU, CHRF
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# 1. 專案設定
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"
COMPARISON_DIR = OUTPUT_DIR / "model_comparison"

RESULT_PATH = OUTPUT_DIR / "test272_result.json"
REFERENCE_PATH = OUTPUT_DIR / "test272_reference.json"

MODELS = {
    "agentlans_en_zhtw": {
        "path": MODELS_DIR / "en-zhtw",
        "prefix": ""
    },
    "opus_mt_en_zh": {
        "path": MODELS_DIR / "opus-mt-en-zh",
        "prefix": ">>cmn_Hant<< "
    }
}

NUM_BEAMS = 4
MAX_NEW_TOKENS = 256
MAX_INPUT_TOKENS = 256


# ============================================================
# 2. 讀取 JSON
# ============================================================

def read_json(path):
    if not path.is_file():
        raise FileNotFoundError(f"找不到檔案：{path}")

    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_test_data(result_path, reference_path):

    result_data = read_json(result_path)
    reference_data = read_json(reference_path)

    predictions = result_data.get("segments", [])
    references = reference_data.get("segments", [])

    if not predictions or not references:
        raise ValueError("測試資料或參考翻譯的 segments 為空")

    # 使用 id 配對，避免單純依列表順序造成錯位
    reference_map = {}

    for item in references:
        segment_id = item.get("id")

        if segment_id is None:
            raise ValueError("參考翻譯有片段缺少 id")

        if segment_id in reference_map:
            raise ValueError(f"參考翻譯有重複 id：{segment_id}")

        reference_map[segment_id] = item

    test_data = []
    seen_ids = set()

    for item in predictions:

        segment_id = item.get("id")

        if segment_id is None:
            raise ValueError("辨識結果有片段缺少 id")

        if segment_id in seen_ids:
            raise ValueError(f"辨識結果有重複 id：{segment_id}")

        seen_ids.add(segment_id)

        if segment_id not in reference_map:
            raise ValueError(
                f"找不到片段 {segment_id} 的人工參考翻譯"
            )

        ref_item = reference_map[segment_id]

        english = item.get("english_text", "").strip()
        reference = ref_item.get("reference_zh", "").strip()

        if not english or not reference:
            raise ValueError(
                f"片段 {segment_id} 的英文或參考翻譯為空"
            )

        # 若參考檔包含英文，額外檢查是否對齊
        ref_english = ref_item.get("english_text", "").strip()

        if ref_english and ref_english != english:
            raise ValueError(
                f"片段 {segment_id} 的英文與參考檔不一致\n"
                f"Result: {english}\n"
                f"Reference: {ref_english}"
            )

        test_data.append({
            "id": segment_id,
            "english_text": english,
            "reference_zh": reference
        })

    if len(test_data) != len(reference_map):
        raise ValueError(
            "兩份 JSON 的片段數量或 id 不一致，"
            "請確認使用同一份測試資料"
        )

    return test_data


# ============================================================
# 3. 載入翻譯模型
# ============================================================

def load_translation_model(model_path, device):

    if not model_path.is_dir():
        raise FileNotFoundError(
            f"找不到本地模型：{model_path}"
        )

    print(f"\n載入模型：{model_path.name}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(model_path),
        local_files_only=True
    )

    model = model.to(device)
    model.eval()

    return tokenizer, model


# ============================================================
# 4. 翻譯函式
# ============================================================

def translate_text(
    text,
    tokenizer,
    model,
    device,
    prefix=""
):

    input_text = prefix + text

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # 使用獨立設定，不修改模型原始 generation_config
    generation_config = copy.deepcopy(
        model.generation_config
    )

    # 解決 max_length 與 max_new_tokens 同時設定的警告
    generation_config.max_length = None
    generation_config.max_new_tokens = MAX_NEW_TOKENS

    generation_config.num_beams = NUM_BEAMS
    generation_config.length_penalty = 1.0
    generation_config.repetition_penalty = 1.0
    generation_config.do_sample = False

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config
        )

    translation = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return translation.strip()


# ============================================================
# 5. 計算 BLEU / chrF
# ============================================================

def calculate_metrics(predictions, references):

    bleu = BLEU(tokenize="char")
    chrf = CHRF()

    bleu_score = bleu.corpus_score(
        predictions,
        [references]
    ).score

    chrf_score = chrf.corpus_score(
        predictions,
        [references]
    ).score

    return round(bleu_score, 2), round(chrf_score, 2)


# ============================================================
# 6. 執行單一模型
# ============================================================

def evaluate_model(model_name, model_config, test_data, device):

    tokenizer, model = load_translation_model(
        model_config["path"],
        device
    )

    predictions = []
    references = []
    details = []

    start_time = time.perf_counter()

    for index, item in enumerate(test_data, start=1):

        english = item["english_text"]
        reference = item["reference_zh"]

        translated = translate_text(
            english,
            tokenizer,
            model,
            device,
            prefix=model_config["prefix"]
        )

        predictions.append(translated)
        references.append(reference)

        sentence_bleu, sentence_chrf = calculate_metrics(
            [translated],
            [reference]
        )

        details.append({
            "id": item["id"],
            "english_text": english,
            "prediction": translated,
            "reference_zh": reference,
            "sentence_bleu": sentence_bleu,
            "sentence_chrf": sentence_chrf
        })

        print(
            f"[{model_name}] "
            f"{index}/{len(test_data)}"
        )
        print(f"EN: {english}")
        print(f"ZH: {translated}\n")

    elapsed = time.perf_counter() - start_time

    bleu_score, chrf_score = calculate_metrics(
        predictions,
        references
    )

    # 儲存逐句結果
    details_path = COMPARISON_DIR / f"{model_name}_details.csv"

    save_csv(details_path, details)

    summary = {
        "model": model_name,
        "segments": len(test_data),
        "bleu": bleu_score,
        "chrf": chrf_score,
        "translation_seconds": round(elapsed, 2)
    }

    print(f"\n========== {model_name} ==========")
    print(f"BLEU：{bleu_score}")
    print(f"chrF：{chrf_score}")
    print(f"翻譯時間：{elapsed:.2f} 秒")

    # 釋放模型，避免兩個模型同時占用記憶體
    del model
    del tokenizer

    if device == "cuda":
        torch.cuda.empty_cache()

    return summary, details


# ============================================================
# 7. CSV 輸出
# ============================================================

def save_csv(path, rows):

    if not rows:
        return

    with open(
        path,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 8. 產生模型對照表
# ============================================================

def save_side_by_side(test_data, all_details):

    lookup = {}

    for model_name, details in all_details.items():
        lookup[model_name] = {
            item["id"]: item
            for item in details
        }

    rows = []

    for item in test_data:

        segment_id = item["id"]

        row = {
            "id": segment_id,
            "english_text": item["english_text"],
            "reference_zh": item["reference_zh"]
        }

        for model_name in MODELS:

            detail = lookup[model_name][segment_id]

            row[f"{model_name}_prediction"] = (
                detail["prediction"]
            )

            row[f"{model_name}_chrf"] = (
                detail["sentence_chrf"]
            )

        rows.append(row)

    save_csv(
        COMPARISON_DIR / "side_by_side.csv",
        rows
    )


# ============================================================
# 9. 主程式
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="比較兩個英文翻譯繁體中文模型"
    )

    parser.add_argument(
        "--result",
        type=Path,
        default=RESULT_PATH
    )

    parser.add_argument(
        "--reference",
        type=Path,
        default=REFERENCE_PATH
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu"
    )

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch 無法使用 CUDA，請改用 --device cpu"
        )

    COMPARISON_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    test_data = load_test_data(
        args.result,
        args.reference
    )

    print("========== 翻譯模型比較 ==========")
    print(f"測試句數：{len(test_data)}")
    print(f"裝置：{args.device}")

    summaries = []
    all_details = {}

    for model_name, config in MODELS.items():

        summary, details = evaluate_model(
            model_name,
            config,
            test_data,
            args.device
        )

        summaries.append(summary)
        all_details[model_name] = details

    save_csv(
        COMPARISON_DIR / "summary.csv",
        summaries
    )

    save_side_by_side(
        test_data,
        all_details
    )

    print("\n========== 最終比較 ==========")

    for row in summaries:
        print(
            f"{row['model']}: "
            f"BLEU={row['bleu']}, "
            f"chrF={row['chrf']}, "
            f"時間={row['translation_seconds']}秒"
        )

    print(f"\n輸出位置：{COMPARISON_DIR}")


if __name__ == "__main__":
    main()
