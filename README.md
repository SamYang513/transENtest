
# 英文語音辨識與繁體中文翻譯系統

本專案使用 faster-whisper 將英文音檔轉換成英文逐字稿，
再透過 Hugging Face 翻譯模型將英文翻譯成繁體中文。

系統支援完整執行，也可以將語音辨識與翻譯分開執行。

## 1. 使用技術

| 技術 | 用途 |
|---|---|
| Python | 程式開發 |
| faster-whisper | 英文語音辨識 |
| CTranslate2 | Whisper 模型推論 |
| PyTorch | 翻譯模型推論 |
| Transformers | 載入及執行翻譯模型 |
| SentencePiece | 翻譯模型文字切詞 |
| Hugging Face Hub | 下載模型 |
| JSON | 儲存結構化逐字稿 |

### 使用模型

英文語音辨識：

- Systran/faster-whisper-medium

英文翻譯繁體中文：

- agentlans/en-zhtw

模型在本機執行，不需要每次執行時重新下載。

## 2. 系統流程

完整流程：

音檔 WAV
    ↓
faster-whisper-medium
    ↓
英文語音辨識
    ↓
片段合併
    ↓
agentlans/en-zhtw
    ↓
繁體中文翻譯
    ↓
TXT / JSON 輸出

拆分流程：

音檔 WAV
    ↓
asr_only.py
    ↓
英文逐字稿 TXT / JSON
    ↓
translate_only.py
    ↓
繁體中文逐字稿 TXT / JSON

## 3. 專案結構

transENtest-main/
│
├── asr_translate.py
├── asr_only.py
├── translate_only.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── faster-whisper-medium/
│   └── en-zhtw/
│
├── output/
│
└── test272.wav

### 程式檔案說明

asr_translate.py
- 執行完整 ASR 與翻譯流程
- 輸入英文音檔
- 輸出英文、繁體中文及 JSON

asr_only.py
- 只執行英文語音辨識
- 不載入翻譯模型
- 保留時間戳記與片段資訊

translate_only.py
- 只執行英文翻譯繁體中文
- 不重新執行語音辨識
- 支援英文 TXT 或 ASR JSON
- 保留可用的時間戳記及片段資訊

注意：
asr_only.py 與 translate_only.py 會引用
asr_translate.py 的共用函式，
因此三支程式必須放在同一個目錄。

## 4. 環境需求

- Windows
- Python 3.12（目前專案使用版本）
- CPU 可執行
- GPU 執行需有相容的 CUDA 與 PyTorch 環境
- 下載模型時需要網路

## 5. 安裝套件

在專案目錄開啟 PowerShell：

```powershell
cd "自訂目錄"
```

建立虛擬環境：

```powershell
py -3.12 -m venv .venv
```

啟用虛擬環境：

```powershell
.\.venv\Scripts\Activate.ps1
```

更新 pip：

```powershell
python -m pip install --upgrade pip
```

安裝依賴：

```powershell
python -m pip install -r requirements.txt
```

檢查依賴：

```powershell
python -m pip check
```

若 PowerShell 阻止啟用虛擬環境：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 6. 下載模型

模型需放在專案根目錄的 models 資料夾。

建立資料夾：

```powershell
New-Item -ItemType Directory -Force models | Out-Null
```

下載英文 ASR 模型：

```powershell
hf download Systran/faster-whisper-medium --local-dir models/faster-whisper-medium
```

下載英文翻譯模型：

```powershell
hf download agentlans/en-zhtw --local-dir models/en-zhtw
```

模型來源：

- https://huggingface.co/Systran/faster-whisper-medium
- https://huggingface.co/agentlans/en-zhtw

若模型已下載完成，不需要重新下載。

## 7. 執行方式

### 方式 A：完整 ASR + 翻譯

執行：

```powershell
python asr_translate.py test.wav
```

流程：

1. 載入 faster-whisper-medium
2. 載入 agentlans/en-zhtw
3. 將英文音訊轉換成文字
4. 合併語音辨識片段
5. 翻譯成繁體中文
6. 儲存結果

輸出：

```text
output/
├── test_raw_en.txt
├── test_en.txt
├── test_zh_tw.txt
└── test_result.json
```

### 方式 B：只執行英文語音辨識

執行：

```powershell
python asr_only.py test.wav
```

流程：

1. 載入 faster-whisper-medium
2. 辨識英文語音
3. 合併語音片段
4. 輸出英文逐字稿

輸出：

```text
output/
├── test_raw_en.txt
├── test_en.txt
└── test_asr.json
```

其中：

- raw_en.txt：原始辨識片段
- en.txt：合併後的英文逐字稿
- asr.json：包含時間戳記及片段 ID

若不希望合併片段：

```powershell
python asr_only.py test.wav --no-merge
```

### 方式 C：只執行英文翻譯

建議使用上一階段產生的 JSON：

```powershell
python translate_only.py output/test_asr.json
```

也可以讀取英文 TXT：

```powershell
python translate_only.py output/test_en.txt
```

流程：

1. 讀取英文逐字稿
2. 載入 agentlans/en-zhtw
3. 將英文片段翻譯成繁體中文
4. 輸出中文 TXT 與 JSON

輸出：

```text
output/
├── test_zh_tw.txt
└── test_result.json
```

使用 JSON 作為輸入，可以保留原始的
時間戳記、片段 ID 及來源片段資訊。

一般純文字 TXT 也可以翻譯，
但若原始內容沒有時間戳記，
輸出就無法保留音訊時間資訊。

### 建議的分階段執行方式

第一步：

```powershell
python asr_only.py test.wav
```

第二步：

```powershell
python translate_only.py output/test_asr.json
```

這樣不需要在翻譯階段重新執行 Whisper。

## 8. CPU 與 GPU

三支程式預設使用 CPU。

指定 CPU：

```powershell
python asr_translate.py test.wav --device cpu
python asr_only.py test.wav --device cpu
python translate_only.py output/test_asr.json --device cpu
```

指定 GPU：

```powershell
python asr_translate.py test.wav --device cuda
python asr_only.py test.wav --device cuda
python translate_only.py output/test_asr.json --device cuda
```

注意：
GPU 執行需要相容的 NVIDIA 驅動、
CUDA、PyTorch 及 CTranslate2 環境。

## 9. 模型設定

ASR：

- language = en
- beam_size = 5
- vad_filter = False

翻譯：

- num_beams = 4
- length_penalty = 1.0
- repetition_penalty = 1.0
- max_new_tokens = 256
- do_sample = False

片段合併：

- MERGE_MAX_CHARS = 300
- MERGE_MAX_GAP = 1.5

以上設定為目前正式使用的參數。

## 10. 輸出格式

### 英文逐字稿

```text
[0.00 - 3.50] Thanks for listening to Radio 101.
```

### 繁體中文逐字稿

```text
[0.00 - 3.50] 感謝您收聽 Radio 101。
```

### JSON

```json
{
  "segments": [
    {
      "id": 1,
      "start": 0.0,
      "end": 3.5,
      "english_text": "Thanks for listening to Radio 101.",
      "source_segment_ids": [1],
      "translated_text": "感謝您收聽 Radio 101。"
    }
  ]
}
```

此 JSON 為片段資料格式示例。
實際輸出還包含模型名稱、裝置與處理時間等資訊。

## 11. 常見問題

### 找不到模型

確認模型已放在：

```text
models/faster-whisper-medium/
models/en-zhtw/
```

### 找不到 Python 套件

確認已啟用虛擬環境，並執行：

```powershell
python -m pip install -r requirements.txt
```

### Transformers 出現長度設定警告

可能看到：

```text
Both max_new_tokens and max_length seem to have been set.
max_new_tokens will take precedence.
```

這是模型預設 max_length 與程式指定
max_new_tokens 同時存在的提示。

目前保留原始生成參數，不為了消除警告
而更改翻譯模型的生成行為。

### 中文顯示亂碼

請確認輸出檔案以 UTF-8 開啟，
並檢查終端機及文字編輯器的編碼設定。

### JSON 與 TXT 有什麼差別？

TXT 適合直接閱讀。

JSON 適合程式處理、保留時間資訊、
後續翻譯及系統整合。

## 12. 注意事項

- 模型必須事先下載到本地
- 完整流程與分階段流程可能覆寫同名輸出檔
- 修改英文片段內容或切分方式可能影響翻譯結果
- 翻譯品質仍需要人工檢查
- 建議使用 JSON 串接 ASR 與翻譯階段
