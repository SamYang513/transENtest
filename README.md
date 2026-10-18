# 英文語音辨識與繁體中文翻譯

`asr_translate.py` 使用 faster-whisper 將英文音訊轉成文字，再用英翻繁中模型翻譯，最後輸出英文、繁體中文及 JSON 檔案。

## 環境需求

- Windows、Python 3.10 以上（目前專案以 Python 3.12 測試）
- CPU 可執行；若使用 NVIDIA GPU，需安裝相容的 CUDA 與 PyTorch
- 第一次下載模型需有網路；執行程式時模型會從本機載入

## 安裝 Python 套件

在專案根目錄開啟 PowerShell：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果 PowerShell 不允許啟用虛擬環境，可在目前視窗執行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 下載模型

程式預期以下目錄與檔案存在，並以 `local_files_only=True` 載入，因此模型必須事先放好：

```text
transENtest/
└── models/
    ├── faster-whisper-medium/
    │   ├── config.json
    │   ├── model.bin
    │   ├── tokenizer.json
    │   └── vocabulary.txt
    └── en-zhtw/
        ├── config.json
        ├── generation_config.json
        ├── model.safetensors
        ├── source.spm
        ├── target.spm
        ├── tokenizer_config.json
        └── vocab.json
```

在專案根目錄執行以下指令，下載模型到程式預設路徑：

```powershell
New-Item -ItemType Directory -Force models | Out-Null
 python -m pip install huggingface_hub
hf download Systran/faster-whisper-medium --local-dir models/faster-whisper-medium
hf download agentlans/en-zhtw --local-dir models/en-zhtw
```

上述模型來源分別是 [Systran/faster-whisper-medium](https://huggingface.co/Systran/faster-whisper-medium) 與 [agentlans/en-zhtw](https://huggingface.co/agentlans/en-zhtw)。若已經有模型，直接將完整模型檔案放入上述目錄即可。下載指令可移除前方多餘空格後照常執行；`huggingface_hub` 也可加入 requirements 安裝。

## 執行

在專案根目錄執行，將 `test.wav` 換成音訊檔路徑：

```powershell
python asr_translate.py .\test.wav
```

也可指定其他位置的檔案：

```powershell
python asr_translate.py "C:\資料\meeting.wav"
```

預設使用 CPU。要使用 CUDA（需已安裝支援 CUDA 的 PyTorch 及相容環境）：

```powershell
python asr_translate.py .\test.wav --device cuda
```

## 輸出

結果會寫入 `output/`，檔名依輸入音檔名稱產生：

- `<音檔名稱>_en.txt`：英文辨識結果，含時間戳
- `<音檔名稱>_zhtw.txt`：繁體中文翻譯，含時間戳
- `<音檔名稱>_result.json`：每段時間、英文、翻譯及處理時間

## 結果檢視

目前提供的 `output/test_result.json` 中，英文辨識內容有正常產生，但 `translated_text` 出現亂碼，表示翻譯輸出目前尚不符合預期。README 和安裝清單只記錄現有程式所需項目；翻譯品質仍需另外檢查模型版本、tokenizer 與輸出文字的編碼流程。
