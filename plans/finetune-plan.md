# shinnku-bot：微調試點

寫於 2026-11-22，接在 [v2-gaps-plan.md](v2-gaps-plan.md) 第 8 節的第 4 條之後。最初打算在 Tinker 上訓，按 token 計費太貴（試點約 US$24，全量約 US$190），改成 Colab，用官方的 `colab` 命令行從這臺服務器上操作。代碼都在這個倉庫裡。

## 0. 為什麼要做

- **提示詞這條路到頂了**。鑑別器 AUC 從裸模型的 0.870 降到加固定樣本和檢索後的 0.724，之後加人物卡、改長度、關推理、加事實清單，都停在 0.72–0.75，一直沒低於群友線 0.666。剩下的露餡是說什麼（英文字母只有我的一半、數字只有三分之一），不是怎麼說，提示詞改不了。
- **文獻也這麼說**。EMNLP 2025 測了 400 多位作者，提示裡的例子從 2 個加到 10 個幾乎沒用。IMPersona 裡 LoRA 微調的 8B 模型加記憶，被認識本人的人判成真人 44.4%，帶記憶的提示詞方案最好的只有 25%；數據越多越像，而它最多用了 1.3 萬條，我們有 80,588 對。
- **底座選 Qwen3.6-35B-A3B**。gpt-6-astra 不能微調；DeepSeek-V4.1-Flash 有 552B，沒有卡跑得動。Qwen3.6-35B-A3B 訓完量化到 4-bit 約 22G，實驗室那張 3090 Ti 能跑。訓練要約 74G 顯存（Unsloth 的數字，bf16 LoRA；它不建議對這一代模型做 4-bit QLoRA），所以在 Colab 的 G4（96G）上訓。

## 1. 結論一覽

| 要決定的 | 決定 |
|---|---|
| 底座 | `unsloth/Qwen3.6-35B-A3B`，關閉思考 |
| 訓練 | Colab G4 上用 Unsloth 做 bf16 LoRA，rank 32，2 輪；試點用 5,000 對，另留 300 對看驗證損失 |
| 訓練樣本 | 線上 harness 拼出的提示詞，去掉固定樣本、檢索和事實清單，末尾接上我當時真實的回覆；只對這條回覆算損失 |
| 回覆格式 | 純文本，一行一條消息，不用工具 |
| 評估 | 同一批 200 個留出場面（`--n 200`，seed 0）。回覆在 Colab 上生成，拿回來走一遍 harness 的清理，再跑 `eval style` |
| 算達標 | AUC 低於群友線 0.666，即 bot 比任何一個群友都更像我。不量 gpt-6-astra 的基線（你定的），之前 DeepSeek 的 0.724 只作參考 |
| 部署 | 達標以後再定，見第 3 節 |

## 2. 步驟

### 第 1 步：導出（`shinnku corpus sft`，已寫好）

- 從訓練集裡抽，不碰留出的 600 個場面。抽法和固定樣本一樣按時間加權、近期多，因為 2025 年起我的消息明顯變長，而留出集是最新的。最後一句是我自己說的場面沒有要回的話，跳過。
- 每個場面走和 `eval run` 相同的 replay 路徑，讓 harness 照線上的樣子拼提示詞，用一個只記錄、不調模型的假模型把 messages 截下來。訓練時和上線時模型看到的格式逐字一致。
- 三樣東西不放進去：
  - **固定樣本**：權重就是用來代替它的，放進去每條要多 5 萬 token。
  - **檢索**：檢索索引就是用訓練集建的，訓練對會檢索到它自己，等於把答案放進題目。
  - **事實清單**：每條多約 3k token，訓練時間翻一倍。上線時照樣帶上。
- 人物卡和群卡照常帶上。
- 寫到 `data/sft/`：`train.jsonl`、`valid.jsonl`（提示詞加我的回覆），`eval.jsonl`、`eval_recall.jsonl`（200 個留出場面的提示詞，後者帶檢索；不含我的真實回覆）。控制台只打印條數和字數。

### 第 2 步：在 Colab 上訓練（`finetune/`，已寫好）

- `finetune/setup.sh`：在新開的 VM 上裝 Unsloth，版本照抄它官方的 Qwen3.5 MoE notebook。
- `finetune/train.py`：讀上面的文件，訓 LoRA，每 100 步在驗證集上算一次損失並存檢查點（斷了重跑會接著訓），存下 LoRA，再給 `eval*.jsonl` 每個場面生成一條回覆（溫度 0.7）。只打印條數、token 數、損失和耗時，不打印任何文字。參數用 `SFT_*` 環境變量傳。
- 命令（都在這臺服務器上跑）：

```bash
colab new -s ft --gpu G4
colab ssh -s ft -- mkdir -p /content/sft
for f in train valid eval eval_recall; do colab upload -s ft data/sft/$f.jsonl /content/sft/$f.jsonl; done
colab upload -s ft finetune/setup.sh /content/setup.sh
colab ssh -s ft -- bash /content/setup.sh
colab exec -s ft --timeout 0 -f finetune/train.py
colab download -s ft /content/out/eval_out.jsonl data/sft/eval_out.jsonl
colab download -s ft /content/out/eval_recall_out.jsonl data/sft/eval_recall_out.jsonl
colab ssh -s ft -- tar czf /content/lora.tgz -C /content/out lora
colab download -s ft /content/lora.tgz data/sft/lora.tgz
colab stop -s ft
```

- **先空跑一次**：在 T4 或 L4 上用 `unsloth/Qwen3.5-0.8B`，`--env SFT_MODEL=unsloth/Qwen3.5-0.8B --env SFT_MAX_STEPS=5 --env SFT_LIMIT=64`，把安裝、訓練、生成、下載整條流程走通，花不到一個計算單元。確認沒問題再開 G4。
  - 2026-11-22 用編的假對話跑過一次，T4 上從開會話到停約 12 分鐘（裝環境 5 分鐘，訓練加生成 5.5 分鐘），花了 0.19 個單元，整條流程沒報錯。LoRA 掛在所有層的 MLP 和全注意力層的 q/k/v/o 上，線性注意力層不掛，和官方 notebook 一樣。

### 第 3 步：評估

```bash
uv run shinnku eval run --name ft --n 200 --replies data/sft/eval_out.jsonl
uv run shinnku eval style --name ft
```

`--replies` 讓每個場面直接用 Colab 上寫好的回覆，不調模型；回覆照樣經過 harness 的清理和檢查，和線上一樣。

| 配置 | 看什麼 |
|---|---|
| A 微調，不帶固定樣本、不帶檢索（`eval_out.jsonl`） | 主結果 |
| B 微調，加檢索（`eval_recall_out.jsonl`） | 微調以後檢索還有沒有用 |

除了 AUC，照常看長度分佈、連發條數和語義接近度。你的 `eval quiz --name ft` 再做一次。

### 第 4 步：決定

- **達標**：用更多的數據（3–8 萬對）再訓一次，然後按第 3 節部署。
- **沒達標**：停下。

## 3. 部署（達標以後）

- 把 LoRA 合併進底座，量化成 4-bit 的 GGUF（約 22G），用 llama.cpp 的服務在實驗室的 3090 Ti 上跑，它提供 OpenAI 兼容的接口。合併和量化可以在 Colab 上用 Unsloth 的 `save_pretrained_gguf` 做。有人報告 Unsloth 訓出來的 MoE LoRA 直接掛到 vLLM 上加載不了，所以先合併再量化。
- 外網連不進那臺機器，要由它主動建隧道到 bot 的服務器。
- `GatewayModel` 加純文本模式：不發 `tools` 和思考參數，回覆按行切成消息（`parse_action` 本來就接受純文本）。純文本沒有“沉默”和 `to` 編號，上線前要定怎麼處理。
- Qwen3.5/3.6 的聊天模板只保留開頭的 system 消息，後面的會被默默丟掉。harness 把考題的“敷衍”和 AI 腔重試的提示放在聊天末尾的 system 消息裡，純文本模式要把它們併進最後一條 user 消息。訓練數據不受影響：回放時不判考題，也只截第一次的提示詞。
- 量化以後再跑一次第 3 步的 A，確認風格沒有掉。
- 分工：說話走微調模型；判斷考題、決定搜什麼、看圖仍然用 gpt-6-astra；備用模型不變。
- 那張卡是和別人共用的，常駐服務會一直佔著約 22G 顯存。

## 4. 費用

Colab 按計算單元扣，100 個單元 US$9.99（約 HK$78）。G4 每小時約 8.7 個單元，這是第三方的數字，Google 不公佈。訓練時間按算力估，試點跑完就知道真實速度。

| 項 | 時間 | 計算單元 | 約合 |
|---|---|---|---|
| 空跑（T4 或 L4，小模型） | 約 20 分鐘 | 不到 1 | 不到 US$0.1 |
| 試點：裝環境、下載 72G 模型、訓 5,000 對 × 2 輪、生成 400 條回覆 | 2–4 小時 | 15–35 | US$1.5–3.5（HK$12–27） |
| 達標後的全量訓練：3–8 萬對 × 1 輪 | 10–20 小時 | 90–175 | US$9–18（HK$70–140） |

賬上現在有 9.22 個單元，夠空跑，不夠試點。Colab Pro 的單次會話最長 12 小時，Pro+ 24 小時；全量訓練可能超過 12 小時，要靠檢查點接著訓。

## 5. 風險

- **74G 只是 Unsloth 的數字**，換到 Qwen3.6、序列長度 4096 可能更多。G4 有 96G，A100 80G 不一定夠。爆了就調小 `SFT_BATCH` 或 `SFT_MAX_LEN`。
- **G4 不一定分得到**，按 Colab 的說法看可用量。2026-11-22 試開：帶 `--high-mem` 被直接拒絕（說沒有權限，賬上是 Google AI Pro 帶的 Colab 權益）；不帶它連試兩次都回 503；之後一小時裡每 10 分鐘試一次，7 次都是 412（命令行說是活動會話太多，或者暫時的用量或容量限制；當時沒有開著的會話）。當時餘額只有 9 個單元，Google 說分不分得到指定的卡也看付費餘額，所以分不清是餘額低還是沒卡。colabtools 的 [#6014](https://github.com/googlecolab/colabtools/issues/6014) 裡，Pro+ 用戶從 2026 年 6 月起就常常連不上 H100、G4、A100，Colab 的人說是空閒算力被用光了。普通規格的 G4 有多少內存，要等真分到了才知道。
- **35B-A3B 比 gpt-6-astra 笨**，難題和代碼會答得差。考題本來就不答；新聞之類有檢索兜著。
- **微調模型的老毛病**：突然跳話題、硬塞記住的細節（IMPersona 都觀察到了）。
- **會背出訓練數據裡的話**，包括群裡別人說過的。
- **量化可能損失風格**，第 3 節裡要重新量。

## 6. 需要你做的

1. 買計算單元：試點至少 100 個（US$9.99）；全量訓練要 200 個左右，或者開一個月 Pro+（US$49.99，600 個單元，會話最長 24 小時）。
2. 同意把約 5,500 段群聊（包括別人說的話）上傳到 Colab 的 VM。這和現在回覆、蒸餾時把內容發給網關上游是同一類事，但換成了 Google。VM 用完就停，LoRA 拿回這臺機器。
3. 試點之後決定要不要全量訓練和部署，以及能不能長期佔用實驗室的卡。

## 7. 依據

- [Catch Me If You Can? Not Yet](https://arxiv.org/abs/2509.14543)
- [IMPersona](https://arxiv.org/abs/2504.04332)
- [Unsloth 的 Qwen3.5 微調指南](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
- [Unsloth 的 Qwen3.5 MoE notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_5_MoE.ipynb)
- [Google Colab CLI](https://github.com/googlecolab/google-colab-cli)
- [MoE LoRA 在 vLLM 上加載失敗的報告](https://forums.developer.nvidia.com/t/anyone-have-a-solution-for-lora-training-of-recent-moe-models-like-qwen3-5-35b-a3b-or-gemma-4-26b-a4b-and-successfully-running-in-vllm/366223)
