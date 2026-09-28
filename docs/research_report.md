# 研究成果報告：Concept Bottleneck Models Benchmark Library

> UCSD International Summer Research Program（2025）
> 研究者：Pin-Ci Huang（國立中央大學）
> 指導：Prof. Lily Weng、Ge Yan（Ph.D.）

---

## 1. 研究目標

Concept Bottleneck Model（CBM）先把影像特徵投影到一組人類可讀的「概念」上，再只用這些概念做分類，因此模型的決策可以被解釋。近年的 CBM 方法（Label-Free CBM、LaBo、LM4CV、VLG-CBM 等）大多用大型語言模型（LLM）自動產生概念集，但各方法的程式碼、概念格式與評估方式都不相同，難以公平比較。

本專案的目標是建立一個**統一的 CBM 訓練／評估工具庫**：

1. **概念生成**：用同一個開源 LLM，依照四種方法的 prompt 流程重新產生概念集，輸出成統一格式。
2. **模型訓練**：在同一套程式碼中實作 LF-CBM 與 VLG-CBM，並共用 final layer 訓練模組。
3. **評估**：整合 ANEC（Accuracy under Number of Effective Concepts）工具，在相同稀疏度下比較不同 CBM。

---

## 2. 系統架構

```
cbm_library/
├── concepts/          # ① 概念生成（依方法分資料夾，見 concepts/README.md）
│   ├── label_free/  labo/  lm4cv/  cb_llm/
│   └── common/  config/  classes/
├── config/            # ② 訓練設定 dataclass（LF-CBM, VLG-CBM, Final layer）
├── models/            # ② 模型實作：lf_cbm.py, vlg_cbm.py, final_layer.py
├── scripts/           # ② 訓練入口：lf_cbm_train.py, vlg_cbm_train.py
├── utils/             # 資料載入、concept dataset、loss、GLM-SAGA、logging
└── evaluation/        # ③ evaluate_cbm.ipynb + ANEC-evaluator
```

資料流：`concepts/<method>/outputs/*` → `scripts/*_train.py` → `saved_models/` → `evaluation/`

---

## 3. 概念生成（Concept Generation）

### 3.1 實驗設定

| 項目 | 設定 |
|---|---|
| LLM | `openchat/openchat-3.5-0106`（以 vLLM 部署、OpenAI 相容 API） |
| 取樣參數 | temperature 0.7、max_tokens 150、top_p 1.0 |
| 影像資料集 | CIFAR-10（10 類）、CIFAR-100（100）、CUB-200（200）、Places365（365）、ImageNet（1000） |
| 文字資料集（CB-LLM） | SST2、YelpP、AGNews、DBpedia |

原論文多使用 GPT-3 / GPT-3.5；本專案改用開源的 OpenChat-3.5，讓整個流程能在實驗室叢集上重現、不依賴付費 API。

### 3.2 各方法流程

| 方法 | Prompt／流程 | 輸出 |
|---|---|---|
| **Label-Free CBM** | 每個類別問三種問題：*important features*、*superclass*、*things around*；合併後做長度／黑名單／泛用詞過濾與去重 | 3 個 JSON（每種 prompt 一個）＋ `<dataset>_filtered.txt` |
| **LaBo** | 每類大量產生描述句並切成短概念，再以 submodular selection（discriminability α=1e7 + coverage β=1）每類選 25 個 | `class2concepts_*.json`（原始）＋ `selected_concepts/*.json` |
| **LM4CV** | 每類詢問視覺屬性，彙整成去重後的屬性清單，供 LM4CV 後續學習屬性子集 | `cls2attributes.json` ＋ `attributes.txt` ＋ 摘要報告 |
| **CB-LLM** | 針對文字分類的每個 label 產生概念 | `cb_llm_<dataset>.json` ＋ `concepts.py` |

### 3.3 產出統計

**Label-Free CBM**（原始數量 = important / superclass / around 三種 prompt 的總和）

| Dataset | 類別 | important | superclass | around | 原始合計 | 去重後（filtered.txt） |
|---|---:|---:|---:|---:|---:|---:|
| CIFAR-10 | 10 | 102 | 101 | 100 | 303 | **283** |
| CIFAR-100 | 100 | 1,021 | 628 | 1,018 | 2,667 | **2,108** |
| CUB | 200 | 1,985 | 983 | 2,003 | 4,971 | **2,077** |
| Places365 | 365 | 4,132 | 2,027 | 4,187 | 10,346 | **6,401** |
| ImageNet | 998* | 10,204 | 5,159 | 10,142 | 25,505 | **14,278** |

**LaBo**（每類固定選 25 個概念）

| Dataset | 原始候選總數 | 平均每類候選 | 選出總數 | 選出中不重複 | 跨類重複比例 |
|---|---:|---:|---:|---:|---:|
| CIFAR-10 | 2,565 | 256.5 | 250 | 190 | 24% |
| CIFAR-100 | 24,425 | 244.2 | 2,500 | 1,698 | 32% |
| CUB | 50,520 | 252.6 | 5,000 | 1,859 | 63% |
| Places365 | 115,128 | 315.4 | 9,125 | 5,412 | 41% |
| ImageNet | 250,509 | 251.0 | 24,950 | 11,935 | 52% |

**LM4CV**

| Dataset | 類別 | 屬性總數 | 平均每類 | 不重複屬性 |
|---|---:|---:|---:|---:|
| CIFAR-10 | 10 | 115 | 11.5 | 99 |
| CIFAR-100 | 100 | 954 | 9.5 | 625 |
| CUB | 200 | 1,900 | 9.5 | 1,052 |
| Places365 | 365 | 3,453 | 9.5 | 1,815 |
| ImageNet | 998* | 9,475 | 9.5 | 3,816 |

**CB-LLM（文字）**

| Dataset | Labels | 概念總數 | 各 label 概念數 |
|---|---:|---:|---|
| SST2 | 2 | 217 | negative 104 / positive 113 |
| YelpP | 2 | 231 | negative 128 / positive 103 |
| AGNews | 4 | 341 | 75 / 75 / 98 / 93 |
| DBpedia | 10 | 675 | 54 – 96 |

\* ImageNet 類別檔有 1000 行，但有重名類別（例如兩個 `crane`、兩個 `maillot`），以 dict 儲存時合併成 998 個 key。

### 3.4 觀察

1. **Label-Free 的過濾實際上只做了去重。** `filtered.txt` 的數量和原始概念去重後完全相同（保留率 100%），代表長度／黑名單／泛用詞規則在 OpenChat 的輸出上沒有移除任何項目。真正的篩選發生在訓練階段（CLIP top-5 activation cutoff 與 interpretability cutoff）。例如 CIFAR-10 的 283 個候選概念，訓練後的 LF-CBM 保留 **281** 個（見第 5 節）。
2. **概念粒度隨資料集變細時，重複增加。** CUB 的 200 個鳥類類別共用大量相似描述（羽色、喙形），LaBo 選出的概念有 63% 跨類重複；ImageNet 也有 52%。對細粒度資料集，重複概念會降低 concept layer 的鑑別力。
3. **LM4CV 的 CIFAR-10 輸出有明顯錯誤。** `cifar10_cls2attributes.json` 中 airplane、automobile、truck 等類別的屬性都是人臉／人體特徵（例如 airplane → *bipedal gait, thin lips, prominent cheekbones*），疑似 prompt 模板或類別對應出錯；CIFAR-10 也缺少 `reports/cifar10_summary.txt`。**使用前需重新生成。** 其他資料集平均每類 9.5 個屬性，低於設定的 `attributes_per_class: 20`，表示 LLM 回應常被 `max_tokens: 150` 截斷。
4. **大小寫與措辭不一致。** Label-Free 輸出混合大小寫（`Cab`、`jet`、`Wing shape`），CB-LLM 多為標題式大寫。目前以小寫去重，但同義不同詞的概念（如 *vibrant colors* / *bright colors*）仍會同時保留。

---

## 4. CBM 訓練模組

### 4.1 Label-Free CBM（`models/lf_cbm.py`、`scripts/lf_cbm_train.py`）

流程：

1. 以凍結的 CLIP 取出影像特徵（backbone 預設 `clip_RN50`，1024 維）。
2. 讀取 `concepts/label_free/outputs/<dataset>_filtered.txt`。
3. **CLIP top-5 過濾**：每個概念取 CLIP 相似度最高的 5 張影像平均，低於 `clip_cutoff = 0.25` 的移除。
4. 以 **cos³ similarity** 訓練 projection layer `W_c`，讓 backbone 特徵投影後對齊 CLIP pseudo-label（`proj_steps = 1000`）。
5. **Interpretability cutoff**：驗證集相似度低於 `0.45` 的概念移除。
6. 以 **GLM-SAGA** 訓練稀疏 final layer（`lam = 0.0007`，`n_iters = 1000`）。

輸出：`saved_models/lf_cbm_<dataset>/`，包含 `kept_concepts.txt` 與 minimal bundle（`W_c`、`W_g`、`b_g`、concept mean/std）。

### 4.2 VLG-CBM（`models/vlg_cbm.py`、`scripts/vlg_cbm_train.py`）

流程：

1. Backbone（`clip_RN50`，使用 penultimate 特徵）＋ Concept Bottleneck Layer（CBL）。
2. 以 Grounding-DINO 產生的 concept 標註（`--annotation_dir`）做監督，BCE loss，信心門檻 `0.15`，訓練 `20` epochs（lr `5e-4`），可選擇 crop-to-concept 資料增強（p=0.5）。
3. 概念經 z-score normalization 後，以 GLM-SAGA 訓練稀疏 final layer（`lam = 7e-4`，`n_iters = 2000`），或選擇 dense head。

### 4.3 共用 Final Layer（`models/final_layer.py`、`config/final_layer_config.py`）

兩種 CBM 共用 `UnifiedFinalTrainer`，支援四種 final layer：

| 類型 | 說明 |
|---|---|
| `SPARSE_GLM` | GLM-SAGA elastic-net（LF-CBM 原始做法） |
| `DENSE_LINEAR` | 一般 dense linear，Adam 訓練 |
| `SPARSE_LINEAR` | dense 訓練後，每類保留 top-k 權重（預設 k=30） |
| `ELASTIC_NET` | 目前等同 dense（預留） |

這讓 LF-CBM 與 VLG-CBM 的差異只剩 concept layer 的訓練方式，final layer 可以在相同設定下比較。

---

## 5. 實驗結果

### 5.1 LF-CBM on CIFAR-10（`evaluation/evaluate_cbm.ipynb`）

| 項目 | 結果 |
|---|---|
| Backbone | CLIP RN50（encoder dim 1024） |
| 候選概念 | 283（`label_free/outputs/cifar10_filtered.txt`） |
| 最終保留概念 | **281** |
| Concept layer `W_c` | (281, 1024) |
| Final layer `W_g` / `b_g` | (10, 281) / (10,) |
| 推論流程驗證 | logits `[2, 10]`、concepts `[2, 281]`、probs `[2, 10]`，shape 正確 |

Notebook 確認了訓練好的 bundle 可以載入，並完成「影像 → CLIP 特徵 → 概念 → 類別」的推論。**Notebook 中的準確率、final-layer 權重分析與範例解釋等 cell 尚未執行，沒有輸出**，因此目前沒有可以報告的準確率數字。

### 5.2 尚未完成的實驗

- LF-CBM 在 CIFAR-100、CUB、Places365、ImageNet 上的訓練結果。
- VLG-CBM 的訓練結果（需要各資料集的 Grounding-DINO 標註）。
- 以 ANEC-evaluator 在 NEC = 5 / 10 / 15 / 20 / 25 / 30 下比較 LF-CBM 與 VLG-CBM。
- 以 LaBo、LM4CV 的概念集取代 Label-Free 概念集，訓練同一個 CBM，比較**概念來源**對準確率與可解釋性的影響。

---

## 6. 評估工具：ANEC

`evaluation/ANEC-evaluator/` 整合了 VLG-CBM 論文提出的 ANEC 指標：固定 **每個類別平均使用的有效概念數（NEC）**，比較各 CBM 的準確率，避免「用更多概念換準確率」造成的不公平比較。使用方式：

```bash
cd evaluation/ANEC-evaluator && pip install -e .
get_anec --load_path <activations_dir> --output_dir <results_dir>
```

輸入為 `{train,val,test}_activations.pt` 與對應 labels。工具會自動掃描 λ，找出達到目標 NEC 的稀疏 final layer。

---

## 7. 貢獻總結

1. **統一的概念生成模組**：以開源 LLM 為 Label-Free、LaBo、LM4CV、CB-LLM 四種方法產生 5 個影像資料集、4 個文字資料集的概念集。所有輸出依方法分類存放，格式一致（`concepts/<method>/outputs/`）。
2. **統一的 CBM 訓練框架**：LF-CBM 與 VLG-CBM 共用 config、資料載入與 final layer 訓練器，切換 final layer 類型只需改設定。
3. **ANEC 評估整合**：提供在相同稀疏度下公平比較 CBM 的工具。
4. **端到端驗證**：在 CIFAR-10 上完成 LF-CBM 的概念生成 → 訓練 → 推論流程（281 個概念）。

## 8. 限制與未來工作

- **修正 LM4CV CIFAR-10 概念**，並提高 `max_tokens`，讓每類屬性數接近設定值。
- **加入語意去重**（例如 sentence-transformer 相似度），處理同義概念與跨類重複。
- 補齊所有資料集的 LF-CBM / VLG-CBM 訓練，並用 ANEC 完成方法間比較。
- 比較不同 LLM（OpenChat vs. GPT 系列）產生的概念集對 CBM 表現的影響。
- `config/query_config.yaml` 目前寫死叢集內 vLLM 端點；應改為環境變數，方便在其他環境重現。

## 參考資料

- Oikarinen et al., *Label-Free Concept Bottleneck Models*, ICLR 2023 — https://github.com/Trustworthy-ML-Lab/Label-free-CBM
- Srivastava, Yan, Weng, *VLG-CBM: Training Concept Bottleneck Models with Vision-Language Guidance*, NeurIPS 2024 — https://github.com/Trustworthy-ML-Lab/VLG-CBM
- Yang et al., *Language in a Bottle (LaBo)*, CVPR 2023
- Yan et al., *Learning Concise and Descriptive Attributes for Visual Recognition (LM4CV)*, ICCV 2023
- Sun, Oikarinen, Weng, *Concept Bottleneck Large Language Models (CB-LLM)*, 2024
- ANEC-evaluator — https://github.com/windymount/ANEC-evaluator
- GLM-SAGA — https://github.com/MadryLab/glm_saga
