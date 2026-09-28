# Concept Generation（概念生成）

本資料夾依「概念生成模型（concept generation model）」分類：每個方法都有自己的資料夾，裡面放該方法的 querier 程式碼與它產生的 concept 檔（`outputs/`）。

```
concepts/
├── run.py                     # 互動式入口（async 版選單）
├── unified_interface.py       # 給 cbm_library 其他模組呼叫的統一介面
├── config/query_config.yaml   # LLM 與資料集設定
├── classes/                   # 各資料集類別名稱（cifar10/100, cub, places365, imagenet）
├── common/                    # 各方法共用：選單介面、LLM client、logger
│   ├── main_interface.py
│   ├── async_main_interface_test.py
│   └── utils/
├── label_free/                # Label-Free CBM (Oikarinen et al., ICLR 2023)
│   ├── label_free_querier.py
│   └── outputs/
│       ├── gpt3_init/gpt3_<dataset>_{important,superclass,around}.json
│       └── <dataset>_filtered.txt          ← LF-CBM 訓練直接讀這個
├── labo/                      # LaBo (Yang et al., CVPR 2023)
│   ├── labo_querier.py
│   └── outputs/
│       ├── concepts/class2concepts_<dataset>.json   # 原始候選概念
│       └── selected_concepts/<DATASET>.json         # submodular 篩選後（每類 25 個）
├── lm4cv/                     # LM4CV (Yan et al., ICCV 2023)
│   ├── lm4cv_querier.py
│   └── outputs/
│       ├── cls2attributes/<dataset>_cls2attributes.json
│       ├── data/<dataset>/<dataset>_attributes.txt  # 去重後屬性清單
│       └── reports/<dataset>_summary.txt
└── cb_llm/                    # CB-LLM（文字分類：SST2, YelpP, AGNews, DBpedia）
    ├── cb_llm_querier.py
    └── outputs/
        ├── cb_llm_<dataset>.json
        └── concepts.py                              # CB-LLM 原始碼可直接 import 的格式
```

每個 querier 都把結果寫到自己資料夾下的 `outputs/`（`OUTPUT_DIR = Path(__file__).parent / "outputs"`），不受目前工作目錄影響。

## 使用方式

```bash
cd concepts
python run.py            # 互動式選單，選擇方法與資料集
```

`config/query_config.yaml` 中的 `classes_file` 是相對於 `concepts/` 的路徑；`run.py` 會先 `chdir` 到這個資料夾。

各方法的產出統計與分析請見 [`docs/research_report.md`](../docs/research_report.md)。
