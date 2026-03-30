# New Chat Handoff — Nemotron Reasoning Challenge

## Objective
Win (or place highly in) the NVIDIA Nemotron Model Reasoning Challenge on Kaggle.
$106,388 prize pool. Deadline: 2026-06-15. 1,063 teams registered.

## Current Status
- **Phase 0: Intelligence Gathering** — COMPLETE
- **Phase 1: Data Engine** — PARTIALLY COMPLETE (base JSONL + cipher CoTs done)
- **Phase 2: LoRA Training** — READY (notebooks rewritten, identity submission next)

## Platform Architecture (2026-03-29 pivot)

**Previous approach**: Colab with model on Google Drive — FAILED (model was never on Drive, 60GB impractical to upload).

**Current approach**: Two-notebook split.

1. **`notebooks/train_lora.ipynb`** — runs on **Colab A100 (High-RAM 80GB)**
   - Model via `kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")`
   - bf16 LoRA (default) or QLoRA 4-bit fallback (`USE_QLORA` flag)
   - Adapter saved to Google Drive for persistence
   - Kaggle API token via Colab Secret `KAGGLE_API_TOKEN`

2. **`notebooks/submit.ipynb`** — runs on **Kaggle Notebooks**
   - Copies adapter from attached Kaggle Dataset → zips → `submission.zip`
   - Identity LoRA cell for pipeline validation (no training)
   - No GPU needed for trained adapter path

**Submission flow**: Train on Colab → download adapter.zip → upload as Kaggle Dataset → run submit.ipynb → score

## Key Facts (Verified from Metric Source Code)
- Submit: LoRA adapter (rank <= 32) → loaded onto Nemotron-3-Nano-30B via vLLM
- Metric: accuracy = num_correct / total (binary per problem)
- Answers: extracted from `\boxed{}` in model output
- Inference: temperature=1.0, top_p=1.0, max_tokens=3584, enable_thinking=True
- Competition eval GPU: RTX PRO 6000 Blackwell (96GB VRAM)

## Data Ready
- `data/train.csv` — 9,500 competition problems
- `data/train_formatted.jsonl` — 8,075 SFT examples with CoT reasoning
- `data/val_formatted.jsonl` — 1,425 validation examples
- `data/cipher_cot_gpt41.jsonl` — GPT-4.1 cipher CoTs (merged into formatted)

## Colab Constraints
- Education Pro: ~100 CU/month (~6-7 hrs A100 time)
- 12hr session limit (no background execution)
- Must select High-RAM for 80GB A100
- Kaggle API token at `/Volumes/WD_BLACK/Desk2/Projects/Keys/kaggle.json`

## Next Actions
1. Submit identity LoRA via submit.ipynb to validate pipeline
2. Run baseline eval (~500 problems) on Colab A100
3. First LoRA training run on Colab
4. Compare baseline vs trained, iterate

## Key Sources
- Competition demo notebook: `/tmp/nemotron-demo/nvidia-nemotron-submission-demo.ipynb`
- Competition metric: `/tmp/nemotron-metric/nvidia-nemotron-metric.ipynb`
- Strategy: `docs/STRATEGY.md`
- Knowledge base: `docs/KNOWLEDGE_BASE.md`

_Updated: 2026-03-29_
