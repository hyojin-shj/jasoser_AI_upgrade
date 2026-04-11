# Changelog

## 2026‑04‑11
- **Model Base Updated**: `model.py` BASE_MODEL switched from `gpt2` to `Bllossom/llama-3.2-Korean-Bllossom-3B` (Korean‑specialized Llama 3.2, 3 B parameters).
- **Test Dataset Expanded**: `data/test_resumes.json` now contains 40+ synthetic resume entries covering a wide range of companies, roles, and quality levels (Pass/Fail). Fields renamed to `직무`, `question1`, `answer1` for consistency.
- **Experiment Script Added**: `experiment.py` evaluates OpenAI, Gemini, base Llama, and fine‑tuned SLM on the test set, outputs a markdown table `experiment_results.md`.
- **Streamlit Multi‑Page UI**:
  - Added `pages/1_모델_평가.py` – shows the original radar/bar/line dashboard for all models.
  - Added `pages/2_SLLM_전후.py` – displays evaluation results of the SLLM **before** and **after** fine‑tuning side‑by‑side.
  - Added `pages/3_비교_결과.py` – aggregates scores from every model and visualises overall performance comparison (average scores, distribution).
- **UI Enhancements**: Consistent custom CSS, dark‑mode friendly colors, and micro‑animations retained across new pages.

## 2026‑04‑06
- Integrated MLflow logging for SLM fine‑tuning.
- Added training UI in `app.py`.

*All changes are committed to the `main` branch.*
