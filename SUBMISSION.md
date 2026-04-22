# Submission checklist

Required files (per `configs/course_config.json:submission.required_files`):

| # | File | Source | Status |
|---|---|---|---|
| 1 | `best_checkpoint/` | `train.py --output-dir <run-dir>` (Colab) | runtime |
| 2 | `configs/colab_runtime_config.json` | already committed | **ready** |
| 3 | `public_eval_bundle/public_eval.json` | `generate_public_rollout.py` + `public_eval.py` (Colab) | runtime |
| 4 | `demo_bundle/demo.mp4` | `test_policy.py` (Colab, with `--episode-length 3500`) | runtime |
| 5 | `short_report.pdf` | rendered from `REPORT.md` | **ready** (`submission/short_report.pdf`) — re-render after filling in TODOs |

Top-level deliverables called out in the assignment description:

| Item | Where |
|---|---|
| Report (≤ 5 pages) | `submission/short_report.pdf` (rendered from `REPORT.md`) |
| GitHub repo link | https://github.com/Martin-qyma/EEC289A_Robotics-Homework |
| Colab notebook | `Assignment_1_Colab.ipynb` (already in the repo, also runs in Colab via the URL above) |
| Video demonstrations | `demo_bundle/demo.mp4` (12 segments × 5 s, all directions) + `public_eval_bundle/public_eval_episode0.mp4` (forward/backward benchmark episode) |

## How to assemble the bundle

1. Run the Colab notebook end-to-end. This produces:
   - `artifacts/run_baseline/best_checkpoint/`  (the trained extended policy)
   - `artifacts/public_eval_bundle/`  (with `public_eval.json` + a benchmark video)
   - `artifacts/demo_bundle/demo.mp4`
   - `artifacts/custom_eval/`  (per-axis sweep, complementary)
2. Pull the artifacts back to your laptop (`gdown` / Colab download / `git lfs` / etc.).
3. Fill in the `_TODO_` slots in `REPORT.md` with the numbers from
   `artifacts/public_eval_bundle/public_eval.json` and
   `artifacts/custom_eval/custom_eval.json`, then re-render:
   ```bash
   python3 -m pip install --user markdown
   python3 -c "import markdown; from pathlib import Path; \
     html = '<style>body{font-family:sans-serif;font-size:10.5pt;line-height:1.45;} \
     h1{font-size:18pt;border-bottom:2px solid #333;} \
     h2{font-size:13pt;border-bottom:1px solid #ccc;} \
     code{background:#f0f0f0;padding:1px 4px;border-radius:3px;} \
     pre{background:#f6f8fa;padding:8px 10px;border-radius:4px;} \
     table{border-collapse:collapse;font-size:9.5pt;} \
     th,td{border:1px solid #bbb;padding:4px 8px;}</style>' \
     + markdown.markdown(Path('REPORT.md').read_text(), extensions=['tables','fenced_code']); \
     Path('submission/REPORT.html').write_text(html)"
   google-chrome --headless --disable-gpu --no-pdf-header-footer \
     --print-to-pdf=submission/short_report.pdf "file://$PWD/submission/REPORT.html"
   ```
4. Run the bundler:
   ```bash
   scripts/prepare_submission.sh artifacts/run_baseline
   ```
   → produces `submission/` and `submission.tar.gz`.
5. Upload `submission.tar.gz` (or the contents of `submission/`) to whatever
   place the course expects.

## What's already in `submission/` right now

- `short_report.pdf` — first render from the current `REPORT.md` (still has
  `_TODO_` slots; replace with real numbers after the Colab run).
- `REPORT.html` — intermediate file used by the PDF render.

The runtime outputs (`best_checkpoint/`, eval JSON, videos) will land here
after step 4 above.
