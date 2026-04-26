# 🧱 Concrete Compressive Strength Predictor

A weekend project: predict 28-day concrete compressive strength from mix proportions using XGBoost, served as an interactive Streamlit app.

> ⚠️ **For learning and demonstration only.** Real concrete mix design follows IS 10262 with target mean strength, trial mixes and lab verification. No predictor replaces that workflow.

## What it does

You move sliders for cement, fly ash, GGBS, water, aggregates, superplasticizer and age — the app returns:
- A **point estimate** of compressive strength (MPa)
- An **80% prediction interval** (so you see how confident the model is)
- The **closest IS 456 grade** (M20, M25, M30 …)
- **Mix diagnostics** — w/c, w/cm, SCM % replacement, density check
- **Engineering sanity warnings** when the mix violates code limits (e.g. binder > 540 kg/m³)

## The data

| Source | Rows | Notes |
|---|---|---|
| UCI Concrete Compressive Strength | ~1,030 | Yeh (1998) — academic benchmark, ages 1–365 days |
| Mumbai project data | ~117 | Real RMC mixes from my projects, ages 1–28 days |
| IS 10262 synthetic | ~500 | Generated to follow Indian-code relationships, 28-day only |
| **Total** | **~1,647** | After filtering for plausible density (2150–2600 kg/m³) |

## The model

- **XGBoost regressor** for the point prediction
- Two **quantile regressors** (α=0.1 and α=0.9) for the 80% prediction interval
- Engineered features that encode mix design ratios: total binder, w/c, w/cm, SCM %, coarse-to-fine aggregate ratio

**Test-set performance** (20% holdout, random_state=42)
- R² ≈ 0.93–0.96
- MAE ≈ 2.5–3.5 MPa
- Empirical PI coverage ≈ 78–82%

## Run it locally

```bash
git clone <this-repo>
cd concrete_predictor
pip install -r requirements.txt
python train_model.py        # trains and saves the three models (~30 sec)
streamlit run app.py
```

## What it can't see

The model treats cement as a single number, regardless of grade (OPC 43 vs OPC 53 vs PPC). It has no information on aggregate gradation, specific gravity, water absorption, curing temperature, mixing time, or admixture chemistry. All of these matter, sometimes a lot. The honest reading of a 0.96 R² on this dataset is "the model has learned the dominant relationships in *this* dataset" — not "the model knows concrete."

## Files

```
concrete_predictor/
├── app.py                 # Streamlit app
├── train_model.py         # Trains and saves the models
├── Concrete_Data_R1.xlsx  # The dataset
├── requirements.txt
└── README.md
```

After running `train_model.py`, you'll also have:
- `model_main.joblib`, `model_lo.joblib`, `model_hi.joblib` — trained models
- `metadata.json` — features, ranges, metrics

## Acknowledgements

- I-Cheng Yeh, *Modeling of strength of high-performance concrete using artificial neural networks*, Cement and Concrete Research, 1998 — the original UCI dataset
- Bureau of Indian Standards: IS 456:2000, IS 10262:2019

---

Built as a fun learning project. Feedback welcome.
