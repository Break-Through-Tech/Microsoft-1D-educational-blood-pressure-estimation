"""A short HTML explanation of the three experiments and their results."""

import base64
import html
import json
from pathlib import Path

from gradient_boosting import data, models

SNAPSHOT = Path(__file__).with_name("results.json")
HTML = Path(__file__).with_name("results.html")
LABELS = {"training_mean": "Always predict training mean", "training_median": "Always predict training median",
          "ridge": "Ridge (linear model)", "gradient_boosting": "Gradient boosting"}


def number(value):
    if value is None:
        return "Unavailable"
    return f"{value:.2f}" if isinstance(value, float) else html.escape(str(value))


def table(rows, columns):
    header = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body = "".join("<tr>" + "".join(f"<td>{number(row.get(key))}</td>" for key, _ in columns) + "</tr>" for row in rows)
    return f"<div class='table'><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>"


def summary_from_run(config):
    folder = data.run_dir(config)
    if not (folder / "test_results.json").exists():
        return json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    info = json.loads((folder / "model_info.json").read_text())
    summary = {"settings": {**config["gradient_boosting"], "feature_count": 17, "ridge_alpha": config["ridge_alpha"]},
               "features": [], "tuning": [], "source": "Original model and predictions reused; reorganizing did not retrain or reselect them."
               if info.get("imported_original") else "Saved model and predictions; this report does not train or select models."}
    for split in ("validation", "test"):
        result = json.loads((folder / f"{split}_results.json").read_text())
        summary[split] = {key: result[key] for key in ("coverage", "metrics", "pressure_ranges", "intervals", "ordering")}
    for name in ("features", "tuning"):
        filename = folder / "experiments" / (name + ".json")
        if filename.exists():
            rows = json.loads(filename.read_text())["rows"]
            for order in sorted({row["order"] for row in rows}):
                pair = {row["target"]: row for row in rows if row["order"] == order}
                first = pair["sbp"]
                summary[name].append({"features": "6 summary features" if first["features"] == "six_features" else "17 features (summary + pulse shape)",
                    **first["parameters"], "sbp_mae": pair["sbp"]["macro_mae"], "dbp_mae": pair["dbp"]["macro_mae"]})
    return summary


def render(summary, plots=None):
    test, validation, settings = summary["test"], summary["validation"], summary["settings"]
    rows = []
    for name in models.MODEL_NAMES:
        scores = {r["target"]: r for r in test["metrics"] if r["model"] == name}
        rows.append({"model": LABELS[name], "sbp": scores["sbp"]["macro_mae"], "dbp": scores["dbp"]["macro_mae"]})
    coverage = test["coverage"]
    feature_rows = summary["features"]
    feature_conclusion = "No feature comparison is saved for this output yet. The 17-feature default comes from the original validation experiment."
    if len(feature_rows) == 2:
        basic = next(r for r in feature_rows if r["features"].startswith("6 "))
        full = next(r for r in feature_rows if r["features"].startswith("17 "))
        feature_conclusion = "The pulse-shape features lowered both errors in this comparison. I kept all 17 features." if all(
            full[t] < basic[t] for t in ("sbp_mae", "dbp_mae")) else "Extra pulse-shape features did not consistently lower both errors in this comparison. Keep the basic features as a useful reference."
    tuning_conclusion = f"The configured model uses learning rate {settings['learning_rate']} and {settings['max_leaf_nodes']} leaves, with {settings['max_iter']} iterations."
    if summary["tuning"]:
        best = [min(summary["tuning"], key=lambda r: r[target]) for target in ("sbp_mae", "dbp_mae")]
        tuning_conclusion += f" Both validation targets favored {best[0]['learning_rate']} and {best[0]['max_leaf_nodes']} leaves." if best[0] == best[1] else " The targets favored different settings in this validation comparison."
    gb = rows[-1]
    beats_all = all(gb[t] is not None and all(row[t] is not None and gb[t] < row[t] for row in rows[:-1]) for t in ("sbp", "dbp"))
    model_conclusion = "Boosting has the lowest overall errors in this comparison." if beats_all else "Boosting does not beat every comparison model on both targets in this run. Keep the baselines visible."
    detail = [{"model": LABELS[r["model"]], "target": r["target"].upper(), **{k: r[k] for k in
               ("pooled_mae", "pooled_rmse", "macro_rmse", "macro_mean_error")}} for r in test["metrics"]]
    val_rows = [{"model": LABELS[r["model"]], "target": r["target"].upper(), "mae": r["macro_mae"]} for r in validation["metrics"]]
    ranges = [{**r, "target": r["target"].upper(), "band": r["band"].replace("_", " ")} for r in test["pressure_ranges"]]
    intervals = [{**r, "baseline": LABELS[r["baseline"]], "target": r["target"].upper()} for r in test["intervals"]]
    pictures = "".join(f"<h3>{name.title()}</h3><img src='{html.escape(p)}' alt='{name}'>" for name, p in (plots or {}).items())
    return ("<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
        "<title>Gradient boosting: what I tried and learned</title><style>"
        "body{font:16px/1.55 system-ui,sans-serif;max-width:980px;margin:35px auto;padding:0 20px;color:#23303b;background:#fafafa}"
        "h1{font-size:28px}h2{font-size:22px;margin-top:32px}h3{font-size:16px;margin-bottom:3px}p{margin:7px 0 15px}"
        "table{border-collapse:collapse;width:100%;background:white}th,td{text-align:left;border-bottom:1px solid #ddd;padding:9px}"
        "th{background:#edf2f6}.table{overflow:auto}details{margin:18px 0;padding:12px;background:#eef3f5}"
        "summary{cursor:pointer;font-weight:600}.conclusion{border-left:4px solid #39745b;padding:9px 14px;background:#edf5ee}"
        "img{max-width:100%;margin:12px 0}code{font-size:14px}small{color:#52616e}</style>"
        "<h1>Gradient boosting: what I tried and learned</h1>"
        "<p>I use five seconds of PPG to estimate SBP (the upper BP number) and DBP (the lower number). "
        "ABP provides the training labels; prediction itself needs only PPG.</p>"
        "<p><b>How to read the scores:</b> MAE is the average absolute error in mmHg. Lower is better. "
        "The main tables give each recording equal weight, so long recordings do not dominate.</p>"
        "<section><h2>1. Do extra pulse features help?</h2><h3>What I tried</h3>"
        "<p>Compared six basic signal summaries with 17 features that also describe pulse timing, width, area and slopes. "
        "Both use the same training/validation windows and boosting settings.</p>"
        + table(summary["features"], (("features", "Input features"), ("sbp_mae", "Validation SBP MAE"), ("dbp_mae", "Validation DBP MAE")))
        + f"<h3>What I learned</h3><p class='conclusion'>{feature_conclusion}</p></section>"
        "<section><h2>2. Which settings worked best?</h2><h3>What I tried</h3>"
        "<p>Tested learning rates 0.05 and 0.1, with 15 or 31 leaves. Each target was selected using validation MAE, "
        "with every recording weighted equally. The test set was not used to choose settings.</p>"
        + table(summary["tuning"], (("learning_rate", "Learning rate"), ("max_leaf_nodes", "Leaves"),
                                  ("sbp_mae", "Validation SBP MAE"), ("dbp_mae", "Validation DBP MAE")))
        + f"<h3>What I learned</h3><p class='conclusion'>{tuning_conclusion} Ridge uses alpha {settings['ridge_alpha']:g}.</p></section>"
        "<section><h2>3. Does learning beat a simple guess?</h2><h3>What I tried</h3>"
        "<p>Compared boosting with a linear Ridge model and two models that always return a training-data constant. "
        "Models were trained on training data, saved, and then evaluated on the reserved test recordings.</p>"
        + table(rows, (("model", "Model"), ("sbp", "Test SBP MAE"), ("dbp", "Test DBP MAE")))
        + f"<p>Scored {coverage['retained_windows']:,} of {coverage['source_windows']:,} test windows from "
        f"{coverage['retained_recordings']:,} recordings. Predictions also exist for usable PPG windows without valid "
        "ABP labels; those windows cannot receive an error score.</p>"
        f"<h3>What I learned</h3><p class='conclusion'>{model_conclusion} Check the pressure ranges below: "
        "lower overall MAE does not guarantee reliable high-pressure predictions.</p>"
        "<details><summary>Validation scores, other error metrics and uncertainty</summary>"
        "<p>Pooled MAE averages all windows. Macro MAE gives each recording equal weight. RMSE penalizes larger errors more.</p>"
        + table(val_rows, (("model", "Model"), ("target", "Target"), ("mae", "Validation macro MAE")))
        + table(detail, (("model", "Model"), ("target", "Target"), ("pooled_mae", "Test pooled MAE"),
                         ("pooled_rmse", "Pooled RMSE"), ("macro_rmse", "Macro RMSE"), ("macro_mean_error", "Mean signed error, macro")))
        + "<p>Negative differences below favor boosting. These are exploratory 95% recording-bootstrap intervals, "
        "conditional on the fitted models. They do not account for unknown patients shared across recordings, "
        "training variability or multiple comparisons.</p>"
        + table(intervals, (("baseline", "Compared with"), ("target", "Target"), ("delta_mae", "Change in MAE"),
                            ("ci_lower", "Lower bound"), ("ci_upper", "Upper bound")))
        + "</details><details><summary>Pressure ranges, coverage and plots</summary>"
        + table(ranges, (("target", "Target"), ("band", "Reference range"), ("source_windows", "Source windows"),
                         ("retained_windows", "Scored windows"), ("macro_mae", "Boosting MAE")))
        + ("<p>The dashed line shows perfect predictions. A wide spread means larger errors. "
           "In the residual plot, points above zero mean overestimation; below zero mean underestimation.</p>" if pictures else "")
        + pictures + "</details></section>"
        "<h2>Limits to remember</h2><ul><li>Recordings are separated, but different recordings may come from the same person.</li>"
        "<li>Reference labels can contain artifacts. Low/high-pressure errors need separate checks.</li>"
        "<li>These results do not establish new-patient or new-device performance. EDA preceded splitting; validation informed development.</li>"
        "<li>The test has now been evaluated. Later model changes are new exploratory experiments.</li></ul>"
        f"<p><small>{html.escape(summary['source'])} See README.md for commands and results.json for the original numeric summary.</small></p></html>")


def plot_predictions(config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    payload = models.load_predictions(config, "test")
    keep = payload["reference_valid"]
    truth, predicted = payload["truth"][keep], payload["predictions"][3, keep]
    outputs = {}
    for residual in (False, True):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        for t, name in enumerate(models.TARGETS):
            axes[t].scatter(truth[:, t], predicted[:, t] - truth[:, t] if residual else predicted[:, t], s=2, alpha=.15)
            bounds = [float(truth[:, t].min()), float(truth[:, t].max())]
            axes[t].plot(bounds, [0, 0] if residual else bounds, "k--", linewidth=1)
            axes[t].set(title=name.upper(), xlabel="Reference (mmHg)",
                        ylabel="Prediction - reference (mmHg)" if residual else "Prediction (mmHg)")
        filename = data.run_dir(config) / ("residuals.png" if residual else "predictions.png")
        fig.savefig(filename, dpi=120)
        plt.close(fig)
        outputs[filename.stem] = filename
    return outputs


def write_report(config, filename=HTML, plots=False):
    summary = summary_from_run(config)
    pictures = plot_predictions(config) if plots else {}
    embedded = {name: "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode("ascii")
                for name, p in pictures.items()}
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    Path(filename).write_text(render(summary, embedded), encoding="utf-8")
    print(f"Open {Path(filename).resolve()}", flush=True)
    return summary
