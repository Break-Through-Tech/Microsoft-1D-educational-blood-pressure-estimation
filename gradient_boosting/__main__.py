"""One entry point: python -m gradient_boosting --help."""

import argparse
import numpy as np

from gradient_boosting import data, experiments, models, report


def main():
    parser = argparse.ArgumentParser(description="PPG features -> gradient boosting -> SBP/DBP")
    parser.add_argument("--config", default=data.CONFIG, help="Use a separate config for a new run")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("run", help="Prepare, train with chosen settings, test, and write the report")
    commands.add_parser("prepare", help="Prepare training/validation data; reuse checked caches")
    commands.add_parser("train", help="Train mean/median, Ridge and boosting; save before testing")
    commands.add_parser("test", help="Evaluate the saved model, or verify existing test results")
    commands.add_parser("verify", help="Reproduce saved predictions and check raw-signal samples")
    html = commands.add_parser("report", help="Write a short HTML report from saved results")
    html.add_argument("--plots", action="store_true", help="Also draw prediction/residual plots from local data")
    experiment = commands.add_parser("experiment", help="Read or run one of the two validation experiments")
    experiment.add_argument("name", choices=("features", "tuning"))
    experiment.add_argument("--rerun", action="store_true", help="Save a separate validation-only rerun; keep the final model unchanged")
    predict = commands.add_parser("predict", help="Predict from a raw PPG .npy array, shape (n,625), without ABP")
    predict.add_argument("input")
    predict.add_argument("--output", required=True, help="New .npz file inside data/derived/gradient_boosting/")
    args = parser.parse_args()
    config = data.load_config(args.config)

    if args.command in ("run", "prepare"):
        source = data.source_report(config, create=True)
        for split in ("train", "validation"):
            data.prepare_split(config, split, source)
    if args.command in ("run", "train"):
        models.train(config)
    if args.command in ("run", "test"):
        models.test(config)
    if args.command in ("run", "report"):
        report.write_report(config, plots=args.command == "run" or args.plots)
    if args.command == "verify":
        for split in ("validation", "test"):
            models.verify_predictions(config, split)
        for split in ("train", "validation", "test"):
            data.verify_raw_samples(config, split)
        print("All checks passed. No models were fitted or tuned.")
    if args.command == "experiment":
        result = experiments.run_experiment(config, args.name, args.rerun)
        for row in result["rows"]:
            print(f"{row['features']}, {row['parameters']}, {row['target'].upper()}: macro MAE {row['macro_mae']:.4f}")
        print("Validation only; the saved final model stays unchanged.")
    if args.command == "predict":
        filename = data.output_path(args.output)
        predicted = models.predict_ppg(models.load_model(config), np.load(args.input, allow_pickle=False), config)
        filename.parent.mkdir(parents=True, exist_ok=True)
        models.save_predictions(filename, predicted)
        print(f"Saved PPG-only predictions to {filename}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        raise SystemExit(f"Error: {error}")
