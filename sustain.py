"""Use a trained Z-score SuStaIn model to stage new subjects.

The input CSV must already be transformed to the same Z-score space used for
training. By default, all inputs and outputs are resolved relative to this file.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pySuStaIn


FEATURE_COLUMNS = [
    "ADC_Mean",
    "ADC_Median",
    "ADC_5th",
    "ADC_95th",
    "ADC_Skewness",
    "DCE_MER",
    "DCE_WIR",
    "DCE_WOR",
    "DCE_SER",
    "DCE_iAUC",
]

# These values must remain identical to those used to train the pickle model.
Z_MAX = np.array([3.0, 3.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 4.0, 4.0])
Z_VALS = np.array(
    [
        [1.0, 2.0, 0.0, 0.0],
        [1.0, 2.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 2.0, 0.0, 0.0],
        [1.0, 2.0, 0.0, 0.0],
        [1.0, 2.0, 0.0, 0.0],
        [1.0, 2.0, 0.0, 0.0],
        [1.0, 2.0, 0.0, 0.0],
        [1.0, 2.0, 0.0, 0.0],
        [1.0, 2.0, 0.0, 0.0],
    ]
)


def load_test_data(csv_path):
    """Load and validate the already-normalized test feature matrix."""
    if not csv_path.exists():
        raise FileNotFoundError("Test CSV not found: {}".format(csv_path))

    df = pd.read_csv(csv_path)
    missing_columns = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            "Test CSV is missing required feature columns: {}".format(missing_columns)
        )

    feature_df = df.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    invalid_mask = ~np.isfinite(feature_df.to_numpy(dtype=float))
    if invalid_mask.any():
        invalid_rows, invalid_cols = np.where(invalid_mask)
        examples = [
            "row {}, {}".format(row + 2, FEATURE_COLUMNS[col])
            for row, col in zip(invalid_rows[:10], invalid_cols[:10])
        ]
        raise ValueError(
            "Test features contain missing or non-finite values at: {}".format(
                ", ".join(examples)
            )
        )

    return df, feature_df


def load_trained_model(pickle_path):
    """Load posterior samples and verify compatibility with this script."""
    if not pickle_path.exists():
        raise FileNotFoundError("Model pickle not found: {}".format(pickle_path))

    with pickle_path.open("rb") as handle:
        model_vars = pickle.load(handle)

    required_keys = {"samples_sequence", "samples_f"}
    missing_keys = required_keys.difference(model_vars)
    if missing_keys:
        raise KeyError("Model pickle is missing keys: {}".format(sorted(missing_keys)))

    samples_sequence = np.asarray(model_vars["samples_sequence"])
    samples_f = np.asarray(model_vars["samples_f"])
    expected_events = int(np.count_nonzero(Z_VALS))

    if samples_sequence.ndim != 3:
        raise ValueError(
            "samples_sequence must be 3-D, got shape {}".format(
                samples_sequence.shape
            )
        )
    if samples_sequence.shape[1] != expected_events:
        raise ValueError(
            "Model has {} events, but current Z_VALS defines {}. "
            "Use the same Z_VALS as model training.".format(
                samples_sequence.shape[1], expected_events
            )
        )
    if samples_f.shape != (samples_sequence.shape[0], samples_sequence.shape[2]):
        raise ValueError(
            "samples_f shape {} is incompatible with samples_sequence shape {}".format(
                samples_f.shape, samples_sequence.shape
            )
        )

    return model_vars, samples_sequence, samples_f


def predict_stages(feature_df, samples_sequence, samples_f, posterior_samples, output_dir):
    """Run posterior-averaged subtype and stage inference."""
    zdata = feature_df.to_numpy(dtype=float)
    n_subtypes = samples_sequence.shape[0]
    n_mcmc_samples = samples_sequence.shape[2]
    posterior_samples = min(max(1, posterior_samples), n_mcmc_samples)

    # Training-related arguments are placeholders here: no fitting is performed.
    sustain_model = pySuStaIn.ZscoreSustain(
        zdata,
        Z_VALS,
        Z_MAX,
        FEATURE_COLUMNS,
        1,
        n_subtypes,
        1,
        str(output_dir),
        "test_inference",
        False,
        seed=42,
    )

    predictions = sustain_model.subtype_and_stage_individuals_newData(
        zdata,
        samples_sequence,
        samples_f,
        posterior_samples,
    )
    names = [
        "ml_subtype",
        "prob_ml_subtype",
        "ml_stage",
        "prob_ml_stage",
        "prob_subtype",
        "prob_stage",
        "prob_subtype_stage",
    ]
    return dict(zip(names, predictions)), posterior_samples


def save_predictions(source_df, predictions, csv_path, array_path):
    """Save a readable table and all detailed posterior probabilities."""
    output_df = source_df.copy()
    output_df["SuStaIn_Subtype"] = (
        np.asarray(predictions["ml_subtype"]).reshape(-1).astype(int) + 1
    )
    output_df["Subtype_Probability"] = np.asarray(
        predictions["prob_ml_subtype"]
    ).reshape(-1)
    output_df["SuStaIn_Stage"] = np.asarray(predictions["ml_stage"]).reshape(-1).astype(int)
    output_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    np.savez_compressed(array_path, **predictions)
    return output_df


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Predict SuStaIn subtype and stage for a normalized test CSV."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=script_dir / "test.csv",
        help="Z-score normalized test CSV (default: github/test.csv)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=script_dir / "result_new_norm_subtype0.pickle",
        help="Trained SuStaIn pickle (subtype0 means a 1-subtype model)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir,
        help="Directory for prediction outputs",
    )
    parser.add_argument(
        "--posterior-samples",
        type=int,
        default=1000,
        help="Number of evenly spaced MCMC samples used for prediction",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_df, feature_df = load_test_data(args.data)
    _, samples_sequence, samples_f = load_trained_model(args.model)

    print("Test subjects: {}".format(len(feature_df)))
    print("Features: {}".format(len(FEATURE_COLUMNS)))
    print("Model subtypes: {}".format(samples_sequence.shape[0]))

    predictions, used_samples = predict_stages(
        feature_df,
        samples_sequence,
        samples_f,
        args.posterior_samples,
        args.output_dir,
    )

    stem = args.data.stem
    csv_path = args.output_dir / "{}_sustain_predictions.csv".format(stem)
    array_path = args.output_dir / "{}_sustain_probabilities.npz".format(stem)
    output_df = save_predictions(source_df, predictions, csv_path, array_path)

    print("Posterior samples used: {}".format(used_samples))
    print("\nPrediction summary:")
    print(
        output_df[
            [
                "SuStaIn_Subtype",
                "Subtype_Probability",
                "SuStaIn_Stage",
            ]
        ].to_string(index=False)
    )
    print("\nSaved CSV: {}".format(csv_path))
    print("Saved probabilities: {}".format(array_path))


if __name__ == "__main__":
    main()
