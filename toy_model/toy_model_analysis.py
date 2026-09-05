"""
Toy-model analyses used in the manuscript.
It reproduces the three population scenarios, PCA visualizations, linear encoding analyses, CCGP analyses, and the
category-neuron sweep.

Task structure
--------------
Context 1:
    A, C -> Left
    B, D -> Right

Context 2:
    A, C -> Right
    B, D -> Left

Abstract stimulus-pair:
    AC vs BD

Dependencies
------------
numpy
pandas
matplotlib
scikit-learn
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# =============================================================================
# Configuration
# =============================================================================

RANDOM_SEED = 1
N_REPEATS = 100
BASELINE = 5.0
NOISE_SD = 1.0

CATEGORY_SWEEP = [0, 5, 10, 15, 20, 30, 40, 60]


# =============================================================================
# 1. Task structure
# =============================================================================

def make_trials_4stim(n_repeats=N_REPEATS):
    """Create the four-stimulus, two-context task used in the toy model."""

    trial_types = [
        # Context 1
        {"stimulus": "A", "context": "C1", "response": "Left",  "stim_pair": "AC"},
        {"stimulus": "B", "context": "C1", "response": "Right", "stim_pair": "BD"},
        {"stimulus": "C", "context": "C1", "response": "Left",  "stim_pair": "AC"},
        {"stimulus": "D", "context": "C1", "response": "Right", "stim_pair": "BD"},

        # Context 2: response mapping reversed
        {"stimulus": "A", "context": "C2", "response": "Right", "stim_pair": "AC"},
        {"stimulus": "B", "context": "C2", "response": "Left",  "stim_pair": "BD"},
        {"stimulus": "C", "context": "C2", "response": "Right", "stim_pair": "AC"},
        {"stimulus": "D", "context": "C2", "response": "Left",  "stim_pair": "BD"},
    ]

    trials = [
        trial.copy()
        for trial in trial_types
        for _ in range(n_repeats)
    ]

    df = pd.DataFrame(trials)

    df["context_code"] = df["context"].map({"C1": 1, "C2": -1})
    df["response_code"] = df["response"].map({"Left": 1, "Right": -1})
    df["stim_pair_code"] = df["stim_pair"].map({"AC": 1, "BD": -1})

    for stim in ["A", "B", "C", "D"]:
        df[f"stim_{stim}"] = (df["stimulus"] == stim).astype(int)

    return df


def get_neuron_columns(df):
    """Return neural-activity columns."""
    return [column for column in df.columns if column.startswith("neuron_")]


# =============================================================================
# 2. Mixed-selectivity neural population
# =============================================================================

def generate_population(
    n_stim_dominant,
    n_response_dominant,
    n_context_dominant,
    n_noise_dominant,
    n_category_dominant=0,
    category_model=False,
    seed=RANDOM_SEED,
    n_repeats=N_REPEATS,
    baseline=BASELINE,
    noise_sd=NOISE_SD,
):
    """
    Generate a heterogeneous mixed-selectivity neural population.

    Each neuron receives contributions from stimulus identity, response,
    AC-vs-BD category, and context. The dominant neuronal class determines
    which weight is strongly enhanced.

    For scenarios without category-dominant neurons, the category contribution
    is omitted to reproduce the original notebook implementation.
    """

    rng = np.random.RandomState(seed)
    df = make_trials_4stim(n_repeats=n_repeats)

    neuron_types = (
        ["stimulus_dominant"] * n_stim_dominant
        + ["response_dominant"] * n_response_dominant
        + ["category_dominant"] * n_category_dominant
        + ["context_dominant"] * n_context_dominant
        + ["noise_dominant"] * n_noise_dominant
    )

    n_neurons = len(neuron_types)
    activity = np.zeros((len(df), n_neurons))
    weights = []

    stim_matrix = df[["stim_A", "stim_B", "stim_C", "stim_D"]].to_numpy()

    include_category_term = category_model

    for neuron_idx, neuron_type in enumerate(neuron_types):
        w_stim = rng.normal(0.05, 0.05, size=4)
        w_response = rng.normal(0.05, 0.05)
        if include_category_term:
            # Preserve the exact random-draw order of the original
            # category-inclusive notebook generator.
            w_category = rng.normal(0.05, 0.05)
            w_context = rng.normal(0.05, 0.05)
        else:
            w_category = 0.0
            w_context = rng.normal(0.05, 0.05)

        if neuron_type == "stimulus_dominant":
            preferred_stim = rng.choice(4)
            w_stim = rng.normal(0.2, 0.05, size=4)
            w_stim[preferred_stim] = rng.normal(2.5, 0.3)

        elif neuron_type == "response_dominant":
            w_response = rng.normal(2.5, 0.3)
            w_stim = rng.normal(0.2, 0.05, size=4)

        elif neuron_type == "category_dominant":
            w_category = rng.normal(2.5, 0.3)
            w_stim = rng.normal(0.2, 0.05, size=4)
            w_response = rng.normal(0.2, 0.05)

        elif neuron_type == "context_dominant":
            w_context = rng.normal(2.5, 0.3)
            w_stim = rng.normal(0.2, 0.05, size=4)
            w_response = rng.normal(0.2, 0.05)
            if include_category_term:
                w_category = rng.normal(0.2, 0.05)

        elif neuron_type == "noise_dominant":
            pass

        signal = (
            stim_matrix @ w_stim
            + w_response * df["response_code"].to_numpy()
            + w_context * df["context_code"].to_numpy()
        )

        if include_category_term:
            signal = signal + w_category * df["stim_pair_code"].to_numpy()

        activity[:, neuron_idx] = (
            baseline
            + signal
            + rng.normal(0, noise_sd, size=len(df))
        )

        weight_row = {
            "neuron": f"neuron_{neuron_idx + 1}",
            "type": neuron_type,
            "w_stim_A": w_stim[0],
            "w_stim_B": w_stim[1],
            "w_stim_C": w_stim[2],
            "w_stim_D": w_stim[3],
            "w_response": w_response,
            "w_context": w_context,
        }

        if include_category_term:
            weight_row["w_category_AC_BD"] = w_category

        weights.append(weight_row)

    neural_df = pd.DataFrame(
        activity,
        columns=[f"neuron_{idx + 1}" for idx in range(n_neurons)],
    )

    df_full = pd.concat(
        [df.reset_index(drop=True), neural_df],
        axis=1,
    )

    return df_full, pd.DataFrame(weights)


# =============================================================================
# 3. Linear encoding analysis
# =============================================================================

def encoding_analysis(df_full, include_category=False):
    """
    Fit a neuron-wise linear encoding model.

    Predictors without category term:
        stimulus A/B/C/D, response, context

    Predictors with category term:
        stimulus A/B/C/D, response, AC-vs-BD category, context

    Predictors are z-scored before fitting, matching the final notebook.
    """

    neuron_cols = get_neuron_columns(df_full)

    predictor_cols = [
        "stim_A",
        "stim_B",
        "stim_C",
        "stim_D",
        "response_code",
    ]

    if include_category:
        predictor_cols.append("stim_pair_code")

    predictor_cols.append("context_code")

    X_design = StandardScaler().fit_transform(
        df_full[predictor_cols].to_numpy()
    )

    results = []

    for neuron in neuron_cols:
        y = df_full[neuron].to_numpy()

        model = LinearRegression()
        model.fit(X_design, y)

        coefficients = model.coef_
        r2 = model.score(X_design, y)

        result = {
            "neuron": neuron,
            "stimulus_beta": np.mean(np.abs(coefficients[:4])),
            "response_beta": abs(coefficients[4]),
            "R2": r2,
        }

        if include_category:
            result["category_beta"] = abs(coefficients[5])
            result["context_beta"] = abs(coefficients[6])
        else:
            result["context_beta"] = abs(coefficients[5])

        results.append(result)

    results_df = pd.DataFrame(results)

    summary = {
        "stimulus_beta": results_df["stimulus_beta"].mean(),
        "response_beta": results_df["response_beta"].mean(),
    }

    if include_category:
        summary["category_beta"] = results_df["category_beta"].mean()

    summary["context_beta"] = results_df["context_beta"].mean()
    summary["R2"] = results_df["R2"].mean()

    return results_df, summary


# =============================================================================
# 5. Cross-condition generalization performance (CCGP)
# =============================================================================

def train_test_decode(df_full, train_mask, test_mask, label_col):
    """
    Train a linear SVM on one condition and test on another condition.

    This reproduces the classifier used in the final notebook:
    StandardScaler -> LinearSVC(C=1.0).
    """

    neuron_cols = get_neuron_columns(df_full)

    X_train = df_full.loc[train_mask, neuron_cols].to_numpy()
    y_train = df_full.loc[train_mask, label_col].to_numpy()

    X_test = df_full.loc[test_mask, neuron_cols].to_numpy()
    y_test = df_full.loc[test_mask, label_col].to_numpy()

    classifier = make_pipeline(
        StandardScaler(),
        LinearSVC(C=1.0, max_iter=10000),
    )

    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    return accuracy_score(y_test, predictions)


def bidirectional_ccgp(
    df_full,
    label_col,
    train_mask_forward,
    test_mask_forward,
):
    """Compute CCGP in both directions and return both accuracies and their mean."""

    forward = train_test_decode(
        df_full,
        train_mask_forward,
        test_mask_forward,
        label_col,
    )

    reverse = train_test_decode(
        df_full,
        test_mask_forward,
        train_mask_forward,
        label_col,
    )

    return forward, reverse, np.mean([forward, reverse])


def run_all_ccgp(df_full):
    """Compute context, AC-vs-BD, and response CCGP."""

    ab_mask = df_full["stimulus"].isin(["A", "B"])
    cd_mask = df_full["stimulus"].isin(["C", "D"])
    c1_mask = df_full["context"] == "C1"
    c2_mask = df_full["context"] == "C2"

    context_forward, context_reverse, context_mean = bidirectional_ccgp(
        df_full,
        label_col="context_code",
        train_mask_forward=ab_mask,
        test_mask_forward=cd_mask,
    )

    pair_forward, pair_reverse, pair_mean = bidirectional_ccgp(
        df_full,
        label_col="stim_pair_code",
        train_mask_forward=c1_mask,
        test_mask_forward=c2_mask,
    )

    response_forward, response_reverse, response_mean = bidirectional_ccgp(
        df_full,
        label_col="response_code",
        train_mask_forward=ab_mask,
        test_mask_forward=cd_mask,
    )

    return {
        "Context CCGP AB->CD": context_forward,
        "Context CCGP CD->AB": context_reverse,
        "Context CCGP mean": context_mean,
        "AC-vs-BD CCGP C1->C2": pair_forward,
        "AC-vs-BD CCGP C2->C1": pair_reverse,
        "AC-vs-BD CCGP mean": pair_mean,
        "Response CCGP AB->CD": response_forward,
        "Response CCGP CD->AB": response_reverse,
        "Response CCGP mean": response_mean,
    }


# =============================================================================
# 6. Scenario analyses
# =============================================================================

def analyze_scenario(
    scenario_name,
    population_kwargs,
    include_category,
    output_dir,
):
    """Generate one population, run all analyses, and save numerical outputs."""

    scenario_dir = Path(output_dir) / scenario_name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    df_full, weights = generate_population(**population_kwargs)

    encoding_results, encoding_summary = encoding_analysis(
        df_full,
        include_category=include_category,
    )

    ccgp_results = run_all_ccgp(df_full)

    weights.to_csv(scenario_dir / "neuron_weights.csv", index=False)
    encoding_results.to_csv(
        scenario_dir / "encoding_by_neuron.csv",
        index=False,
    )

    pd.DataFrame(
        [encoding_summary]
    ).to_csv(
        scenario_dir / "encoding_summary.csv",
        index=False,
    )

    pd.DataFrame(
        [ccgp_results]
    ).to_csv(
        scenario_dir / "ccgp_results.csv",
        index=False,
    )

    return {
        "data": df_full,
        "weights": weights,
        "encoding_results": encoding_results,
        "encoding_summary": encoding_summary,
        "ccgp_results": ccgp_results,
    }


# =============================================================================
# 7. Category-neuron sweep
# =============================================================================

def run_category_sweep(category_values=CATEGORY_SWEEP):
    """
    Vary the number of category-dominant neurons while holding the other
    subpopulations fixed.
    """

    sweep_results = []

    for n_category in category_values:
        df_full, _ = generate_population(
            n_stim_dominant=15,
            n_response_dominant=15,
            n_category_dominant=n_category,
            category_model=True,
            n_context_dominant=60,
            n_noise_dominant=10,
            seed=RANDOM_SEED,
        )

        # The original final notebook uses the category-inclusive encoding
        # model throughout the sweep, including the n_category=0 condition.
        encoding_results, encoding_summary = encoding_analysis(
            df_full,
            include_category=True,
        )

        ccgp_results = run_all_ccgp(df_full)

        sweep_results.append(
            {
                "n_category_neurons": n_category,
                "stimulus_beta": encoding_summary["stimulus_beta"],
                "response_beta": encoding_summary["response_beta"],
                "category_beta": encoding_summary["category_beta"],
                "context_beta": encoding_summary["context_beta"],
                "mean_R2": encoding_summary["R2"],
                "context_CCGP": ccgp_results["Context CCGP mean"],
                "AC_vs_BD_CCGP": ccgp_results["AC-vs-BD CCGP mean"],
                "response_CCGP": ccgp_results["Response CCGP mean"],
            }
        )

    return pd.DataFrame(sweep_results)


def save_category_sweep_plots(sweep_df, output_dir):
    """
    Generate the two figures used for the category-neuron sweep.

    Figure 1A:
        Category encoding coefficient beta as a function of
        category-selective neuron number.

    Figure 1B:
        AC-vs-BD CCGP as a function of category-selective neuron number.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Figure 1A: Category encoding
    # ------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(4, 3))

    ax.plot(
        sweep_df["n_category_neurons"],
        sweep_df["category_beta"],
        marker="o",
        linewidth=2
    )

    ax.set_xlabel("Number of category-selective neurons")
    ax.set_ylabel("Category encoding coefficient beta")
    ax.set_title("A. Category encoding")

    fig.tight_layout()

    fig.savefig(
        output_dir / "Figure1A_category_encoding.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    # ------------------------------------------------------------
    # Figure 1B: AC-vs-BD CCGP
    # ------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(4, 3))

    ax.plot(
        sweep_df["n_category_neurons"],
        sweep_df["AC_vs_BD_CCGP"],
        marker="o",
        linewidth=2
    )

    ax.axhline(
        0.5,
        linestyle="--",
        linewidth=1,
        label="Chance level"
    )

    ax.set_xlabel("Number of category-selective neurons")
    ax.set_ylabel("AC-vs-BD CCGP")
    ax.set_title("B. Stimulus-pair CCGP")

    ax.legend(frameon=False)

    fig.tight_layout()

    fig.savefig(
        output_dir / "Figure1B_category_CCGP.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# =============================================================================
# 8. Main reproducible analysis
# =============================================================================

def main():
    output_dir = Path("toy_model_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = {
        "scenario_1_context_dominated": {
            "population_kwargs": {
                "n_stim_dominant": 1,
                "n_response_dominant": 1,
                "n_category_dominant": 0,
                "category_model": False,
                "n_context_dominant": 60,
                "n_noise_dominant": 38,
                "seed": RANDOM_SEED,
            },
            "include_category": False,
        },
        "scenario_2_increased_stimulus_response": {
            "population_kwargs": {
                "n_stim_dominant": 15,
                "n_response_dominant": 15,
                "n_category_dominant": 0,
                "category_model": False,
                "n_context_dominant": 60,
                "n_noise_dominant": 10,
                "seed": RANDOM_SEED,
            },
            "include_category": False,
        },
        "scenario_3_added_category_neurons": {
            "population_kwargs": {
                "n_stim_dominant": 15,
                "n_response_dominant": 15,
                "n_category_dominant": 15,
                "category_model": True,
                "n_context_dominant": 60,
                "n_noise_dominant": 10,
                "seed": RANDOM_SEED,
            },
            "include_category": True,
        },
    }

    scenario_results = {}

    for scenario_name, config in scenarios.items():
        scenario_results[scenario_name] = analyze_scenario(
            scenario_name=scenario_name,
            population_kwargs=config["population_kwargs"],
            include_category=config["include_category"],
            output_dir=output_dir,
        )

    comparison_rows = []

    for scenario_name, result in scenario_results.items():
        encoding = result["encoding_summary"]
        ccgp = result["ccgp_results"]

        comparison_rows.append(
            {
                "scenario": scenario_name,
                "stimulus_beta": encoding["stimulus_beta"],
                "response_beta": encoding["response_beta"],
                "category_beta": encoding.get("category_beta", np.nan),
                "context_beta": encoding["context_beta"],
                "mean_R2": encoding["R2"],
                "context_CCGP": ccgp["Context CCGP mean"],
                "AC_vs_BD_CCGP": ccgp["AC-vs-BD CCGP mean"],
                "response_CCGP": ccgp["Response CCGP mean"],
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(
        output_dir / "scenario_comparison.csv",
        index=False,
    )

    sweep_df = run_category_sweep()
    sweep_df.to_csv(
        output_dir / "category_neuron_sweep.csv",
        index=False,
    )
    save_category_sweep_plots(sweep_df, output_dir)

    print("\nScenario comparison")
    print(comparison_df.to_string(index=False))

    print("\nCategory-neuron sweep")
    print(sweep_df.to_string(index=False))


if __name__ == "__main__":
    main()
