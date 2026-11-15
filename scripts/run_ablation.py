import os
import shutil
import matplotlib.pyplot as plt
import numpy as np

from src.config import Config
from src.experiments import ExperimentRunner


def preseed_existing_dadr_checkpoints(seeds):
    # linking existing main benchmark dadr s3 checkpoints to avoid redundant runs
    ckpt_dir = os.path.join(Config.RESULTS_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    for seed in seeds:
        main_ckpt = os.path.join(ckpt_dir, f"main_cicids2017_S3_DADR_seed{seed}.json")
        if os.path.exists(main_ckpt):
            for tag in (
                f"ablation_budget_cicids2017_S3_B5000_seed{seed}.json",
                f"ablation_ratio_cicids2017_S3_A70_N30_seed{seed}.json",
            ):
                dst = os.path.join(ckpt_dir, tag)
                if not os.path.exists(dst):
                    shutil.copy2(main_ckpt, dst)


def plot_ablation_figure(results_dict, repo_fig_dir, drive_fig_dir):
    # generating a three-panel ablation study visualization across budget, pool ratio, and psi threshold
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
    })

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), dpi=300)

    # plotting memory budget sensitivity
    df_b = results_dict["budget"]
    x_b = np.arange(len(df_b))
    ax_b_twin = axes[0].twinx()
    l1 = axes[0].errorbar(
        x_b, df_b["final_cum_macro_f1_mean"], yerr=df_b["final_cum_macro_f1_std"],
        color="#1b9e77", marker="o", linewidth=2.2, capsize=3, label="Macro F1 (Left)"
    )
    l2 = ax_b_twin.errorbar(
        x_b, df_b["mean_cum_false_alerts_10k_mean"], yerr=df_b["mean_cum_false_alerts_10k_std"],
        color="#d95f02", marker="s", linestyle="--", linewidth=2.0, capsize=3, label="False Alerts/10k (Right)"
    )
    axes[0].set_xticks(x_b)
    axes[0].set_xticklabels([str(int(b)) for b in df_b["budget"]])
    axes[0].set_title("(a) Memory Budget Sensitivity", pad=12)
    axes[0].set_xlabel("Total Replay Buffer Capacity")
    axes[0].set_ylabel("Final Cumulative Macro F1", color="#1b9e77")
    ax_b_twin.set_ylabel("False Alerts / 10k Flows", color="#d95f02")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].spines["top"].set_visible(False)
    ax_b_twin.spines["top"].set_visible(False)
    axes[0].legend([l1, l2], ["Macro F1 (Left)", "False Alerts/10k (Right)"], loc="best", fontsize=8.5)

    # plotting attack vs normal pool ratio sensitivity
    df_r = results_dict["ratio"]
    x_r = np.arange(len(df_r))
    labels_r = [f"{int(a*100)}/{int(n*100)}" for a, n in zip(df_r["attack_ratio"], df_r["normal_ratio"])]
    ax_r_twin = axes[1].twinx()
    l3 = axes[1].errorbar(
        x_r, df_r["final_cum_macro_f1_mean"], yerr=df_r["final_cum_macro_f1_std"],
        color="#1b9e77", marker="o", linewidth=2.2, capsize=3, label="Macro F1 (Left)"
    )
    l4 = ax_r_twin.errorbar(
        x_r, df_r["mean_cum_false_alerts_10k_mean"], yerr=df_r["mean_cum_false_alerts_10k_std"],
        color="#d95f02", marker="s", linestyle="--", linewidth=2.0, capsize=3, label="False Alerts/10k (Right)"
    )
    axes[1].set_xticks(x_r)
    axes[1].set_xticklabels(labels_r)
    axes[1].set_title("(b) Attack / Normal Pool Ratio", pad=12)
    axes[1].set_xlabel("Attack% / Normal% Split")
    axes[1].set_ylabel("Final Cumulative Macro F1", color="#1b9e77")
    ax_r_twin.set_ylabel("False Alerts / 10k Flows", color="#d95f02")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].spines["top"].set_visible(False)
    ax_r_twin.spines["top"].set_visible(False)
    axes[1].legend([l3, l4], ["Macro F1 (Left)", "False Alerts/10k (Right)"], loc="best", fontsize=8.5)

    # plotting psi drift threshold sensitivity
    df_p = results_dict["psi"]
    x_p = np.arange(len(df_p))
    ax_p_twin = axes[2].twinx()
    l5 = axes[2].errorbar(
        x_p, df_p["final_cum_macro_f1_mean"], yerr=df_p["final_cum_macro_f1_std"],
        color="#1b9e77", marker="o", linewidth=2.2, capsize=3, label="Macro F1 (Left)"
    )
    l6 = ax_p_twin.errorbar(
        x_p, df_p["drift_triggers_mean"], yerr=df_p["drift_triggers_std"],
        color="#7570b3", marker="^", linestyle="--", linewidth=2.0, capsize=3, label="Drift Triggers (Right)"
    )
    axes[2].set_xticks(x_p)
    axes[2].set_xticklabels([f"{p:.2f}" for p in df_p["psi_threshold"]])
    axes[2].set_title("(c) PSI Drift Threshold Sensitivity", pad=12)
    axes[2].set_xlabel("PSI Detection Threshold")
    axes[2].set_ylabel("Final Cumulative Macro F1", color="#1b9e77")
    ax_p_twin.set_ylabel("Mean Drift Triggers", color="#7570b3")
    axes[2].grid(True, linestyle="--", alpha=0.4)
    axes[2].spines["top"].set_visible(False)
    ax_p_twin.spines["top"].set_visible(False)
    axes[2].legend([l5, l6], ["Macro F1 (Left)", "Drift Triggers (Right)"], loc="best", fontsize=8.5)

    plt.tight_layout()
    for folder in (repo_fig_dir, drive_fig_dir):
        os.makedirs(folder, exist_ok=True)
        for ext in ("png", "pdf"):
            plt.savefig(os.path.join(folder, f"fig5_ablation_study.{ext}"), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def main():
    # running hyperparameter sensitivity ablations on scenario s3 across three benchmark seeds
    eval_seeds = [42, 101, 202]
    preseed_existing_dadr_checkpoints(eval_seeds)

    runner = ExperimentRunner(dataset_name="cicids2017")
    ablation_results = runner.run_ablations(scenario="S3", seeds=eval_seeds)

    # copying generated ablation csv tables into the repository results folder
    repo_results_dir = os.path.join(Config.REPO_DIR, "results")
    os.makedirs(repo_results_dir, exist_ok=True)
    for csv_name in ("ablation_budget.csv", "ablation_ratio.csv", "ablation_psi.csv"):
        src = os.path.join(Config.RESULTS_DIR, csv_name)
        dst = os.path.join(repo_results_dir, csv_name)
        shutil.copy2(src, dst)

    # generating fig5 ablation summary plot
    plot_ablation_figure(
        ablation_results,
        repo_fig_dir=os.path.join(repo_results_dir, "figures"),
        drive_fig_dir=os.path.join(Config.RESULTS_DIR, "figures"),
    )
    print("ablation study and fig5 completed successfully")


if __name__ == "__main__":
    main()
