import json
import os
import numpy as np
import pandas as pd
import torch

from src.config import Config
from src.data_loader import ContinualIDSDataLoader
from src.trainer import ContinualIDSTrainer
from src.baselines import DADRVariantStrategy


ALL_METHODS = [
    "Naive",
    "Joint",
    "EWC",
    "LwF",
    "ER-Reservoir",
    "ER-ClassBalanced",
    "Attack-Only",
    "DADR-NoDrift",
    "DADR",
]


class ExperimentRunner:
    def __init__(self, dataset_name: str = "cicids2017", device: torch.device = None):
        # initializing experiment runner and ensuring drive output directories exist
        Config.ensure_dirs()
        self.dataset_name = dataset_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = os.path.join(Config.RESULTS_DIR, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _ckpt_path(self, tag: str) -> str:
        # returning standardized json checkpoint filepath for incremental saving
        safe_tag = tag.replace("/", "_").replace(" ", "_")
        return os.path.join(self.checkpoint_dir, f"{safe_tag}.json")

    def _load_or_run_single(
        self,
        run_tag: str,
        data_dict: dict,
        method_name: str,
        seed: int,
        capacity: int = None,
        attack_ratio: float = None,
        psi_threshold: float = None,
    ) -> dict:
        # loading cached result if already completed, otherwise executing and checkpointing
        ckpt_file = self._ckpt_path(run_tag)
        if os.path.exists(ckpt_file):
            with open(ckpt_file, "r") as f:
                return json.load(f)

        cap = capacity if capacity is not None else Config.TOTAL_MEMORY_BUDGET

        # supporting custom ratio and psi threshold ablations cleanly
        if attack_ratio is not None or psi_threshold is not None:
            result = self._run_custom_dadr(
                data_dict=data_dict,
                seed=seed,
                capacity=cap,
                attack_ratio=attack_ratio if attack_ratio is not None else Config.ATTACK_POOL_RATIO,
                psi_threshold=psi_threshold if psi_threshold is not None else Config.PSI_THRESHOLD,
            )
        else:
            trainer = ContinualIDSTrainer(
                method_name=method_name,
                seed=seed,
                capacity=cap,
                device=self.device,
            )
            result = trainer.run(data_dict)

        with open(ckpt_file, "w") as f:
            json.dump(result, f, indent=2)
        return result

    def _run_custom_dadr(
        self,
        data_dict: dict,
        seed: int,
        capacity: int,
        attack_ratio: float,
        psi_threshold: float,
    ) -> dict:
        # running dadr with overridden pool ratio or psi threshold for ablation studies
        import src.baselines as baselines_mod

        orig_build = baselines_mod.build_strategy

        def custom_builder(method_name, device, seed=None, capacity=None):
            strat = DADRVariantStrategy(
                name=f"DADR_r{attack_ratio:.2f}_psi{psi_threshold:.2f}",
                device=device,
                seed=seed,
                capacity=capacity,
                attack_ratio=attack_ratio,
                enable_drift_refresh=True,
            )
            strat.buffer.drift_detector.threshold = psi_threshold
            return strat

        baselines_mod.build_strategy = custom_builder
        try:
            trainer = ContinualIDSTrainer(
                method_name="DADR",
                seed=seed,
                capacity=capacity,
                device=self.device,
            )
            return trainer.run(data_dict)
        finally:
            baselines_mod.build_strategy = orig_build

    def _aggregate_summaries(self, summaries: list, group_cols: list) -> pd.DataFrame:
        # computing mean and standard deviation across seeds for paper tables
        df = pd.DataFrame(summaries)
        metric_cols = [
            "mean_preq_macro_f1",
            "mean_cum_macro_f1",
            "final_cum_macro_f1",
            "mean_cum_attack_recall",
            "mean_cum_fpr",
            "mean_cum_false_alerts_10k",
            "avg_forgetting",
            "drift_triggers",
            "runtime_sec",
        ]
        grouped = df.groupby(group_cols, sort=False)[metric_cols].agg(["mean", "std"]).reset_index()

        flat_cols = list(group_cols)
        for m in metric_cols:
            flat_cols.extend([f"{m}_mean", f"{m}_std"])
        grouped.columns = flat_cols
        return grouped

    def run_main_benchmark(
        self,
        scenarios: list = None,
        methods: list = None,
        seeds: list = None,
        max_rows_per_file: int = None,
    ) -> pd.DataFrame:
        # executing main benchmark across scenarios and random seeds
        scenarios = scenarios or ["S1", "S3"]
        methods = methods or ALL_METHODS
        seeds = seeds or Config.SEEDS

        all_summaries = []
        all_window_logs = []

        for scenario in scenarios:
            print(f"\n=== Loading {self.dataset_name.upper()} | Scenario {scenario} ===")
            loader = ContinualIDSDataLoader(dataset_name=self.dataset_name, scenario=scenario)
            data_dict = loader.load_and_split(max_rows_per_file=max_rows_per_file)

            for method in methods:
                for seed in seeds:
                    tag = f"main_{self.dataset_name}_{scenario}_{method}_seed{seed}"
                    res = self._load_or_run_single(
                        run_tag=tag,
                        data_dict=data_dict,
                        method_name=method,
                        seed=seed,
                    )
                    summary = res["summary"].copy()
                    summary["scenario"] = scenario
                    all_summaries.append(summary)

                    for w_log in res["window_logs"]:
                        row = w_log.copy()
                        row["scenario"] = scenario
                        row["method"] = method
                        row["seed"] = seed
                        all_window_logs.append(row)

                m_scores = [
                    s["final_cum_macro_f1"]
                    for s in all_summaries
                    if s["scenario"] == scenario and s["method"] == method
                ]
                print(
                    f"[{scenario}] {method:18s} | Final F1 ({len(m_scores)} seeds): "
                    f"{np.mean(m_scores):.4f} ± {np.std(m_scores):.4f}"
                )

        raw_df = pd.DataFrame(all_summaries)
        win_df = pd.DataFrame(all_window_logs)
        agg_df = self._aggregate_summaries(all_summaries, group_cols=["scenario", "method"])

        raw_df.to_csv(os.path.join(Config.RESULTS_DIR, "main_benchmark_raw.csv"), index=False)
        win_df.to_csv(os.path.join(Config.RESULTS_DIR, "main_benchmark_windows.csv"), index=False)
        agg_df.to_csv(os.path.join(Config.RESULTS_DIR, "main_benchmark_summary.csv"), index=False)
        print(f"\nSaved main benchmark tables to {Config.RESULTS_DIR}")
        return agg_df

    def run_ablations(
        self,
        scenario: str = "S3",
        seeds: list = None,
        max_rows_per_file: int = None,
    ) -> dict:
        # running budget, attack/normal ratio, and psi threshold sensitivity ablations
        seeds = seeds or Config.SEEDS[:5]
        loader = ContinualIDSDataLoader(dataset_name=self.dataset_name, scenario=scenario)
        data_dict = loader.load_and_split(max_rows_per_file=max_rows_per_file)

        # running memory budget ablation
        budget_rows = []
        for budget in Config.BUDGET_ABLATIONS:
            for seed in seeds:
                tag = f"ablation_budget_{self.dataset_name}_{scenario}_B{budget}_seed{seed}"
                res = self._load_or_run_single(
                    run_tag=tag,
                    data_dict=data_dict,
                    method_name="DADR",
                    seed=seed,
                    capacity=budget,
                )
                row = res["summary"].copy()
                row["budget"] = budget
                budget_rows.append(row)
        budget_df = self._aggregate_summaries(budget_rows, group_cols=["budget"])
        budget_df.to_csv(os.path.join(Config.RESULTS_DIR, "ablation_budget.csv"), index=False)

        # running attack vs normal ratio ablation
        ratio_rows = []
        for att_r, norm_r in Config.RATIO_ABLATIONS:
            for seed in seeds:
                tag = f"ablation_ratio_{self.dataset_name}_{scenario}_A{int(att_r*100)}_N{int(norm_r*100)}_seed{seed}"
                res = self._load_or_run_single(
                    run_tag=tag,
                    data_dict=data_dict,
                    method_name="DADR",
                    seed=seed,
                    attack_ratio=att_r,
                )
                row = res["summary"].copy()
                row["attack_ratio"] = att_r
                row["normal_ratio"] = norm_r
                ratio_rows.append(row)
        ratio_df = self._aggregate_summaries(ratio_rows, group_cols=["attack_ratio", "normal_ratio"])
        ratio_df.to_csv(os.path.join(Config.RESULTS_DIR, "ablation_ratio.csv"), index=False)

        # running psi threshold sensitivity ablation
        psi_rows = []
        for psi_th in Config.PSI_ABLATIONS:
            for seed in seeds:
                tag = f"ablation_psi_{self.dataset_name}_{scenario}_P{int(psi_th*100)}_seed{seed}"
                res = self._load_or_run_single(
                    run_tag=tag,
                    data_dict=data_dict,
                    method_name="DADR",
                    seed=seed,
                    psi_threshold=psi_th,
                )
                row = res["summary"].copy()
                row["psi_threshold"] = psi_th
                psi_rows.append(row)
        psi_df = self._aggregate_summaries(psi_rows, group_cols=["psi_threshold"])
        psi_df.to_csv(os.path.join(Config.RESULTS_DIR, "ablation_psi.csv"), index=False)

        print(f"Saved all ablation tables to {Config.RESULTS_DIR}")
        return {"budget": budget_df, "ratio": ratio_df, "psi": psi_df}
