"""Batch export pipeline and dataset evidence generator.

Implements Tasks 1, 5, and 6:
- Processes all 90 trips (80 train, 10 student_test)
- Exports separate observations.parquet, candidates.parquet, and labels.parquet for each partition
- Generates evidence/data/inventory.csv
- Generates data-quality diagnostic plots
- Saves configuration metadata
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure local imports work
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from data_pipeline import (
    load_inputs,
    build_candidates,
    load_labels,
    load_raw_route,
    FEATURE_NAMES,
    __version__
)


def run_full_pipeline(raw_root: Path, output_root: Path, config: Dict[str, Any] = None):
    t_start = time.time()
    if config is None:
        config = {
            "search_radius_m": 50.0,
            "max_candidates_per_obs": 15,
            "schema_version": "1.0.0"
        }
        
    prep_dir = output_root / "prepared_data"
    evidence_dir = output_root / "evidence" / "data"
    prep_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(repo_root / 'feature_schema.json', prep_dir / 'feature_schema.json')
    
    inventory_records = []
    
    # Store aggregated metrics for plotting
    all_dt = []
    all_speeds = []
    all_cand_counts = []
    all_cand_dists = []
    all_alignment_reasons = []
    node_counts = []
    arc_counts = []
    route_lens = []
    
    for partition in ["train", "student_test"]:
        part_dir = prep_dir / partition
        part_dir.mkdir(parents=True, exist_ok=True)
        
        # Resolve raw partition directory
        raw_part_dir = raw_root / partition
        if not raw_part_dir.exists():
            raw_part_dir = raw_root / "Student folder" / partition
            
        trips = sorted([d.name for d in raw_part_dir.iterdir() if d.is_dir()])
        print(f"\nProcessing {len(trips)} trips for partition '{partition}'...")
        
        part_obs_list = []
        part_cands_list = []
        part_labels_list = []
        
        for idx, tid in enumerate(trips):
            t_trip_0 = time.time()
            inputs = load_inputs(raw_root, partition, tid)
            cand_batch = build_candidates(inputs, config)
            label_table = load_labels(raw_root, partition, tid)
            raw_route = load_raw_route(raw_root, partition, tid)
            
            part_obs_list.append(cand_batch.observations.assign(trip_id=tid))
            part_cands_list.append(cand_batch.candidates)
            part_labels_list.append(label_table.labels)
            
            # Trajectory stats
            obs_df = inputs.observations
            time_arr = obs_df["timestamp_s"].values
            dt = np.diff(time_arr) if len(time_arr) > 1 else np.array([1.0])
            all_dt.extend(dt) # Preserve every interval for a correct sampling denominator.
            
            # Kinematic stats
            mean_lat = float(obs_df["lat"].mean())
            mean_lon = float(obs_df["lon"].mean())
            kx = 111320.0 * np.cos(np.radians(mean_lat))
            ky = 111320.0
            dx = np.diff(obs_df["lon"].values) * kx
            dy = np.diff(obs_df["lat"].values) * ky
            step_dists_m = np.sqrt(dx**2 + dy**2)
            total_km = float(np.sum(step_dists_m) / 1000.0)
            speeds_mps = step_dists_m / np.where(dt <= 0, 1.0, dt)
            speeds_kmh = speeds_mps * 3.6
            all_speeds.extend(speeds_kmh[speeds_kmh <= 180.0])
            
            # Candidate stats
            counts_per_obs = cand_batch.candidates.groupby("obs_id")["candidate_id"].count()
            # Account for zero-candidate observations
            full_counts = np.zeros(len(obs_df), dtype=int)
            full_counts[counts_per_obs.index] = counts_per_obs.values
            all_cand_counts.extend(full_counts)
            if len(cand_batch.candidates) > 0:
                all_cand_dists.extend(cand_batch.candidates["distance_to_segment_m"].values[:500])
                
            # Alignment stats
            all_alignment_reasons.extend(label_table.labels["alignment_reason"].tolist())
            
            node_counts.append(len(inputs.nodes))
            arc_counts.append(len(inputs.arcs))
            route_len = len(raw_route) if raw_route is not None else 0
            route_lens.append(route_len)
            
            # Quality flags
            issues = []
            if dt.max() > 60.0:
                issues.append(f"large_gap_{int(dt.max())}s")
            if (dt <= 0).any():
                issues.append("non_monotonic_time")
            if speeds_kmh.max() > 180.0:
                issues.append(f"high_speed_{int(speeds_kmh.max())}kmh")
            if raw_route is not None and len(raw_route) != raw_route["arc_id"].nunique():
                issues.append("repeated_route_arcs")
            if label_table.coverage < 0.85:
                issues.append(f"low_coverage_{int(label_table.coverage*100)}pct")
                
            inv_rec = {
                "partition": partition,
                "trip_id": tid,
                "obs_count": len(obs_df),
                "duration_s": round(float(time_arr[-1] - time_arr[0]), 1),
                "dt_median_s": round(float(np.median(dt)), 2),
                "dt_max_s": round(float(np.max(dt)), 1),
                "dt_gaps_count": int(np.sum(dt > 3.0)),
                "node_count": len(inputs.nodes),
                "arc_count": len(inputs.arcs),
                "route_arc_count": route_len,
                "route_unique_arcs": raw_route["arc_id"].nunique() if raw_route is not None else 0,
                "track_length_km": round(total_km, 2),
                "mean_speed_kmh": round(float(np.mean(speeds_kmh)), 1),
                "max_speed_kmh": round(float(np.max(speeds_kmh)), 1),
                "candidate_count": len(cand_batch.candidates),
                "mean_candidates_per_obs": round(float(len(cand_batch.candidates) / max(len(obs_df), 1)), 2),
                "alignment_valid_coverage": round(label_table.coverage, 4),
                "lon_min": round(float(obs_df["lon"].min()), 6),
                "lon_max": round(float(obs_df["lon"].max()), 6),
                "lat_min": round(float(obs_df["lat"].min()), 6),
                "lat_max": round(float(obs_df["lat"].max()), 6),
                "data_quality_issues": ";".join(issues) if issues else "clean"
            }
            inventory_records.append(inv_rec)
            
            if (idx + 1) % 10 == 0 or (idx + 1) == len(trips):
                print(f"  Processed {idx + 1}/{len(trips)} trips ({time.time() - t_trip_0:.2f}s/trip)")
                
        # Export concatenated Parquet files for partition
        print(f"Exporting Parquet files for partition '{partition}'...")
        all_obs_df = pd.concat(part_obs_list, ignore_index=True)
        all_cands_df = pd.concat(part_cands_list, ignore_index=True)
        all_labels_df = pd.concat(part_labels_list, ignore_index=True)
        
        all_obs_df.to_parquet(part_dir / "observations.parquet", index=False)
        all_cands_df.to_parquet(part_dir / "candidates.parquet", index=False)
        all_labels_df.to_parquet(part_dir / "labels.parquet", index=False)
        
        print(f"  Saved {part_dir / 'observations.parquet'} ({len(all_obs_df):,} rows)")
        print(f"  Saved {part_dir / 'candidates.parquet'} ({len(all_cands_df):,} rows)")
        print(f"  Saved {part_dir / 'labels.parquet'} ({len(all_labels_df):,} rows)")

    # Save inventory.csv
    inv_df = pd.DataFrame(inventory_records)
    inv_csv_path = evidence_dir / "inventory.csv"
    inv_df.to_csv(inv_csv_path, index=False)
    print(f"\nSaved inventory to {inv_csv_path} ({len(inv_df)} trips)")
    
    # Save config.json
    config_metadata = {
        "pipeline_version": __version__,
        "schema_version": config["schema_version"],
        "search_radius_m": config["search_radius_m"],
        "max_candidates_per_obs": config["max_candidates_per_obs"],
        "total_trips": len(inv_df),
        "train_trips": int((inv_df["partition"] == "train").sum()),
        "student_test_trips": int((inv_df["partition"] == "student_test").sum()),
        "total_observations": int(inv_df["obs_count"].sum()),
        "total_candidates": int(inv_df["candidate_count"].sum()),
        "generated_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "feature_order": FEATURE_NAMES
    }
    with open(prep_dir / "pipeline_config.json", "w") as f:
        json.dump(config_metadata, f, indent=2)

    # Generate Data Quality Plots
    print("\nGenerating data quality evidence plots...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # 1. Sampling Interval Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    dt_subset = [x for x in all_dt if 0 <= x <= 5.0]
    ax.hist(dt_subset, bins=50, color="#1f77b4", edgecolor="black", alpha=0.8)
    ax.set_title(r"GPS Sampling Interval Distribution ($\Delta t \leq 5$s)", fontsize=13, fontweight="bold")
    ax.set_xlabel(r"Time Difference $\Delta t$ (seconds)", fontsize=11)
    ax.set_ylabel("Observation Count", fontsize=11)
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1.5, label=f"1.0 s intervals ({100*np.isclose(all_dt, 1.0).mean():.2f}%)")
    ax.legend(fontsize=11)
    plt.tight_layout()
    fig.savefig(evidence_dir / "sampling_interval_dist.png", dpi=200)
    plt.close(fig)
    
    # 2. Vehicle Speed Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_speeds, bins=60, color="#2ca02c", edgecolor="black", alpha=0.8)
    ax.set_title("Vehicle Speed Distribution Across Observations", fontsize=13, fontweight="bold")
    ax.set_xlabel("GPS Calculated Speed (km/h)", fontsize=11)
    ax.set_ylabel("Observation Count", fontsize=11)
    ax.axvline(50.0, color="orange", linestyle="--", label="Urban Limit (50 km/h)")
    ax.axvline(120.0, color="red", linestyle="--", label="Highway Limit (120 km/h)")
    ax.legend(fontsize=11)
    plt.tight_layout()
    fig.savefig(evidence_dir / "speed_distribution.png", dpi=200)
    plt.close(fig)
    
    # 3. Candidate Count per Observation
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_cand_counts, bins=np.arange(-0.5, 16.5, 1), color="#9467bd", edgecolor="black", alpha=0.8)
    ax.set_title(r"Road Segment Candidates per GPS Observation ($R \leq 50$m, max 15)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Candidate Count", fontsize=11)
    ax.set_ylabel("Observation Count", fontsize=11)
    mean_c = np.mean(all_cand_counts)
    ax.axvline(mean_c, color="darkred", linestyle="--", linewidth=1.8, label=f"Mean = {mean_c:.2f}")
    ax.legend(fontsize=11)
    plt.tight_layout()
    fig.savefig(evidence_dir / "candidate_count_dist.png", dpi=200)
    plt.close(fig)
    
    # 4. Candidate Distance to Segment Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_cand_dists, bins=50, color="#e377c2", edgecolor="black", alpha=0.8)
    ax.set_title("Metric Distance from GPS Observation to Candidate Segment", fontsize=13, fontweight="bold")
    ax.set_xlabel("Distance to Segment (metres)", fontsize=11)
    ax.set_ylabel("Candidate Count", fontsize=11)
    ax.axvline(20.0, color="blue", linestyle="--", label="20m Threshold")
    ax.legend(fontsize=11)
    plt.tight_layout()
    fig.savefig(evidence_dir / "candidate_distance_dist.png", dpi=200)
    plt.close(fig)
    
    # 5. Route Alignment Coverage Breakdown
    fig, ax = plt.subplots(figsize=(8, 5))
    reason_series = pd.Series(all_alignment_reasons).value_counts()
    colors = ["#2ca02c" if r == "valid_aligned" else "#d62728" for r in reason_series.index]
    bars = ax.bar(reason_series.index, reason_series.values, color=colors, edgecolor="black", alpha=0.85)
    ax.set_title("Reference Route Alignment Diagnostic Breakdown", fontsize=13, fontweight="bold")
    ax.set_ylabel("Observation Count", fontsize=11)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 1000, f"{h:,} ({h/len(all_alignment_reasons)*100:.1f}%)",
                ha="center", va="bottom", fontsize=10)
    plt.xticks(rotation=15, ha="right", fontsize=10)
    plt.tight_layout()
    fig.savefig(evidence_dir / "alignment_coverage.png", dpi=200)
    plt.close(fig)
    
    # 6. Road Network Scale Summary
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.boxplot([node_counts, arc_counts], tick_labels=["Nodes", "Arcs"], patch_artist=True,
                 boxprops=dict(facecolor="#8c564b", alpha=0.7))
    ax1.set_title("Road Network Node & Arc Counts per Trip", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Count (log scale)", fontsize=10)
    ax1.set_yscale("log")
    
    ax2.hist(route_lens, bins=25, color="#17becf", edgecolor="black", alpha=0.8)
    ax2.set_title("Ground Truth Route Arc Count per Trip", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Traversed Route Arcs", fontsize=10)
    ax2.set_ylabel("Trip Count", fontsize=10)
    plt.tight_layout()
    fig.savefig(evidence_dir / "network_scale_summary.png", dpi=200)
    plt.close(fig)
    
    print(f"All plots saved to {evidence_dir}")
    print(f"Full pipeline completed in {time.time() - t_start:.2f} seconds!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, default=repo_root / 'Student folder')
    parser.add_argument('--output-root', type=Path, default=repo_root)
    args = parser.parse_args()
    run_full_pipeline(args.raw_root, args.output_root)
