"""Regenerate the compact student report from frozen scores and numeric logs."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import report_helpers as helpers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    campaign = root / "campaign"
    snapshot = campaign / "campaign_snapshot.json"
    design = json.loads(snapshot.read_text())
    digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    rows = []
    for policy in design["policies"]:
        for condition in design["conditions"]:
            for seed in design["sampling_seeds"]:
                case = f"{policy['id']}__{condition['id']}__seed{seed}"
                score = json.loads(
                    (campaign / "analysis/trials" / f"{case}.json").read_text()
                )
                row, _ = helpers.trial_row(
                    campaign, policy["id"], condition, seed, score, {}, digest
                )
                row["video_relative_path"] = (
                    f"../video_reviews/{case}/policy_review.mp4"
                )
                row["shared_controller_tactile_not_student_model_input"] = True
                rows.append(row)
    if len(rows) != 4 or not all(row["valid_for_scoring"] for row in rows):
        raise ValueError(
            "Expected the complete four-case smoke, without invalid scores"
        )
    videos = json.loads((root / "video_reviews/manifest.json").read_text())
    summary = {
        "campaign_id": design["campaign_id"],
        "campaign_sha256": digest,
        "valid": 4,
        "invalid": 0,
        "missing": 0,
        "policy_counts": {
            policy["id"]: helpers.aggregate(
                [row for row in rows if row["policy_id"] == policy["id"]]
            )
            for policy in design["policies"]
        },
        "inference_settings": design["inference_settings"],
        "terminal_causes": {row["case_id"]: row["terminal_cause"] for row in rows},
        "data_scope": "Two nominal sampling seeds per student; compatibility/failure evidence only. No paired teacher comparisons, confidence intervals, rankings or stress-suite extrapolation.",
        "source_report_helper_sha256": hashlib.sha256(
            Path(helpers.__file__).read_bytes()
        ).hexdigest(),
        "video_semantics": "Controller/simulator tactile proxy panels; students do not consume observed fingertip pixels. Wrist retained.",
        "trials": rows,
        "videos_verified": videos["verified_count"],
        "video_total_bytes": videos["total_video_bytes"],
    }
    out.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (out / "per_trial.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            {key: "NA" if value is None else value for key, value in row.items()}
            for row in rows
        )
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({"valid": 4, "out": str(out)}))


if __name__ == "__main__":
    main()
