from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from careerpilot.service import run_careerpilot


def main() -> None:
    case_path = PROJECT_ROOT / "examples" / "evaluation_cases.json"
    cases = json.loads(case_path.read_text(encoding="utf-8"))

    rows = []
    for case in cases:
        result = run_careerpilot(
            resume_text=case["resume_text"],
            jd_text=case["jd_text"],
            offline=True,
        )
        report = result["match_report"]
        rows.append(
            (
                case["name"],
                report["score"],
                report["level"],
                f"{report['matched_count']}/{report['required_count']}",
                ", ".join(report["missing_skills"]) or "无",
            )
        )

    print(f"{'案例':<10} {'分数':>6} {'等级':<8} {'命中':<8} 缺失技能")
    print("-" * 80)
    for name, score, level, coverage, missing in rows:
        print(f"{name:<10} {score:>6.1f} {level:<8} {coverage:<8} {missing}")


if __name__ == "__main__":
    main()
