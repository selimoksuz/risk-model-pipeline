#!/usr/bin/env python
"""Simple CLI for production monitoring.

This utility exposes the :func:`risk_pipeline.monitoring.monitor_scores`
function via command line so that score and feature PSI calculations can be
run on demand.
"""

import argparse
import json
from risk_pipeline.monitoring import monitor_scores


def main():
    p = argparse.ArgumentParser(description="Monitor model drift via PSI")
    p.add_argument("baseline", help="Path to baseline CSV")
    p.add_argument("new", help="Path to new CSV")
    p.add_argument("mapping", help="WOE mapping JSON path")
    p.add_argument("final_vars", help="Comma separated list of model variables")
    p.add_argument("model", help="Trained model path")
    p.add_argument("--calibrator", help="Optional calibrator path")
    p.add_argument("--expected-model-type", dest="model_type", help="Expected model class name")
    args = p.parse_args()

    vars_list = [v.strip() for v in args.final_vars.split(",") if v.strip()]
    res = monitor_scores(
        args.baseline,
        args.new,
        args.mapping,
        vars_list,
        args.model,
        calibrator_path=args.calibrator,
        expected_model_type=args.model_type,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
