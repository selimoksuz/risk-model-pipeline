from pathlib import Path
import pandas as pd

from risk_pipeline.core.config import Config
from risk_pipeline.unified_pipeline import UnifiedRiskPipeline
from risk_pipeline.data.sample import load_credit_risk_sample


def main():
    out_dir = Path('examples/quick_validation')
    out_dir.mkdir(parents=True, exist_ok=True)

    sample = load_credit_risk_sample()

    cfg = Config(
        target_column='target',
        id_column='customer_id',
        time_column='app_dt',
        create_test_split=True,
        stratify_test=True,
        train_ratio=0.8,
        test_ratio=0.2,
        oot_ratio=0.0,
        oot_months=3,
        output_folder=str(out_dir),
        output_excel_path=str(out_dir / 'risk_pipeline_report.xlsx'),
        enable_tsfresh_features=False,
        selection_steps=['univariate','psi','vif','correlation','iv','boruta','stepwise'],
        min_univariate_gini=0.05,
        psi_threshold=0.25,
        monthly_psi_threshold=0.15,
        oot_psi_threshold=0.25,
        vif_threshold=5.0,
        correlation_threshold=0.9,
        iv_threshold=0.02,
        stepwise_method='forward',
        stepwise_max_features=25,
        algorithms=['logistic'],
        model_selection_method='gini_oot',
        model_stability_weight=0.2,
        min_gini_threshold=0.5,
        max_train_oot_gap=0.03,
        use_optuna=False,
        hpo_trials=1,
        hpo_timeout_sec=600,
        use_noise_sentinel=True,
        enable_dual=True,
        enable_woe_boost_scorecard=True,
        calculate_shap=False,
        enable_scoring=True,
        score_model_name='best',
        enable_stage2_calibration=True,
        n_risk_bands=10,
        risk_band_method='pd_constraints',
        risk_band_min_bins=7,
        risk_band_max_bins=10,
        risk_band_hhi_threshold=0.15,
        risk_band_binomial_pass_weight=0.85,
        random_state=42,
        n_jobs=2,
    )

    pipe = UnifiedRiskPipeline(cfg)
    dev = sample.development
    cal_long = sample.calibration_longrun
    cal_recent = sample.calibration_recent
    score_df = sample.scoring_future

    results = pipe.fit(
        dev,
        data_dictionary=getattr(sample, 'data_dictionary', None),
        calibration_df=cal_long,
        stage2_df=cal_recent,
        risk_band_df=dev,
        score_df=score_df,
    )

    reports = pipe.run_reporting(force=True)
    excel_path = reports.get('excel_path')
    print('Excel:', excel_path)
    print('models_summary present:', isinstance(reports.get('models_summary'), pd.DataFrame))
    print('band_metrics present:', isinstance(reports.get('band_metrics'), pd.DataFrame))
    dlo = reports.get('data_layers_overview')
    rps = reports.get('raw_preprocessing_summary')
    print('data_layers_overview shape:', None if dlo is None else dlo.shape)
    print('raw_preprocessing_summary shape:', None if rps is None else rps.shape)


if __name__ == '__main__':
    main()
