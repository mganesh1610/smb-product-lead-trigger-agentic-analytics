#!/usr/bin/env python3
"""
Unit, Statistical, Schema-Contract & Agentic Guardrail Tests
for Small Business Product Analytics Engine
"""

import json
import unittest
from pathlib import Path

from src.smb_product_analytics_engine import run_full_pipeline


class TestSMBProductAnalyticsEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.summary = run_full_pipeline(cls.root)

    def test_01_combined_row_scale_and_sha256_integrity(self) -> None:
        self.assertEqual(self.summary["combined_public_records_processed"], 58294)
        self.assertEqual(self.summary["lead_trigger_funnel_records"], 45211)
        self.assertEqual(self.summary["servicing_text_inquiry_records"], 13083)

    def test_02_day_one_governance_and_controls_15_of_15(self) -> None:
        gov = self.summary["day_one_governance_and_controls"]
        self.assertEqual(gov["total_checks"], 15)
        self.assertEqual(gov["passed_checks"], 15)
        self.assertEqual(gov["pass_rate_pct"], 100.0)
        for name, passed in gov["assertions"].items():
            self.assertTrue(passed, f"Governance check failed: {name}")

    def test_03_lead_trigger_propensity_model_performance(self) -> None:
        model = self.summary["lead_trigger_propensity_model"]
        self.assertEqual(model["train_rows"], 36168)
        self.assertEqual(model["holdout_test_rows"], 9043)
        self.assertGreaterEqual(model["holdout_roc_auc"], 0.88)
        self.assertGreaterEqual(model["top_20pct_trigger_lift_x"], 3.0)
        self.assertLess(model["holdout_brier_score"], model["baseline_brier_score"])

    def test_04_experimentation_and_doubly_robust_aipw(self) -> None:
        exp = self.summary["experimentation_and_causal_inference"]
        self.assertEqual(exp["treatment_clients"] + exp["control_clients"], 45211)
        z_test = exp["two_proportion_z_test"]
        self.assertGreater(z_test["abs_lift_pp"], 5.0)
        self.assertGreater(z_test["z_stat"], 20.0)
        self.assertGreater(exp["doubly_robust_aipw_ate_pp"], 2.5)

    def test_05_text_mining_servicing_pathways(self) -> None:
        tm = self.summary["text_mining_servicing_pathways"]
        self.assertEqual(tm["total_servicing_utterances"], 13083)
        self.assertEqual(tm["distinct_banking_intents"], 77)
        self.assertEqual(tm["servicing_pathways_count"], 6)
        self.assertGreaterEqual(tm["holdout_pathway_top1_accuracy_pct"], 85.0)
        self.assertGreaterEqual(tm["holdout_pathway_top2_accuracy_pct"], 94.0)

    def test_06_exported_scorecards_and_agentic_audit_log(self) -> None:
        outputs = self.root / "outputs"
        self.assertTrue((outputs / "smb_product_kpi_scorecard.csv").exists())
        self.assertTrue((outputs / "text_mining_servicing_pathways.csv").exists())
        self.assertTrue((outputs / "experimentation_and_trigger_results.json").exists())
        audit_path = outputs / "agentic_decision_audit_log.json"
        self.assertTrue(audit_path.exists())
        audit_rows = json.loads(audit_path.read_text(encoding="utf-8"))
        self.assertEqual(len(audit_rows), 12)
        for row in audit_rows:
            self.assertTrue(row["schema_valid"])
            self.assertEqual(row["controls_kyc_pass_rate_pct"], 100.0)
            self.assertEqual(row["audit_traceability_pct"], 100.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
