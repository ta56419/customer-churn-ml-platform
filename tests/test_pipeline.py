import unittest

import pandas as pd

from churn_ml.features import validate_frame
from churn_ml.pipeline import build_pipeline


class PipelineTests(unittest.TestCase):
    def test_schema_rejects_missing_columns(self):
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            validate_frame(pd.DataFrame({"tenure_months": [3]}))

    def test_schema_rejects_unknown_category(self):
        row = {"tenure_months": 3, "monthly_charges": 50, "support_tickets_90d": 0,
               "contract": "unknown", "payment_method": "card"}
        with self.assertRaisesRegex(ValueError, "Unsupported contract"):
            validate_frame(pd.DataFrame([row]), require_target=False)

    def test_pipeline_predicts_probabilities(self):
        frame = pd.DataFrame([
            {"tenure_months": t, "monthly_charges": 70, "support_tickets_90d": n,
             "contract": c, "payment_method": "card"}
            for t, n, c in [(2, 3, "month_to_month"), (4, 2, "month_to_month"),
                            (48, 0, "two_year"), (60, 0, "one_year"),
                            (8, 2, "month_to_month"), (55, 1, "two_year")]
        ])
        labels = [1, 1, 0, 0, 1, 0]
        model = build_pipeline().fit(frame, labels)
        probabilities = model.predict_proba(frame)[:, 1]
        self.assertEqual(len(probabilities), len(labels))
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())


if __name__ == "__main__":
    unittest.main()
