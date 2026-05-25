import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from services.portfolio_service import build_portfolio_change_summary, record_portfolio_snapshot


class PortfolioSnapshotTests(unittest.TestCase):
    @patch("services.portfolio_service.save_user_preference")
    @patch("services.portfolio_service.get_user_preference")
    def test_record_portfolio_snapshot_does_not_dedupe_backdated_runs(
        self,
        mock_get_user_preference,
        mock_save_user_preference,
    ):
        mock_get_user_preference.return_value = [
            {
                "captured_at": "2026-05-25T12:00:00+00:00",
                "source": "alpaca_sync",
                "holdings_count": 1,
                "total_cost_basis": 400.0,
                "total_market_value": 900.0,
                "market_values_available": True,
                "total_unrealized_pl": 500.0,
                "holdings": [
                    {
                        "symbol": "MSFT",
                        "quantity": 2.0,
                        "cost_basis": 200.0,
                        "market_value": 900.0,
                    }
                ],
                "signature": "same-signature",
                "metadata": {},
            }
        ]

        snapshot = record_portfolio_snapshot(
            [
                {
                    "symbol": "MSFT",
                    "quantity": 2,
                    "cost_basis": 200,
                    "market_value": 900,
                }
            ],
            source="portfolio_digest",
            captured_at="2026-05-24T09:00:00+00:00",
        )

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot["captured_at"], "2026-05-24T09:00:00+00:00")
        mock_save_user_preference.assert_called_once()

    @patch("services.portfolio_service.save_user_preference")
    @patch("services.portfolio_service.get_user_preference")
    def test_build_portfolio_change_summary_initializes_first_baseline(
        self,
        mock_get_user_preference,
        mock_save_user_preference,
    ):
        mock_get_user_preference.return_value = []

        summary = build_portfolio_change_summary(
            [
                {
                    "symbol": "MSFT",
                    "quantity": 2,
                    "cost_basis": 200,
                    "market_value": 450,
                }
            ]
        )

        self.assertIsNotNone(summary["current_snapshot"])
        self.assertIsNone(summary["baseline_snapshot"])
        self.assertIsNone(summary["market_value_change"])
        self.assertEqual(summary["new_positions"], [])
        mock_save_user_preference.assert_called_once()

    @patch("services.portfolio_service.save_user_preference")
    @patch("services.portfolio_service.get_user_preference")
    def test_build_portfolio_change_summary_compares_against_prior_weekly_snapshot(
        self,
        mock_get_user_preference,
        mock_save_user_preference,
    ):
        baseline_time = (datetime.now(timezone.utc) - timedelta(days=8)).replace(microsecond=0).isoformat()
        baseline_snapshot = {
            "captured_at": baseline_time,
            "source": "portfolio_digest",
            "holdings_count": 1,
            "total_cost_basis": 400.0,
            "total_market_value": 900.0,
            "market_values_available": True,
            "total_unrealized_pl": 500.0,
            "holdings": [
                {
                    "symbol": "MSFT",
                    "quantity": 2.0,
                    "cost_basis": 200.0,
                    "market_value": 900.0,
                }
            ],
            "signature": "baseline",
            "metadata": {},
        }
        mock_get_user_preference.return_value = [baseline_snapshot]

        summary = build_portfolio_change_summary(
            [
                {
                    "symbol": "MSFT",
                    "quantity": 2,
                    "cost_basis": 200,
                    "market_value": 1100,
                },
                {
                    "symbol": "NVDA",
                    "quantity": 1,
                    "cost_basis": 300,
                    "market_value": 320,
                },
            ]
        )

        self.assertIsNotNone(summary["baseline_snapshot"])
        self.assertEqual(summary["market_value_change"], 520.0)
        self.assertAlmostEqual(summary["market_value_change_pct"], 57.78, places=2)
        self.assertEqual(summary["new_positions"], ["NVDA"])
        self.assertEqual(summary["removed_positions"], [])
        self.assertTrue(any(item["symbol"] == "MSFT" for item in summary["largest_position_changes"]))
        mock_save_user_preference.assert_called_once()


if __name__ == "__main__":
    unittest.main()
