import sys
import types
import unittest
from unittest.mock import patch

if "requests" not in sys.modules:
    requests_stub = types.ModuleType("requests")

    class RequestException(Exception):
        pass

    class HTTPError(RequestException):
        pass

    class Session:
        pass

    requests_stub.RequestException = RequestException
    requests_stub.HTTPError = HTTPError
    requests_stub.Session = Session
    sys.modules["requests"] = requests_stub

if "dotenv" not in sys.modules:
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv_stub

if "google.generativeai" not in sys.modules:
    google_stub = sys.modules.get("google") or types.ModuleType("google")
    generativeai_stub = types.ModuleType("google.generativeai")
    generativeai_stub.configure = lambda *args, **kwargs: None
    generativeai_stub.embed_content = lambda *args, **kwargs: {"embedding": [0.0] * 768}
    google_stub.generativeai = generativeai_stub
    sys.modules["google"] = google_stub
    sys.modules["google.generativeai"] = generativeai_stub

if "supabase" not in sys.modules:
    supabase_stub = types.ModuleType("supabase")

    class Client:
        pass

    def create_client(*args, **kwargs):
        return None

    supabase_stub.Client = Client
    supabase_stub.create_client = create_client
    sys.modules["supabase"] = supabase_stub

import requests

from integrations.brokerage import AlpacaTradingClient
from services.portfolio_service import get_portfolio_snapshot, sync_portfolio_from_brokerage


class FakeResponse:
    def __init__(self, payload, status_code: int = 200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, headers=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "timeout": timeout})
        return self.responses.pop(0)


class BrokerageIntegrationTests(unittest.TestCase):
    def test_alpaca_client_fetches_account_and_positions(self):
        session = FakeSession(
            [
                FakeResponse(
                    {
                        "id": "acct-1",
                        "account_number": "PA123456",
                        "equity": "10250.44",
                        "buying_power": "5000.00",
                        "currency": "USD",
                    }
                ),
                FakeResponse(
                    [
                        {
                            "symbol": "AAPL",
                            "qty": "2.5",
                            "avg_entry_price": "180.10",
                            "current_price": "192.40",
                            "market_value": "481.00",
                            "unrealized_pl": "30.75",
                        },
                        {
                            "symbol": "CASH",
                            "qty": "0",
                            "avg_entry_price": "1",
                        },
                    ]
                ),
            ]
        )

        client = AlpacaTradingClient(
            api_key="key",
            api_secret="secret",
            base_url="https://paper-api.alpaca.markets",
            session=session,
        )
        snapshot = client.fetch_account_snapshot()

        self.assertEqual(snapshot.provider, "alpaca")
        self.assertEqual(snapshot.account_number, "PA123456")
        self.assertAlmostEqual(snapshot.equity or 0.0, 10250.44)
        self.assertEqual(len(snapshot.holdings), 1)
        self.assertEqual(snapshot.holdings[0].symbol, "AAPL")
        self.assertAlmostEqual(snapshot.holdings[0].quantity, 2.5)
        self.assertAlmostEqual(snapshot.holdings[0].cost_basis, 180.10)
        self.assertEqual(session.calls[0]["url"], "https://paper-api.alpaca.markets/v2/account")
        self.assertEqual(session.calls[1]["url"], "https://paper-api.alpaca.markets/v2/positions")

    @patch("services.portfolio_service.save_user_preference")
    @patch("services.portfolio_service.replace_portfolio_holdings")
    @patch("services.portfolio_service.read_brokerage_portfolio")
    def test_sync_portfolio_from_brokerage_persists_snapshot(
        self,
        mock_read_brokerage_portfolio,
        mock_replace_portfolio_holdings,
        mock_save_user_preference,
    ):
        mock_read_brokerage_portfolio.return_value = {
            "ok": True,
            "provider": "alpaca",
            "fetched_at": "2026-05-24T20:00:00+00:00",
            "account_id": "acct-1",
            "account_number": "PA123456",
            "currency": "USD",
            "equity": 1000.0,
            "buying_power": 400.0,
            "holdings": [
                {
                    "symbol": "MSFT",
                    "quantity": 3.0,
                    "cost_basis": 410.0,
                    "market_value": 1290.0,
                }
            ],
        }
        mock_replace_portfolio_holdings.return_value = {
            "ok": True,
            "persisted": True,
            "removed_symbols": ["TSLA"],
            "upserted_count": 1,
        }

        result = sync_portfolio_from_brokerage()

        self.assertTrue(result["ok"])
        self.assertTrue(result["persisted"])
        self.assertEqual(result["imported_count"], 1)
        self.assertEqual(result["removed_symbols"], ["TSLA"])
        self.assertEqual(result["holdings"][0]["symbol"], "MSFT")
        mock_save_user_preference.assert_called_once()

    @patch("services.portfolio_service.read_brokerage_portfolio")
    @patch("services.portfolio_service.fetch_portfolio")
    @patch("services.portfolio_service.brokerage_enabled")
    def test_get_portfolio_snapshot_falls_back_to_live_broker_when_db_empty(
        self,
        mock_brokerage_enabled,
        mock_fetch_portfolio,
        mock_read_brokerage_portfolio,
    ):
        mock_brokerage_enabled.return_value = True
        mock_fetch_portfolio.return_value = []
        mock_read_brokerage_portfolio.return_value = {
            "ok": True,
            "provider": "alpaca",
            "holdings": [
                {
                    "symbol": "NVDA",
                    "quantity": 4.0,
                    "cost_basis": 900.0,
                }
            ],
        }

        holdings = get_portfolio_snapshot(sync_from_broker=False)

        self.assertEqual(len(holdings), 1)
        self.assertEqual(holdings[0]["symbol"], "NVDA")


if __name__ == "__main__":
    unittest.main()
