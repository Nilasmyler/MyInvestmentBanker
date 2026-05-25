import copy
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

from agents.communication_agent import CommunicationAgent
from database import supabase_client as db_client
from services import portfolio_service


class FakeResponse:
    def __init__(self, data):
        self.data = data


class FakeTable:
    def __init__(self, store, table_name: str):
        self.store = store
        self.table_name = table_name
        self.operation = "select"
        self.payload = None
        self.filters = []
        self.order_field = None
        self.order_desc = False
        self.limit_count = None
        self.delete_called = False
        self.selected_fields = "*"

    def select(self, _fields="*"):
        self.operation = "select"
        self.selected_fields = _fields
        return self

    def eq(self, field, value):
        self.filters.append((field, value))
        return self

    def order(self, field, desc=False):
        self.order_field = field
        self.order_desc = desc
        return self

    def limit(self, count):
        self.limit_count = count
        return self

    def insert(self, payload):
        self.operation = "insert"
        self.payload = payload
        return self

    def upsert(self, payload):
        self.operation = "upsert"
        self.payload = payload
        return self

    def update(self, payload):
        self.operation = "update"
        self.payload = payload
        return self

    def delete(self):
        self.delete_called = True
        self.operation = "delete"
        return self

    def execute(self):
        rows = self.store[self.table_name]
        if self.operation == "select":
            self._raise_if_unsupported_news_digest_columns()
            filtered = [copy.deepcopy(row) for row in rows if self._matches(row)]
            if self.order_field:
                filtered.sort(key=lambda row: row.get(self.order_field), reverse=self.order_desc)
            if self.limit_count is not None:
                filtered = filtered[: self.limit_count]
            return FakeResponse(filtered)

        if self.operation == "insert":
            payloads = self.payload if isinstance(self.payload, list) else [self.payload]
            inserted = []
            for payload in payloads:
                self._raise_if_unsupported_news_digest_columns(payload)
                record = copy.deepcopy(payload)
                if "id" not in record:
                    record["id"] = f"{self.table_name}-{len(rows) + 1}"
                rows.append(record)
                inserted.append(copy.deepcopy(record))
            return FakeResponse(inserted)

        if self.operation == "upsert":
            payloads = self.payload if isinstance(self.payload, list) else [self.payload]
            upserted = []
            for payload in payloads:
                record = copy.deepcopy(payload)
                key_field = {"portfolio_holdings": "symbol", "investment_thesis": "symbol", "user_preferences": "key"}.get(
                    self.table_name,
                    "id",
                )
                existing = None
                for row in rows:
                    if row.get(key_field) == record.get(key_field):
                        existing = row
                        break
                if existing is not None:
                    existing.update(record)
                    upserted.append(copy.deepcopy(existing))
                else:
                    if "id" not in record and key_field == "id":
                        record["id"] = f"{self.table_name}-{len(rows) + 1}"
                    rows.append(record)
                    upserted.append(copy.deepcopy(record))
            return FakeResponse(upserted)

        if self.operation == "update":
            updated = []
            for row in rows:
                if self._matches(row):
                    row.update(copy.deepcopy(self.payload))
                    updated.append(copy.deepcopy(row))
            return FakeResponse(updated)

        if self.operation == "delete":
            remaining = [row for row in rows if not self._matches(row)]
            self.store[self.table_name] = remaining
            return FakeResponse([])

        return FakeResponse([])

    def _matches(self, row):
        return all(row.get(field) == value for field, value in self.filters)

    def _raise_if_unsupported_news_digest_columns(self, payload=None):
        if self.table_name != "news_digests":
            return
        unsupported_columns = self.store.get("__unsupported_news_digest_columns__", set())
        referenced_columns = {field for field, _value in self.filters}
        if isinstance(self.selected_fields, str):
            referenced_columns.update(part.strip() for part in self.selected_fields.split(","))
        if payload:
            referenced_columns.update(payload.keys())
        for column in unsupported_columns:
            if column in referenced_columns:
                if column in {"entity_type", "entity_key"}:
                    raise Exception({"message": f"column news_digests.{column} does not exist"})
                raise Exception({"message": f"Could not find the '{column}' column of 'news_digests' in the schema cache"})


class FakeSupabaseClient:
    def __init__(self, store, unsupported_news_digest_columns=None):
        self.store = store
        self.store["__unsupported_news_digest_columns__"] = unsupported_news_digest_columns or set()

    def table(self, table_name: str):
        self.store.setdefault(table_name, [])
        return FakeTable(self.store, table_name)


class SupabasePersistenceTests(unittest.TestCase):
    def setUp(self):
        db_client.NEWS_DIGEST_UNSUPPORTED_COLUMNS.clear()
        self.store = {
            "portfolio_holdings": [
                {"symbol": "MSFT", "name": "Microsoft", "quantity": 5.0, "cost_basis": 410.0},
            ],
            "corporate_analyst_memos": [
                {
                    "id": "memo-1",
                    "symbol": "AAPL",
                    "period": "Q2_2026",
                    "memo_text": "Old memo",
                    "metrics": {"pe_ratio": 30},
                    "created_at": "2026-05-01T10:00:00+00:00",
                },
                {
                    "id": "memo-2",
                    "symbol": "AAPL",
                    "period": "Q2_2026",
                    "memo_text": "Duplicate memo",
                    "metrics": {"pe_ratio": 31},
                    "created_at": "2026-04-30T10:00:00+00:00",
                },
                {
                    "id": "memo-3",
                    "symbol": "AAPL",
                    "period": "Q1_2026",
                    "memo_text": "Prior quarter",
                    "metrics": {"pe_ratio": 28},
                    "created_at": "2026-02-01T10:00:00+00:00",
                },
            ],
            "news_digests": [
                {
                    "id": "news-1",
                    "entity_type": "symbol",
                    "entity_key": "NVDA",
                    "title": "NVIDIA lands new datacenter deal",
                    "summary": "Original summary",
                    "url": "https://example.com/nvda-deal",
                    "published_at": "2026-05-18T10:00:00+00:00",
                }
            ],
        }
        self.fake_client = FakeSupabaseClient(self.store)
        self.supabase_patch = patch.object(db_client, "supabase_client", self.fake_client)
        self.supabase_patch.start()

    def tearDown(self):
        self.supabase_patch.stop()
        db_client.NEWS_DIGEST_UNSUPPORTED_COLUMNS.clear()

    def test_update_portfolio_holding_retires_position_without_delete(self):
        success = db_client.update_portfolio_holding("MSFT", 0, 0)

        self.assertTrue(success)
        self.assertEqual(len(self.store["portfolio_holdings"]), 1)
        self.assertEqual(self.store["portfolio_holdings"][0]["symbol"], "MSFT")
        self.assertEqual(self.store["portfolio_holdings"][0]["quantity"], 0.0)
        self.assertEqual(self.store["portfolio_holdings"][0]["cost_basis"], 410.0)

    def test_save_analyst_memo_updates_existing_same_period_record(self):
        success = db_client.save_analyst_memo("AAPL", "Q2_2026", "Updated memo", {"pe_ratio": 32})

        self.assertTrue(success)
        self.assertEqual(len(self.store["corporate_analyst_memos"]), 3)
        latest_same_period = next(row for row in self.store["corporate_analyst_memos"] if row["id"] == "memo-1")
        self.assertEqual(latest_same_period["memo_text"], "Updated memo")
        self.assertEqual(latest_same_period["metrics"]["pe_ratio"], 32)

    def test_fetch_historical_memos_dedupes_same_period_entries(self):
        memos = db_client.fetch_historical_memos("AAPL", limit=2)

        self.assertEqual([memo["period"] for memo in memos], ["Q2_2026", "Q1_2026"])

    @patch("database.supabase_client.get_embedding", return_value=[0.0] * 768)
    def test_cache_news_digest_skips_duplicate_url(self, _mock_embedding):
        success = db_client.cache_news_digest(
            symbol="NVDA",
            title="NVIDIA lands new datacenter deal",
            summary="Duplicate summary",
            url="https://example.com/nvda-deal",
            published_at="2026-05-18T10:00:00Z",
            entity_type="symbol",
            entity_key="NVDA",
        )

        self.assertTrue(success)
        self.assertEqual(len(self.store["news_digests"]), 1)

    @patch("database.supabase_client.get_embedding", return_value=[0.0] * 768)
    def test_cache_news_digest_falls_back_when_legacy_table_lacks_optional_columns(self, _mock_embedding):
        self.supabase_patch.stop()
        legacy_store = {"news_digests": []}
        legacy_client = FakeSupabaseClient(
            legacy_store,
            unsupported_news_digest_columns={"entity_type", "entity_key", "metadata", "article_vector"},
        )
        with patch.object(db_client, "supabase_client", legacy_client):
            db_client.NEWS_DIGEST_UNSUPPORTED_COLUMNS.clear()
            success = db_client.cache_news_digest(
                symbol="NVDA",
                title="Legacy cache path",
                summary="Older schema still accepts the row.",
                url="https://example.com/legacy",
                published_at="2026-05-18T10:00:00Z",
                entity_type="symbol",
                entity_key="NVDA",
                metadata={"event_type": "contract"},
            )

        self.assertTrue(success)
        self.assertEqual(len(legacy_store["news_digests"]), 1)
        saved_row = legacy_store["news_digests"][0]
        self.assertEqual(saved_row["symbol"], "NVDA")
        self.assertNotIn("entity_type", saved_row)
        self.assertNotIn("entity_key", saved_row)
        self.assertNotIn("metadata", saved_row)
        self.assertNotIn("article_vector", saved_row)


class PortfolioSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.preference_store = {}

        def fake_get(key):
            return copy.deepcopy(self.preference_store.get(key))

        def fake_save(key, value):
            self.preference_store[key] = copy.deepcopy(value)
            return True

        self.get_patch = patch.object(portfolio_service, "get_user_preference", side_effect=fake_get)
        self.save_patch = patch.object(portfolio_service, "save_user_preference", side_effect=fake_save)
        self.get_patch.start()
        self.save_patch.start()

    def tearDown(self):
        self.get_patch.stop()
        self.save_patch.stop()

    def test_build_portfolio_change_summary_compares_against_prior_weekly_snapshot(self):
        first = portfolio_service.build_portfolio_change_summary(
            [{"symbol": "MSFT", "quantity": 2.0, "cost_basis": 400.0, "market_value": 900.0}]
        )
        self.assertIsNone(first["baseline_snapshot"])

        self.preference_store[portfolio_service.PORTFOLIO_SNAPSHOT_HISTORY_KEY][0]["captured_at"] = "2020-01-01T09:00:00+00:00"
        second = portfolio_service.build_portfolio_change_summary(
            [
                {"symbol": "MSFT", "quantity": 2.0, "cost_basis": 400.0, "market_value": 980.0},
                {"symbol": "NVDA", "quantity": 1.0, "cost_basis": 900.0, "market_value": 1025.0},
            ]
        )

        self.assertEqual(second["market_value_change"], 1105.0)
        self.assertEqual(second["new_positions"], ["NVDA"])
        self.assertEqual(second["removed_positions"], [])
        self.assertEqual(second["largest_position_changes"][0]["symbol"], "MSFT")


class CommunicationReportTests(unittest.TestCase):
    @patch("agents.communication_agent.llm_available", return_value=False)
    @patch(
        "agents.communication_agent.build_portfolio_change_summary",
        return_value={
            "current_snapshot": {"total_market_value": 1500.0, "total_cost_basis": 1200.0},
            "baseline_snapshot": {"captured_at": "2026-05-19T09:00:00+00:00"},
            "elapsed_days": 7,
            "market_value_change": 150.0,
            "market_value_change_pct": 11.11,
            "cost_basis_change": 0.0,
            "new_positions": ["NVDA"],
            "removed_positions": ["TSLA"],
            "largest_position_changes": [{"symbol": "MSFT", "delta": 100.0, "delta_pct": 8.0}],
        },
    )
    def test_compile_synthesis_report_includes_snapshot_summary(self, _mock_change_summary, _mock_llm_available):
        report = CommunicationAgent.compile_synthesis_report(
            portfolio_state=[{"symbol": "MSFT", "quantity": 2.0, "cost_basis": 400.0, "market_value": 900.0}],
            macro_data={"fed_funds_rate": 4.75, "cpi_inflation_index": 320.0},
            cfa_memos=[],
            risk_memos={"summary": "Risk summary"},
        )

        self.assertIn("📊 **Snapshot Change Summary**", report)
        self.assertIn("Portfolio value is up `150.00` USD (+11.11%)", report)
        self.assertIn("New active positions since then: `NVDA`.", report)
        self.assertIn("Positions that left the active book since then: `TSLA`.", report)


if __name__ == "__main__":
    unittest.main()
