import sys
import types
import unittest
from unittest.mock import patch

if "requests" not in sys.modules:
    requests_stub = types.ModuleType("requests")
    requests_exceptions_stub = types.ModuleType("requests.exceptions")

    class RequestException(Exception):
        pass

    class HTTPError(RequestException):
        pass

    class Session:
        pass

    requests_stub.RequestException = RequestException
    requests_stub.HTTPError = HTTPError
    requests_stub.Session = Session
    requests_stub.get = lambda *args, **kwargs: None
    requests_exceptions_stub.RequestException = RequestException
    requests_exceptions_stub.HTTPError = HTTPError
    requests_stub.exceptions = requests_exceptions_stub
    sys.modules["requests"] = requests_stub
    sys.modules["requests.exceptions"] = requests_exceptions_stub

if "yfinance" not in sys.modules:
    yfinance_stub = types.ModuleType("yfinance")

    class Ticker:
        def __init__(self, *args, **kwargs):
            pass

    yfinance_stub.Ticker = Ticker
    sys.modules["yfinance"] = yfinance_stub

if "dotenv" not in sys.modules:
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv_stub

if "google.generativeai" not in sys.modules:
    google_stub = sys.modules.get("google") or types.ModuleType("google")
    generativeai_stub = types.ModuleType("google.generativeai")
    generativeai_stub.configure = lambda *args, **kwargs: None
    google_stub.generativeai = generativeai_stub
    sys.modules["google"] = google_stub
    sys.modules["google.generativeai"] = generativeai_stub

if "llmlingua" not in sys.modules:
    llmlingua_stub = types.ModuleType("llmlingua")

    class PromptCompressor:
        def __init__(self, *args, **kwargs):
            pass

        def compress_prompt(self, *args, **kwargs):
            return {"compressed_prompt": ""}

    llmlingua_stub.PromptCompressor = PromptCompressor
    sys.modules["llmlingua"] = llmlingua_stub

if "supabase" not in sys.modules:
    supabase_stub = types.ModuleType("supabase")

    class Client:
        pass

    def create_client(*args, **kwargs):
        return None

    supabase_stub.Client = Client
    supabase_stub.create_client = create_client
    sys.modules["supabase"] = supabase_stub

from agents.research_planner_agent import ResearchPlannerAgent
from agents.ownership_intel_agent import OwnershipIntelAgent
from agents.risk_agent import RiskAgent
from agents.scout_agent import ScoutAgent
from agents.street_consensus_agent import StreetConsensusAgent


class ResearchPlannerTests(unittest.TestCase):
    def test_build_symbol_plan_routes_specialists_for_high_signal_existing_position(self):
        plan = ResearchPlannerAgent.build_symbol_plan(
            "NVDA",
            {
                "market_data": {"five_day_change_pct": 8.4, "current_price": 100.0},
                "material_news": [{"headline": "Datacenter win"}],
                "recent_filings": [{"form": "8-K"}],
            },
            run_type="deep",
            is_existing_position=True,
        )

        self.assertTrue(plan["include_ownership_intel"])
        self.assertTrue(plan["include_street_consensus"])
        self.assertEqual(plan["priority"], "high")
        self.assertIn("existing_position", plan["reasons"])
        self.assertIn("deep_run", plan["reasons"])

    def test_score_candidate_expression_rewards_specialist_evidence(self):
        score = ResearchPlannerAgent.score_candidate_expression(
            {
                "symbol": "MSFT",
                "signal_count": 4,
                "is_existing_position": True,
                "material_news": [{"headline": "x"}],
                "relevant_filings": [{"form": "10-Q"}],
                "ownership_intel": {"signal_strength": "high"},
                "street_consensus": {"signal_strength": "medium"},
            }
        )

        self.assertGreaterEqual(score, 20.0)

    def test_llm_review_symbols_are_budget_capped(self):
        candidate_expressions = [
            {"symbol": "A", "triage_score": 12},
            {"symbol": "B", "triage_score": 11},
            {"symbol": "C", "triage_score": 10},
            {"symbol": "D", "triage_score": 9},
        ]

        self.assertEqual(
            ResearchPlannerAgent.select_llm_review_symbols(candidate_expressions, run_type="sweep"),
            ["A", "B"],
        )
        self.assertEqual(
            ResearchPlannerAgent.select_llm_review_symbols(candidate_expressions, run_type="deep"),
            ["A", "B", "C"],
        )

    def test_llm_theme_keys_prioritize_high_confidence_and_overlap(self):
        theme_keys = ResearchPlannerAgent.select_llm_theme_keys(
            [
                {
                    "theme_key": "software",
                    "confidence_level": "medium",
                    "trigger_sources": ["news"],
                    "portfolio_overlap": [],
                },
                {
                    "theme_key": "semiconductors",
                    "confidence_level": "high",
                    "trigger_sources": ["news", "filings", "market_action"],
                    "portfolio_overlap": ["NVDA"],
                },
                {
                    "theme_key": "cybersecurity",
                    "confidence_level": "high",
                    "trigger_sources": ["news", "market_action"],
                    "portfolio_overlap": [],
                },
            ],
            run_type="deep",
        )

        self.assertEqual(theme_keys, ["semiconductors", "cybersecurity"])
        self.assertEqual(ResearchPlannerAgent.select_llm_theme_keys([], run_type="sweep"), [])

    @patch("agents.ownership_intel_agent.fetch_institutional_holder_snapshot")
    @patch("agents.ownership_intel_agent.fetch_recent_sec_ownership_filings")
    def test_ownership_snapshot_without_fresh_filings_stays_low_signal(
        self,
        mock_fetch_recent_sec_ownership_filings,
        mock_fetch_institutional_holder_snapshot,
    ):
        mock_fetch_recent_sec_ownership_filings.return_value = []
        mock_fetch_institutional_holder_snapshot.return_value = {
            "institutional_ownership_pct": 82.0,
            "insider_ownership_pct": 1.5,
            "top_holders": [{"holder": "Vanguard"}],
        }

        intel = OwnershipIntelAgent.collect_symbol_intel("MSFT")

        self.assertEqual(intel["signal_strength"], "low")
        self.assertIn("no fresh sec ownership catalyst", intel["summary"].lower())
        self.assertNotIn("error", intel)

    @patch("agents.street_consensus_agent.fetch_analyst_consensus")
    def test_street_snapshot_requires_stronger_conditions_for_medium_signal(self, mock_fetch_analyst_consensus):
        mock_fetch_analyst_consensus.return_value = {
            "analyst_count": 4,
            "recommendation_key": "buy",
            "recommendation_mean": 1.9,
            "price_target_premium_pct": 6.0,
            "target_mean_price": 110.0,
            "target_high_price": 120.0,
            "target_low_price": 95.0,
            "recommendation_breakdown": {},
        }

        consensus = StreetConsensusAgent.collect_symbol_consensus("NVDA")

        self.assertEqual(consensus["signal_strength"], "low")
        self.assertIn("does not include a fresh revision signal", consensus["summary"].lower())

    def test_low_signal_specialist_snapshots_do_not_become_catalysts(self):
        candidate_expression = ScoutAgent.build_candidate_expression(
            {"theme_key": "software", "theme_name": "Software", "mapped_etfs": ["IGV"]},
            {
                "symbol": "MSFT",
                "market_data": {"name": "Microsoft", "five_day_change_pct": 0, "current_price": 100.0},
                "material_news": [],
                "recent_filings": [],
                "ownership_intel": {"signal_strength": "low", "summary": "Ownership snapshot is available, but no fresh SEC ownership catalyst was captured."},
                "street_consensus": {"signal_strength": "low", "summary": "Analyst snapshot is available, but no strong consensus signal stood out."},
            },
            portfolio_symbols=set(),
        )

        self.assertEqual(candidate_expression["company_catalysts"], [])
        self.assertIn("No fresh ownership catalyst", candidate_expression["data_gaps"])
        self.assertIn("No strong analyst-consensus signal", candidate_expression["data_gaps"])

    @patch("agents.street_consensus_agent.fetch_analyst_consensus")
    @patch("agents.ownership_intel_agent.fetch_institutional_holder_snapshot")
    @patch("agents.ownership_intel_agent.fetch_recent_sec_ownership_filings")
    def test_specialist_agents_preserve_upstream_errors(
        self,
        mock_fetch_recent_sec_ownership_filings,
        mock_fetch_institutional_holder_snapshot,
        mock_fetch_analyst_consensus,
    ):
        mock_fetch_recent_sec_ownership_filings.return_value = []
        mock_fetch_institutional_holder_snapshot.return_value = {"error": "rate limited", "top_holders": []}
        mock_fetch_analyst_consensus.return_value = {"error": "timeout", "analyst_count": 0}

        ownership_intel = OwnershipIntelAgent.collect_symbol_intel("AAPL")
        street_consensus = StreetConsensusAgent.collect_symbol_consensus("AAPL")

        self.assertEqual(ownership_intel.get("error"), "rate limited")
        self.assertEqual(street_consensus.get("error"), "timeout")

    def test_risk_sanitizer_normalizes_symbols_and_hydrates_deterministic_evidence(self):
        sanitized = RiskAgent._sanitize_llm_audit_result(
            {"theme_key": "semiconductors", "theme_name": "Semiconductors", "invalidators": ["Demand slows"]},
            {
                "recommendations": [{"symbol": " nvda ", "why_now": "Custom text"}],
                "rejected": [],
                "risk_summary": "ok",
            },
            [
                {
                    "symbol": "NVDA",
                    "is_existing_position": False,
                    "thesis_alignment": "Theme fit",
                    "analyst_verdict": "Deterministic verdict",
                    "cautions": ["Valuation"],
                    "source_etf": "SMH",
                    "company_catalysts": ["Catalyst"],
                    "material_news": [{"headline": "News"}],
                    "relevant_filings": [{"form": "8-K"}],
                    "ownership_intel": {"signal_strength": "low"},
                    "street_consensus": {"signal_strength": "low"},
                }
            ],
            3,
        )

        self.assertIsNotNone(sanitized)
        self.assertEqual(sanitized["recommendations"][0]["symbol"], "NVDA")
        self.assertEqual(sanitized["recommendations"][0]["theme_key"], "semiconductors")
        self.assertIn("material_news", sanitized["recommendations"][0]["evidence"])


if __name__ == "__main__":
    unittest.main()
