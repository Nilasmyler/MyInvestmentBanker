import importlib
import os
import sys
import types
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient


def install_main_test_stubs():
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv_stub

    agents_package = types.ModuleType("agents")
    agents_package.__path__ = []
    sys.modules["agents"] = agents_package

    communication_stub = types.ModuleType("agents.communication_agent")

    class CommunicationAgent:
        @staticmethod
        def prepare_portfolio_follow_up(chat_id):
            return ""

        @staticmethod
        def prepare_discovery_follow_up(chat_id, result):
            return ""

        @staticmethod
        def generate_single_stock_analysis(chat_id, symbol, user_context=""):
            return f"Analysis for {symbol}"

        @staticmethod
        def detect_stock_analysis_request(message_text):
            return None

        @staticmethod
        def parse_user_command(chat_id, message_text):
            return f"Echo: {message_text}"

    communication_stub.CommunicationAgent = CommunicationAgent
    sys.modules["agents.communication_agent"] = communication_stub

    orchestrator_stub = types.ModuleType("agents.orchestrator")
    orchestrator_stub.trigger_autonomous_discovery_check = (
        lambda return_result=False: {"bulletin": "sweep"} if return_result else "sweep"
    )
    orchestrator_stub.trigger_opportunity_discovery = (
        lambda run_type="deep", return_result=False: {"bulletin": f"{run_type} bulletin"} if return_result else run_type
    )
    orchestrator_stub.trigger_wealth_manager_run = lambda: "portfolio bulletin"
    sys.modules["agents.orchestrator"] = orchestrator_stub

    database_package = types.ModuleType("database")
    database_package.__path__ = []
    sys.modules["database"] = database_package

    supabase_client_stub = types.ModuleType("database.supabase_client")
    supabase_client_stub.log_chat_message = lambda *args, **kwargs: None
    sys.modules["database.supabase_client"] = supabase_client_stub


class MainRouteSecurityTests(unittest.TestCase):
    @staticmethod
    def restore_modules(original_modules):
        for name, module in original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def load_app(self, env):
        module_names = [
            "dotenv",
            "agents",
            "agents.communication_agent",
            "agents.orchestrator",
            "database",
            "database.supabase_client",
            "main",
        ]
        original_modules = {name: sys.modules.get(name) for name in module_names}
        self.addCleanup(self.restore_modules, original_modules)

        install_main_test_stubs()

        env_patcher = patch.dict(os.environ, env, clear=True)
        env_patcher.start()
        self.addCleanup(env_patcher.stop)

        sys.modules.pop("main", None)
        module = importlib.import_module("main")
        module = importlib.reload(module)

        recorded_calls = {
            "pipeline": [],
            "discovery": [],
            "analysis": [],
            "send": [],
            "log": [],
        }

        async def fake_pipeline(chat_id):
            recorded_calls["pipeline"].append(chat_id)

        async def fake_discovery(chat_id, run_type="deep"):
            recorded_calls["discovery"].append((chat_id, run_type))

        async def fake_analysis(chat_id, symbol, user_context=""):
            recorded_calls["analysis"].append((chat_id, symbol, user_context))

        async def fake_send(chat_id, text):
            recorded_calls["send"].append((chat_id, text))
            return True

        def fake_log(*args, **kwargs):
            recorded_calls["log"].append((args, kwargs))

        module.background_pipeline_execution = fake_pipeline
        module.background_discovery_execution = fake_discovery
        module.background_single_stock_analysis = fake_analysis
        module.send_telegram_message = fake_send
        module.log_chat_message = fake_log
        module.get_telegram_bot_token = lambda: env.get("TELEGRAM_BOT_TOKEN", "").strip()
        module.get_telegram_user_id = lambda: env.get("TELEGRAM_USER_ID", "").strip()
        module.get_telegram_webhook_secret = lambda: env.get("TELEGRAM_WEBHOOK_SECRET", "").strip()
        module.get_scheduled_run_secret = lambda: env.get("SCHEDULED_RUN_SECRET", "").strip()

        return module, TestClient(module.app), recorded_calls

    def test_telegram_webhook_accepts_authorized_update_request(self):
        _, client, recorded_calls = self.load_app(
            {
                "TELEGRAM_BOT_TOKEN": "bot-token",
                "TELEGRAM_USER_ID": "12345",
                "TELEGRAM_WEBHOOK_SECRET": "telegram-secret",
            }
        )

        response = client.post(
            "/telegram-webhook",
            headers={"X-Telegram-Bot-Api-Secret-Token": "telegram-secret"},
            json={
                "message": {
                    "chat": {"id": "12345"},
                    "from": {"id": "12345", "username": "owner"},
                    "text": "/update",
                }
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "processing")
        self.assertEqual(recorded_calls["pipeline"], ["12345"])
        self.assertEqual(len(recorded_calls["log"]), 1)

    def test_telegram_webhook_rejects_invalid_secret_token(self):
        _, client, recorded_calls = self.load_app(
            {
                "TELEGRAM_BOT_TOKEN": "bot-token",
                "TELEGRAM_USER_ID": "12345",
                "TELEGRAM_WEBHOOK_SECRET": "telegram-secret",
            }
        )

        response = client.post(
            "/telegram-webhook",
            headers={"X-Telegram-Bot-Api-Secret-Token": "wrong-secret"},
            json={
                "message": {
                    "chat": {"id": "12345"},
                    "from": {"id": "12345", "username": "owner"},
                    "text": "/update",
                }
            },
        )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(recorded_calls["pipeline"], [])
        self.assertEqual(recorded_calls["log"], [])

    def test_telegram_webhook_rejects_unauthorized_sender(self):
        _, client, recorded_calls = self.load_app(
            {
                "TELEGRAM_BOT_TOKEN": "bot-token",
                "TELEGRAM_USER_ID": "12345",
                "TELEGRAM_WEBHOOK_SECRET": "telegram-secret",
            }
        )

        response = client.post(
            "/telegram-webhook",
            headers={"X-Telegram-Bot-Api-Secret-Token": "telegram-secret"},
            json={
                "message": {
                    "chat": {"id": "99999"},
                    "from": {"id": "99999", "username": "intruder"},
                    "text": "/update",
                }
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "rejected")
        self.assertEqual(recorded_calls["pipeline"], [])
        self.assertEqual(recorded_calls["send"], [])
        self.assertEqual(recorded_calls["log"], [])

    def test_telegram_webhook_requires_user_id_for_live_mode(self):
        _, client, recorded_calls = self.load_app(
            {
                "TELEGRAM_BOT_TOKEN": "bot-token",
                "TELEGRAM_WEBHOOK_SECRET": "telegram-secret",
            }
        )

        response = client.post(
            "/telegram-webhook",
            headers={"X-Telegram-Bot-Api-Secret-Token": "telegram-secret"},
            json={
                "message": {
                    "chat": {"id": "12345"},
                    "from": {"id": "12345", "username": "owner"},
                    "text": "/update",
                }
            },
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(recorded_calls["pipeline"], [])
        self.assertEqual(recorded_calls["log"], [])

    def test_scheduled_run_accepts_matching_shared_secret(self):
        _, client, recorded_calls = self.load_app(
            {
                "TELEGRAM_BOT_TOKEN": "bot-token",
                "TELEGRAM_USER_ID": "12345",
                "SCHEDULED_RUN_SECRET": "scheduled-secret",
            }
        )

        response = client.post(
            "/scheduled-run?run_type=daily",
            headers={"X-Scheduled-Run-Secret": "scheduled-secret"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["target_user_id"], "12345")
        self.assertEqual(recorded_calls["pipeline"], ["12345"])

    def test_scheduled_run_rejects_invalid_shared_secret(self):
        _, client, recorded_calls = self.load_app(
            {
                "TELEGRAM_BOT_TOKEN": "bot-token",
                "TELEGRAM_USER_ID": "12345",
                "SCHEDULED_RUN_SECRET": "scheduled-secret",
            }
        )

        response = client.post(
            "/scheduled-run?run_type=daily",
            headers={"X-Scheduled-Run-Secret": "wrong-secret"},
        )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(recorded_calls["pipeline"], [])

    def test_scheduled_run_allows_local_mock_without_live_secrets(self):
        _, client, recorded_calls = self.load_app({})

        response = client.post("/scheduled-run?run_type=daily")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["target_user_id"], "local-mock")
        self.assertEqual(recorded_calls["pipeline"], ["local-mock"])


if __name__ == "__main__":
    unittest.main()
