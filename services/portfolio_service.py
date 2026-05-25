import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from database.supabase_client import fetch_portfolio, replace_portfolio_holdings, save_user_preference
from integrations.brokerage import (
    BrokerageAccountSnapshot,
    BrokerageClientError,
    brokerage_is_configured,
    build_brokerage_client_from_env,
    get_configured_brokerage_provider,
)

load_dotenv()

logger = logging.getLogger("MyInvestmentBanker.services.portfolio")
logging.basicConfig(level=logging.INFO)

BROKERAGE_SYNC_STATUS_KEY = "brokerage_sync_status"


def _env_flag(key: str, default: bool) -> bool:
    raw_value = os.getenv(key)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def should_sync_portfolio_on_read() -> bool:
    return _env_flag("BROKERAGE_SYNC_ON_READ", False)


def should_sync_portfolio_before_analysis() -> bool:
    return _env_flag("BROKERAGE_SYNC_BEFORE_RUNS", True)


def brokerage_enabled() -> bool:
    return brokerage_is_configured()


def _snapshot_to_holdings(snapshot: BrokerageAccountSnapshot) -> List[Dict[str, Any]]:
    return [holding.to_portfolio_record() for holding in snapshot.holdings]


def _build_sync_payload(snapshot: BrokerageAccountSnapshot, persisted: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    holdings = _snapshot_to_holdings(snapshot)
    return {
        "ok": True,
        "provider": snapshot.provider,
        "fetched_at": snapshot.fetched_at,
        "account_id": snapshot.account_id,
        "account_number": snapshot.account_number,
        "currency": snapshot.currency,
        "equity": snapshot.equity,
        "buying_power": snapshot.buying_power,
        "holdings": holdings,
        "imported_count": len(holdings),
        "persisted": bool(persisted and persisted.get("persisted")),
        "removed_symbols": (persisted or {}).get("removed_symbols", []),
    }


def _save_sync_status(result: Dict[str, Any]) -> None:
    save_user_preference(
        BROKERAGE_SYNC_STATUS_KEY,
        {
            "provider": result.get("provider", ""),
            "ok": result.get("ok", False),
            "fetched_at": result.get("fetched_at", ""),
            "imported_count": result.get("imported_count", 0),
            "removed_symbols": result.get("removed_symbols", []),
            "persisted": result.get("persisted", False),
            "account_id": result.get("account_id", ""),
            "account_number": result.get("account_number", ""),
            "equity": result.get("equity"),
        },
    )


def read_brokerage_portfolio() -> Dict[str, Any]:
    provider = get_configured_brokerage_provider()
    if not provider:
        return {
            "ok": False,
            "provider": "",
            "reason": "BROKERAGE_PROVIDER is not configured.",
            "holdings": [],
        }

    try:
        client = build_brokerage_client_from_env()
        if client is None:
            return {
                "ok": False,
                "provider": provider,
                "reason": "Brokerage provider is not configured.",
                "holdings": [],
            }
        snapshot = client.fetch_account_snapshot()
        return _build_sync_payload(snapshot)
    except BrokerageClientError as exc:
        logger.error(f"Brokerage read failed: {exc}")
        return {
            "ok": False,
            "provider": provider,
            "reason": str(exc),
            "holdings": [],
        }
    except Exception as exc:
        logger.error(f"Unexpected brokerage read failure: {exc}")
        return {
            "ok": False,
            "provider": provider,
            "reason": str(exc),
            "holdings": [],
        }


def sync_portfolio_from_brokerage() -> Dict[str, Any]:
    live_result = read_brokerage_portfolio()
    if not live_result.get("ok"):
        return live_result

    snapshot = BrokerageAccountSnapshot(
        provider=live_result.get("provider", ""),
        holdings=[],
        fetched_at=str(live_result.get("fetched_at", "")),
        account_id=str(live_result.get("account_id", "")),
        account_number=str(live_result.get("account_number", "")),
        equity=live_result.get("equity"),
        buying_power=live_result.get("buying_power"),
        currency=str(live_result.get("currency", "USD")),
    )
    holdings = live_result.get("holdings", [])
    persisted = replace_portfolio_holdings(holdings)
    result = _build_sync_payload(snapshot, persisted=persisted)
    result["holdings"] = holdings
    result["imported_count"] = len(holdings)
    result["persisted"] = bool(persisted.get("persisted"))
    result["removed_symbols"] = persisted.get("removed_symbols", [])
    if persisted.get("reason"):
        result["persistence_warning"] = persisted["reason"]
    _save_sync_status(result)
    return result


def get_portfolio_snapshot(sync_from_broker: bool = False) -> List[Dict[str, Any]]:
    if sync_from_broker and brokerage_enabled():
        result = sync_portfolio_from_brokerage()
        if result.get("ok"):
            return result.get("holdings", [])
        logger.warning(f"Brokerage sync failed, falling back to stored portfolio: {result.get('reason')}")

    holdings = fetch_portfolio()
    if holdings:
        return holdings

    if brokerage_enabled():
        live_result = read_brokerage_portfolio()
        if live_result.get("ok"):
            return live_result.get("holdings", [])

    return []
