import logging
import os
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from database.supabase_client import fetch_portfolio, get_user_preference, replace_portfolio_holdings, save_user_preference
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
PORTFOLIO_SNAPSHOT_HISTORY_KEY = "portfolio_snapshot_history"
PORTFOLIO_SNAPSHOT_HISTORY_LIMIT = 16
SNAPSHOT_RECENT_DEDUPE_SECONDS = 2 * 60 * 60


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


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_iso_datetime(raw_value: str) -> Optional[datetime]:
    if not raw_value:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw_value).replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _snapshot_signature(holdings: List[Dict[str, Any]]) -> str:
    sanitized = []
    for holding in holdings:
        sanitized.append(
            {
                "symbol": str(holding.get("symbol", "")).upper().strip(),
                "quantity": _safe_float(holding.get("quantity")) or 0.0,
                "cost_basis": _safe_float(holding.get("cost_basis")) or 0.0,
                "current_price": _safe_float(holding.get("current_price")),
                "market_value": _safe_float(holding.get("market_value")),
            }
        )
    sanitized.sort(key=lambda row: row["symbol"])
    return json.dumps(sanitized, sort_keys=True, separators=(",", ":"))


def summarize_portfolio_state(holdings: List[Dict[str, Any]]) -> Dict[str, Any]:
    cleaned_holdings = []
    total_cost_basis = 0.0
    total_market_value = 0.0
    market_values_available = False
    total_unrealized_pl = 0.0

    for holding in holdings:
        symbol = str(holding.get("symbol", "")).upper().strip()
        if not symbol:
            continue

        quantity = _safe_float(holding.get("quantity")) or 0.0
        cost_basis = _safe_float(holding.get("cost_basis")) or 0.0
        current_price = _safe_float(holding.get("current_price"))
        market_value = _safe_float(holding.get("market_value"))
        name = str(holding.get("name", "") or "")

        total_cost_basis += quantity * cost_basis
        if market_value is not None:
            total_market_value += market_value
            total_unrealized_pl += market_value - (quantity * cost_basis)
            market_values_available = True

        cleaned_holding = {
            "symbol": symbol,
            "quantity": quantity,
            "cost_basis": cost_basis,
        }
        if name:
            cleaned_holding["name"] = name
        if current_price is not None:
            cleaned_holding["current_price"] = current_price
        if market_value is not None:
            cleaned_holding["market_value"] = market_value
        cleaned_holdings.append(cleaned_holding)

    cleaned_holdings.sort(key=lambda row: row["symbol"])
    return {
        "holdings": cleaned_holdings,
        "holdings_count": len(cleaned_holdings),
        "total_cost_basis": round(total_cost_basis, 2),
        "total_market_value": round(total_market_value, 2) if market_values_available else None,
        "market_values_available": market_values_available,
        "total_unrealized_pl": round(total_unrealized_pl, 2) if market_values_available else None,
        "signature": _snapshot_signature(cleaned_holdings),
    }


def get_portfolio_snapshot_history(limit: int = PORTFOLIO_SNAPSHOT_HISTORY_LIMIT) -> List[Dict[str, Any]]:
    raw_history = get_user_preference(PORTFOLIO_SNAPSHOT_HISTORY_KEY)
    if not isinstance(raw_history, list):
        return []

    normalized_history = [item for item in raw_history if isinstance(item, dict)]
    normalized_history.sort(
        key=lambda item: _parse_iso_datetime(str(item.get("captured_at", ""))) or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    return normalized_history[:limit]


def record_portfolio_snapshot(
    holdings: List[Dict[str, Any]],
    source: str,
    captured_at: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    summary = summarize_portfolio_state(holdings)
    snapshot_time = _parse_iso_datetime(captured_at) or datetime.now(timezone.utc)
    snapshot = {
        "captured_at": snapshot_time.replace(microsecond=0).isoformat(),
        "source": source,
        "holdings_count": summary["holdings_count"],
        "total_cost_basis": summary["total_cost_basis"],
        "total_market_value": summary["total_market_value"],
        "market_values_available": summary["market_values_available"],
        "total_unrealized_pl": summary["total_unrealized_pl"],
        "holdings": summary["holdings"],
        "signature": summary["signature"],
        "metadata": metadata or {},
    }

    history = get_portfolio_snapshot_history(limit=PORTFOLIO_SNAPSHOT_HISTORY_LIMIT)
    latest = history[0] if history else None
    if latest:
        latest_time = _parse_iso_datetime(str(latest.get("captured_at", "")))
        if latest.get("signature") == snapshot["signature"] and latest_time:
            if latest.get("captured_at") == snapshot["captured_at"]:
                return latest
            time_delta_seconds = (snapshot_time - latest_time).total_seconds()
            if 0 <= time_delta_seconds <= SNAPSHOT_RECENT_DEDUPE_SECONDS:
                return latest

    updated_history = [snapshot]
    updated_history.extend(history)
    save_user_preference(PORTFOLIO_SNAPSHOT_HISTORY_KEY, updated_history[:PORTFOLIO_SNAPSHOT_HISTORY_LIMIT])
    return snapshot


def build_portfolio_change_summary(holdings: List[Dict[str, Any]], baseline_days: int = 6) -> Dict[str, Any]:
    current_snapshot = record_portfolio_snapshot(holdings, source="portfolio_digest")
    if not current_snapshot:
        return {"current_snapshot": None, "baseline_snapshot": None}

    history = get_portfolio_snapshot_history(limit=PORTFOLIO_SNAPSHOT_HISTORY_LIMIT)
    current_time = _parse_iso_datetime(str(current_snapshot.get("captured_at", "")))
    baseline_snapshot = None
    fallback_snapshot = None

    for candidate in history:
        if candidate.get("captured_at") == current_snapshot.get("captured_at") and candidate.get("signature") == current_snapshot.get("signature"):
            continue

        candidate_time = _parse_iso_datetime(str(candidate.get("captured_at", "")))
        if not fallback_snapshot:
            fallback_snapshot = candidate
        if candidate_time and current_time and (current_time - candidate_time).days >= baseline_days:
            baseline_snapshot = candidate
            break

    baseline_snapshot = baseline_snapshot or fallback_snapshot
    if not baseline_snapshot:
        return {
            "current_snapshot": current_snapshot,
            "baseline_snapshot": None,
            "market_value_change": None,
            "market_value_change_pct": None,
            "cost_basis_change": None,
            "new_positions": [],
            "removed_positions": [],
            "largest_position_changes": [],
        }

    current_holdings = {item["symbol"]: item for item in current_snapshot.get("holdings", []) if item.get("symbol")}
    baseline_holdings = {item["symbol"]: item for item in baseline_snapshot.get("holdings", []) if item.get("symbol")}
    current_total_market_value = current_snapshot.get("total_market_value")
    baseline_total_market_value = baseline_snapshot.get("total_market_value")

    market_value_change = None
    market_value_change_pct = None
    if current_total_market_value is not None and baseline_total_market_value is not None:
        market_value_change = round(float(current_total_market_value) - float(baseline_total_market_value), 2)
        if baseline_total_market_value:
            market_value_change_pct = round((market_value_change / float(baseline_total_market_value)) * 100, 2)

    largest_position_changes = []
    for symbol in sorted(set(current_holdings.keys()) & set(baseline_holdings.keys())):
        current_market_value = current_holdings[symbol].get("market_value")
        baseline_market_value = baseline_holdings[symbol].get("market_value")
        if current_market_value is None or baseline_market_value is None:
            continue
        delta = round(float(current_market_value) - float(baseline_market_value), 2)
        if delta == 0:
            continue
        delta_pct = None
        if baseline_market_value:
            delta_pct = round((delta / float(baseline_market_value)) * 100, 2)
        largest_position_changes.append({"symbol": symbol, "delta": delta, "delta_pct": delta_pct})

    largest_position_changes.sort(key=lambda item: abs(item["delta"]), reverse=True)

    baseline_time = _parse_iso_datetime(str(baseline_snapshot.get("captured_at", "")))
    elapsed_days = None
    if current_time and baseline_time:
        elapsed_days = max((current_time - baseline_time).days, 0)

    return {
        "current_snapshot": current_snapshot,
        "baseline_snapshot": baseline_snapshot,
        "elapsed_days": elapsed_days,
        "market_value_change": market_value_change,
        "market_value_change_pct": market_value_change_pct,
        "cost_basis_change": round(
            float(current_snapshot.get("total_cost_basis", 0.0)) - float(baseline_snapshot.get("total_cost_basis", 0.0)),
            2,
        ),
        "new_positions": sorted(set(current_holdings.keys()) - set(baseline_holdings.keys())),
        "removed_positions": sorted(set(baseline_holdings.keys()) - set(current_holdings.keys())),
        "largest_position_changes": largest_position_changes[:3],
    }


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
    record_portfolio_snapshot(
        holdings,
        source=f"{result.get('provider', 'brokerage')}_sync",
        captured_at=str(result.get("fetched_at", "")),
        metadata={
            "provider": result.get("provider", ""),
            "account_id": result.get("account_id", ""),
            "account_number": result.get("account_number", ""),
            "equity": result.get("equity"),
            "buying_power": result.get("buying_power"),
            "currency": result.get("currency", "USD"),
        },
    )
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
