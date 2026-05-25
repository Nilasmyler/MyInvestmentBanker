import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("MyInvestmentBanker.integrations.brokerage")
logging.basicConfig(level=logging.INFO)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> Optional[float]:
    if value in [None, ""]:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class BrokerageHolding:
    symbol: str
    quantity: float
    cost_basis: float
    name: str = ""
    current_price: Optional[float] = None
    market_value: Optional[float] = None
    unrealized_pl: Optional[float] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_portfolio_record(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "symbol": self.symbol.upper().strip(),
            "name": self.name,
            "quantity": float(self.quantity),
            "cost_basis": float(self.cost_basis),
            "source": "brokerage",
        }
        if self.current_price is not None:
            payload["current_price"] = float(self.current_price)
        if self.market_value is not None:
            payload["market_value"] = float(self.market_value)
        if self.unrealized_pl is not None:
            payload["unrealized_pl"] = float(self.unrealized_pl)
        return payload


@dataclass
class BrokerageAccountSnapshot:
    provider: str
    holdings: List[BrokerageHolding]
    fetched_at: str
    account_id: str = ""
    account_number: str = ""
    equity: Optional[float] = None
    buying_power: Optional[float] = None
    currency: str = "USD"
    raw_account: Dict[str, Any] = field(default_factory=dict)


class BrokerageClientError(RuntimeError):
    pass


class BrokerageClient:
    provider = ""

    def fetch_account_snapshot(self) -> BrokerageAccountSnapshot:
        raise NotImplementedError


class AlpacaTradingClient(BrokerageClient):
    provider = "alpaca"

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: Optional[str] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        if not api_key or not api_secret:
            raise BrokerageClientError("ALPACA_API_KEY and ALPACA_API_SECRET are required.")

        self.api_key = api_key.strip()
        self.api_secret = api_secret.strip()
        self.base_url = (base_url or os.getenv("ALPACA_API_BASE_URL") or "https://paper-api.alpaca.markets").rstrip("/")
        self.session = session or requests.Session()
        self.timeout_seconds = float(os.getenv("BROKERAGE_REQUEST_TIMEOUT_SECONDS", "10"))

    def _headers(self) -> Dict[str, str]:
        return {
            "accept": "application/json",
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
        }

    def _get(self, path: str) -> Any:
        url = f"{self.base_url}{path}"
        try:
            response = self.session.get(url, headers=self._headers(), timeout=self.timeout_seconds)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            raise BrokerageClientError(f"Alpaca request failed for {path}: {exc}") from exc

    @staticmethod
    def _parse_position(position: Dict[str, Any]) -> Optional[BrokerageHolding]:
        symbol = str(position.get("symbol", "")).upper().strip()
        quantity = _safe_float(position.get("qty"))
        if not symbol or quantity is None or quantity <= 0:
            return None

        avg_entry_price = _safe_float(position.get("avg_entry_price"))
        total_cost_basis = _safe_float(position.get("cost_basis"))
        per_share_cost_basis = avg_entry_price
        if per_share_cost_basis is None and total_cost_basis is not None and quantity:
            per_share_cost_basis = total_cost_basis / quantity

        return BrokerageHolding(
            symbol=symbol,
            quantity=quantity,
            cost_basis=per_share_cost_basis or 0.0,
            current_price=_safe_float(position.get("current_price")),
            market_value=_safe_float(position.get("market_value")),
            unrealized_pl=_safe_float(position.get("unrealized_pl")),
            raw=position,
        )

    def fetch_account_snapshot(self) -> BrokerageAccountSnapshot:
        account = self._get("/v2/account")
        positions = self._get("/v2/positions")

        holdings: List[BrokerageHolding] = []
        for position in positions:
            holding = self._parse_position(position)
            if holding is not None:
                holdings.append(holding)

        return BrokerageAccountSnapshot(
            provider=self.provider,
            holdings=holdings,
            fetched_at=_utc_now_iso(),
            account_id=str(account.get("id", "")),
            account_number=str(account.get("account_number", "")),
            equity=_safe_float(account.get("equity")),
            buying_power=_safe_float(account.get("buying_power")),
            currency=str(account.get("currency", "USD") or "USD"),
            raw_account=account,
        )


def get_configured_brokerage_provider() -> str:
    return str(os.getenv("BROKERAGE_PROVIDER", "")).strip().lower()


def brokerage_is_configured() -> bool:
    return bool(get_configured_brokerage_provider())


def build_brokerage_client_from_env() -> Optional[BrokerageClient]:
    provider = get_configured_brokerage_provider()
    if not provider:
        return None

    if provider == "alpaca":
        api_key = os.getenv("ALPACA_API_KEY") or os.getenv("ALPACA_KEY_ID")
        api_secret = os.getenv("ALPACA_API_SECRET") or os.getenv("ALPACA_SECRET_KEY")
        return AlpacaTradingClient(
            api_key=api_key or "",
            api_secret=api_secret or "",
            base_url=os.getenv("ALPACA_API_BASE_URL") or None,
        )

    raise BrokerageClientError(
        f"Unsupported brokerage provider `{provider}`. Supported providers: alpaca."
    )
