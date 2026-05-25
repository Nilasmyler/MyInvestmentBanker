import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from utils.discovery_support import OwnershipIntel
from utils.financial_tools import fetch_institutional_holder_snapshot, fetch_recent_sec_ownership_filings

load_dotenv()
logger = logging.getLogger("MyInvestmentBanker.agents.ownership_intel")
logging.basicConfig(level=logging.INFO)


def _parse_iso_date(raw_value: str) -> Optional[datetime]:
    if not raw_value:
        return None
    try:
        parsed = datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        try:
            return datetime.strptime(raw_value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except Exception:
            return None


class OwnershipIntelAgent:
    """
    Collects ownership and insider-intelligence signals for a symbol.
    """

    @staticmethod
    def collect_symbol_intel(symbol: str) -> OwnershipIntel:
        symbol = symbol.upper().strip()
        filings = fetch_recent_sec_ownership_filings(symbol, limit=None)
        holder_snapshot = fetch_institutional_holder_snapshot(symbol, limit=5)

        cutoff = datetime.now(timezone.utc) - timedelta(days=90)
        recent_insider_forms = 0
        recent_beneficial_forms = 0
        recent_form4_count = 0
        beneficial_form_types = []
        recent_initial_beneficial_forms = 0
        for filing in filings:
            filing_date = _parse_iso_date(str(filing.get("date", "")))
            if filing_date and filing_date < cutoff:
                continue
            category = filing.get("category")
            form_type = str(filing.get("form", "")).upper()
            if category == "insider":
                recent_insider_forms += 1
                if form_type == "4":
                    recent_form4_count += 1
            elif category == "beneficial_ownership":
                recent_beneficial_forms += 1
                if form_type:
                    beneficial_form_types.append(form_type)
                if not form_type.endswith("/A"):
                    recent_initial_beneficial_forms += 1

        institutional_ownership_pct = holder_snapshot.get("institutional_ownership_pct")
        insider_ownership_pct = holder_snapshot.get("insider_ownership_pct")
        top_holders = holder_snapshot.get("top_holders", [])
        holder_snapshot_error = holder_snapshot.get("error")

        alerts: List[str] = []
        if recent_form4_count >= 2:
            alerts.append(
                f"`{recent_form4_count}` recent Form 4 filings were disclosed, but the header-only feed does not yet classify buys versus sales."
            )
        if recent_beneficial_forms:
            alerts.append(
                f"`{recent_beneficial_forms}` recent beneficial-ownership filing(s) detected"
                f"{' (' + ', '.join(sorted(set(beneficial_form_types))[:2]) + ')' if beneficial_form_types else ''};"
                " details still need deeper parsing."
            )
        if institutional_ownership_pct is not None and institutional_ownership_pct >= 70:
            alerts.append(f"Holder snapshot shows roughly `{institutional_ownership_pct}%` institutional ownership.")
        if insider_ownership_pct is not None and insider_ownership_pct >= 10:
            alerts.append(f"Holder snapshot shows roughly `{insider_ownership_pct}%` insider ownership.")
        if top_holders:
            alerts.append(
                f"Top institutions in the snapshot: `{', '.join([item.get('holder', '') for item in top_holders[:3] if item.get('holder')])}`."
            )

        signal_strength = "low"

        if recent_initial_beneficial_forms:
            summary = (
                "Fresh beneficial-ownership filings were detected, but the current header-only parser keeps this informational "
                "until the filing details are parsed."
            )
            if alerts:
                summary = f"{summary} {' '.join(alerts[:2])}"
        elif top_holders or institutional_ownership_pct is not None or insider_ownership_pct is not None:
            parts = ["Ownership snapshot shows the following details:"]
            if institutional_ownership_pct is not None:
                parts.append(f"Institutional ownership is roughly `{institutional_ownership_pct}%` of shares.")
            if insider_ownership_pct is not None:
                parts.append(f"Insider ownership is roughly `{insider_ownership_pct}%` of shares.")
            if top_holders:
                top_holders_desc = []
                for holder in top_holders[:3]:
                    name = holder.get("holder")
                    shares = holder.get("shares")
                    pct = holder.get("pct_out")
                    if name:
                        details = []
                        if shares is not None:
                            if shares >= 1_000_000:
                                details.append(f"{shares / 1_000_000:.2f}M shares")
                            elif shares >= 1_000:
                                details.append(f"{shares / 1_000:.1f}K shares")
                            else:
                                details.append(f"{shares} shares")
                        if pct is not None:
                            details.append(f"{pct}%")
                        if details:
                            top_holders_desc.append(f"{name} ({', '.join(details)})")
                        else:
                            top_holders_desc.append(name)
                if top_holders_desc:
                    parts.append(f"Top holders: {'; '.join(top_holders_desc)}.")
            if len(parts) > 1:
                summary = " ".join(parts)
            else:
                summary = "Ownership snapshot is available, but no fresh SEC ownership catalyst was captured."
        else:
            summary = "Ownership intelligence was thin in this pass."

        result: OwnershipIntel = {
            "symbol": symbol,
            "summary": summary,
            "signal_strength": signal_strength,
            "recent_form4_count": recent_form4_count,
            "recent_beneficial_ownership_count": recent_beneficial_forms,
            "institutional_ownership_pct": institutional_ownership_pct,
            "insider_ownership_pct": insider_ownership_pct,
            "top_holders": top_holders,
            "recent_filings": filings,
            "alerts": alerts,
        }
        if holder_snapshot_error:
            result["error"] = str(holder_snapshot_error)
        return result
