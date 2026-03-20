from __future__ import annotations

"""
Roostoo REST API client (type-safe).

This module provides a small, strict-parsing client for the Roostoo mock REST API:
https://mock-api.roostoo.com (see roostoo_api/DOCS.md for the official schema).

Key points
- Transport: synchronous `httpx.Client` (use `asyncio.to_thread(...)` if calling from async code).
- Parsing: strict; raises `RoostooParseError` if the response JSON shape/types don’t match docs.
- HTTP errors: raises `RoostooHTTPError` on non-2xx/3xx responses (includes status_code/body).

Auth levels (per DOCS.md)
- RCL_NoVerification: no auth required
- RCL_TSCheck: requires a `timestamp` query parameter (13-digit ms string)
- RCL_TopLevelCheck (SIGNED): requires headers and a signed payload:
  - Header `RST-API-KEY`: your API key
  - Header `MSG-SIGNATURE`: HMAC-SHA256(secretKey, totalParams).hexdigest()
  - `totalParams`: all params + `timestamp`, sorted by key, joined as `k=v&k2=v2`
  - POST endpoints must use `Content-Type: application/x-www-form-urlencoded` and send
    `totalParams` as the raw body (not JSON).

Environment variables
- ROOSTOO_API_KEY: API key for signed endpoints
- ROOSTOO_SECRET: secret key for signed endpoints
- ROOSTOO_BASE_URL: optional override (defaults to https://mock-api.roostoo.com)

Quick start
```python
from roostoo_api import RoostooClient, Side, OrderType

with RoostooClient.from_env() as api:
  server_time = api.server_time()
  exchange = api.exchange_info()
  ticker = api.ticker(pair="BTC/USD")

  balance = api.balance()  # requires ROOSTOO_API_KEY + ROOSTOO_SECRET
  order = api.place_order(
    pair="BTC/USD",
    side=Side.BUY,
    order_type=OrderType.MARKET,
    quantity=1,
  )
```
"""

import hashlib
import hmac
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

import httpx


class RoostooError(Exception):
  pass


class RoostooHTTPError(RoostooError):
  def __init__(self, message: str, *, status_code: int | None = None, body: str | None = None):
    super().__init__(message)
    self.status_code = status_code
    self.body = body


class RoostooParseError(RoostooError):
  pass


class Side(str, Enum):
  """Order side."""

  BUY = "BUY"
  SELL = "SELL"


class OrderType(str, Enum):
  """Order type."""

  LIMIT = "LIMIT"
  MARKET = "MARKET"


@dataclass(frozen=True, slots=True)
class ServerTime:
  server_time: int


@dataclass(frozen=True, slots=True)
class TradePairInfo:
  coin: str
  coin_full_name: str
  unit: str
  unit_full_name: str
  can_trade: bool
  price_precision: int
  amount_precision: int
  mini_order: float


@dataclass(frozen=True, slots=True)
class ExchangeInfo:
  is_running: bool
  initial_wallet: dict[str, float]
  trade_pairs: dict[str, TradePairInfo]


@dataclass(frozen=True, slots=True)
class TickerEntry:
  max_bid: float
  min_ask: float
  last_price: float
  change: float
  coin_trade_value: float
  unit_trade_value: float


@dataclass(frozen=True, slots=True)
class TickerResponse:
  success: bool
  err_msg: str
  server_time: int
  data: dict[str, TickerEntry]


@dataclass(frozen=True, slots=True)
class BalanceEntry:
  free: float
  lock: float


@dataclass(frozen=True, slots=True)
class BalanceResponse:
  success: bool
  err_msg: str
  wallet: dict[str, BalanceEntry]


@dataclass(frozen=True, slots=True)
class PendingCountResponse:
  success: bool
  err_msg: str
  total_pending: int
  order_pairs: dict[str, int]


@dataclass(frozen=True, slots=True)
class OrderDetail:
  pair: str
  order_id: int
  status: str
  role: str
  server_time_usage: float
  create_timestamp: int
  finish_timestamp: int
  side: str
  type: str
  stop_type: str
  price: float
  quantity: float
  filled_quantity: float
  filled_aver_price: float
  coin_change: float
  unit_change: float
  commission_coin: str
  commission_charge_value: float
  commission_percent: float


@dataclass(frozen=True, slots=True)
class PlaceOrderResponse:
  success: bool
  err_msg: str
  order_detail: OrderDetail | None


@dataclass(frozen=True, slots=True)
class QueryOrderResponse:
  success: bool
  err_msg: str
  order_matched: list[OrderDetail] | None


@dataclass(frozen=True, slots=True)
class CancelOrderResponse:
  success: bool
  err_msg: str
  canceled_list: list[int]


def _timestamp_ms() -> str:
  return str(int(time.time() * 1000))


def _as_dict(value: Any, *, name: str) -> dict[str, Any]:
  if not isinstance(value, dict):
    raise RoostooParseError(f"{name} must be an object")
  out: dict[str, Any] = {}
  for k, v in value.items():
    if not isinstance(k, str):
      raise RoostooParseError(f"{name} keys must be strings")
    out[k] = v
  return out


def _as_str(value: Any, *, name: str) -> str:
  if not isinstance(value, str):
    raise RoostooParseError(f"{name} must be a string")
  return value


def _as_bool(value: Any, *, name: str) -> bool:
  if not isinstance(value, bool):
    raise RoostooParseError(f"{name} must be a boolean")
  return value


def _as_int(value: Any, *, name: str) -> int:
  if isinstance(value, bool) or not isinstance(value, int):
    raise RoostooParseError(f"{name} must be an integer")
  return value


def _as_float(value: Any, *, name: str) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise RoostooParseError(f"{name} must be a number")
  return float(value)


def _as_list(value: Any, *, name: str) -> list[Any]:
  if not isinstance(value, list):
    raise RoostooParseError(f"{name} must be a list")
  return value


def _parse_server_time(payload: Any) -> ServerTime:
  obj = _as_dict(payload, name="serverTime response")
  return ServerTime(server_time=_as_int(obj.get("ServerTime"), name="ServerTime"))


def _parse_exchange_info(payload: Any) -> ExchangeInfo:
  obj = _as_dict(payload, name="exchangeInfo response")
  initial_wallet_raw = _as_dict(obj.get("InitialWallet"), name="InitialWallet")
  initial_wallet: dict[str, float] = {
    k: _as_float(v, name=f"InitialWallet.{k}") for k, v in initial_wallet_raw.items()
  }

  trade_pairs_raw = _as_dict(obj.get("TradePairs"), name="TradePairs")
  trade_pairs: dict[str, TradePairInfo] = {}
  for pair, info_any in trade_pairs_raw.items():
    info = _as_dict(info_any, name=f"TradePairs.{pair}")
    trade_pairs[pair] = TradePairInfo(
      coin=_as_str(info.get("Coin"), name=f"TradePairs.{pair}.Coin"),
      coin_full_name=_as_str(info.get("CoinFullName"), name=f"TradePairs.{pair}.CoinFullName"),
      unit=_as_str(info.get("Unit"), name=f"TradePairs.{pair}.Unit"),
      unit_full_name=_as_str(info.get("UnitFullName"), name=f"TradePairs.{pair}.UnitFullName"),
      can_trade=_as_bool(info.get("CanTrade"), name=f"TradePairs.{pair}.CanTrade"),
      price_precision=_as_int(info.get("PricePrecision"), name=f"TradePairs.{pair}.PricePrecision"),
      amount_precision=_as_int(info.get("AmountPrecision"), name=f"TradePairs.{pair}.AmountPrecision"),
      mini_order=_as_float(info.get("MiniOrder"), name=f"TradePairs.{pair}.MiniOrder"),
    )

  return ExchangeInfo(
    is_running=_as_bool(obj.get("IsRunning"), name="IsRunning"),
    initial_wallet=initial_wallet,
    trade_pairs=trade_pairs,
  )


def _parse_ticker_response(payload: Any) -> TickerResponse:
  obj = _as_dict(payload, name="ticker response")
  data_raw = _as_dict(obj.get("Data"), name="Data")
  data: dict[str, TickerEntry] = {}
  for pair, entry_any in data_raw.items():
    entry = _as_dict(entry_any, name=f"Data.{pair}")
    data[pair] = TickerEntry(
      max_bid=_as_float(entry.get("MaxBid"), name=f"Data.{pair}.MaxBid"),
      min_ask=_as_float(entry.get("MinAsk"), name=f"Data.{pair}.MinAsk"),
      last_price=_as_float(entry.get("LastPrice"), name=f"Data.{pair}.LastPrice"),
      change=_as_float(entry.get("Change"), name=f"Data.{pair}.Change"),
      coin_trade_value=_as_float(entry.get("CoinTradeValue"), name=f"Data.{pair}.CoinTradeValue"),
      unit_trade_value=_as_float(entry.get("UnitTradeValue"), name=f"Data.{pair}.UnitTradeValue"),
    )
  return TickerResponse(
    success=_as_bool(obj.get("Success"), name="Success"),
    err_msg=_as_str(obj.get("ErrMsg"), name="ErrMsg"),
    server_time=_as_int(obj.get("ServerTime"), name="ServerTime"),
    data=data,
  )


def _parse_balance_response(payload: Any) -> BalanceResponse:
  obj = _as_dict(payload, name="balance response")
  success = _as_bool(obj.get("Success"), name="Success")
  wallet: dict[str, BalanceEntry] = {}

  def _merge_wallet(raw: Any, *, label: str) -> None:
    if not isinstance(raw, dict):
      return
    wallet_raw = _as_dict(raw, name=label)
    for asset, entry_any in wallet_raw.items():
      entry = _as_dict(entry_any, name=f"{label}.{asset}")
      free = _as_float(entry.get("Free"), name=f"{label}.{asset}.Free")
      lock = _as_float(entry.get("Lock"), name=f"{label}.{asset}.Lock")
      prev = wallet.get(asset)
      if prev is None:
        wallet[asset] = BalanceEntry(free=free, lock=lock)
      else:
        wallet[asset] = BalanceEntry(free=prev.free + free, lock=prev.lock + lock)

  _merge_wallet(obj.get("Wallet"), label="Wallet")
  _merge_wallet(obj.get("SpotWallet"), label="SpotWallet")
  _merge_wallet(obj.get("MarginWallet"), label="MarginWallet")

  return BalanceResponse(
    success=success,
    err_msg=_as_str(obj.get("ErrMsg"), name="ErrMsg"),
    wallet=wallet,
  )


def _parse_pending_count_response(payload: Any) -> PendingCountResponse:
  obj = _as_dict(payload, name="pending_count response")
  order_pairs_raw = _as_dict(obj.get("OrderPairs"), name="OrderPairs")
  order_pairs: dict[str, int] = {
    k: _as_int(v, name=f"OrderPairs.{k}") for k, v in order_pairs_raw.items()
  }
  return PendingCountResponse(
    success=_as_bool(obj.get("Success"), name="Success"),
    err_msg=_as_str(obj.get("ErrMsg"), name="ErrMsg"),
    total_pending=_as_int(obj.get("TotalPending"), name="TotalPending"),
    order_pairs=order_pairs,
  )


def _parse_order_detail(obj_any: Any, *, name: str) -> OrderDetail:
  obj = _as_dict(obj_any, name=name)
  return OrderDetail(
    pair=_as_str(obj.get("Pair"), name=f"{name}.Pair"),
    order_id=_as_int(obj.get("OrderID"), name=f"{name}.OrderID"),
    status=_as_str(obj.get("Status"), name=f"{name}.Status"),
    role=_as_str(obj.get("Role"), name=f"{name}.Role"),
    server_time_usage=_as_float(obj.get("ServerTimeUsage"), name=f"{name}.ServerTimeUsage"),
    create_timestamp=_as_int(obj.get("CreateTimestamp"), name=f"{name}.CreateTimestamp"),
    finish_timestamp=_as_int(obj.get("FinishTimestamp"), name=f"{name}.FinishTimestamp"),
    side=_as_str(obj.get("Side"), name=f"{name}.Side"),
    type=_as_str(obj.get("Type"), name=f"{name}.Type"),
    stop_type=_as_str(obj.get("StopType"), name=f"{name}.StopType"),
    price=_as_float(obj.get("Price"), name=f"{name}.Price"),
    quantity=_as_float(obj.get("Quantity"), name=f"{name}.Quantity"),
    filled_quantity=_as_float(obj.get("FilledQuantity"), name=f"{name}.FilledQuantity"),
    filled_aver_price=_as_float(obj.get("FilledAverPrice"), name=f"{name}.FilledAverPrice"),
    coin_change=_as_float(obj.get("CoinChange"), name=f"{name}.CoinChange"),
    unit_change=_as_float(obj.get("UnitChange"), name=f"{name}.UnitChange"),
    commission_coin=_as_str(obj.get("CommissionCoin"), name=f"{name}.CommissionCoin"),
    commission_charge_value=_as_float(
      obj.get("CommissionChargeValue"), name=f"{name}.CommissionChargeValue"
    ),
    commission_percent=_as_float(obj.get("CommissionPercent"), name=f"{name}.CommissionPercent"),
  )


def _parse_place_order_response(payload: Any) -> PlaceOrderResponse:
  obj = _as_dict(payload, name="place_order response")
  detail_any = obj.get("OrderDetail")
  detail = _parse_order_detail(detail_any, name="OrderDetail") if detail_any is not None else None
  return PlaceOrderResponse(
    success=_as_bool(obj.get("Success"), name="Success"),
    err_msg=_as_str(obj.get("ErrMsg"), name="ErrMsg"),
    order_detail=detail,
  )


def _parse_query_order_response(payload: Any) -> QueryOrderResponse:
  obj = _as_dict(payload, name="query_order response")
  matched_any = obj.get("OrderMatched")
  matched: list[OrderDetail] | None = None
  if matched_any is not None:
    matched_list = _as_list(matched_any, name="OrderMatched")
    matched = [_parse_order_detail(o, name="OrderMatched[]") for o in matched_list]
  return QueryOrderResponse(
    success=_as_bool(obj.get("Success"), name="Success"),
    err_msg=_as_str(obj.get("ErrMsg"), name="ErrMsg"),
    order_matched=matched,
  )


def _parse_cancel_order_response(payload: Any) -> CancelOrderResponse:
  obj = _as_dict(payload, name="cancel_order response")
  canceled_any = obj.get("CanceledList") or []
  canceled_list = [_as_int(v, name="CanceledList[]") for v in _as_list(canceled_any, name="CanceledList")]
  return CancelOrderResponse(
    success=_as_bool(obj.get("Success"), name="Success"),
    err_msg=_as_str(obj.get("ErrMsg"), name="ErrMsg"),
    canceled_list=canceled_list,
  )


class RoostooClient:
  """
  Synchronous Roostoo REST client.

  Use `from_env()` to construct a client that reads credentials/base_url from environment.
  Signed endpoints require `api_key` + `secret_key` (or ROOSTOO_API_KEY/ROOSTOO_SECRET).

  Public methods map 1:1 to DOCS.md endpoints:
  - server_time() -> GET /v3/serverTime (no auth)
  - exchange_info() -> GET /v3/exchangeInfo (no auth)
  - ticker(pair=...) -> GET /v3/ticker (timestamp required)
  - balance() -> GET /v3/balance (SIGNED)
  - pending_count() -> GET /v3/pending_count (SIGNED)
  - place_order(...) -> POST /v3/place_order (SIGNED)
  - query_order(...) -> POST /v3/query_order (SIGNED)
  - cancel_order(...) -> POST /v3/cancel_order (SIGNED)
  """

  @classmethod
  def from_env(
    cls,
    *,
    base_url: str | None = None,
    timeout_s: float = 10.0,
  ) -> "RoostooClient":
    """
    Create a client using environment variables.

    - ROOSTOO_API_KEY / ROOSTOO_SECRET are used for signed endpoints
    - ROOSTOO_BASE_URL optionally overrides the default base URL
    """

    return cls(
      api_key=os.getenv("ROOSTOO_API_KEY"),
      secret_key=os.getenv("ROOSTOO_SECRET"),
      base_url=base_url or os.getenv("ROOSTOO_BASE_URL", "https://mock-api.roostoo.com"),
      timeout_s=timeout_s,
    )

  def __init__(
    self,
    *,
    api_key: str | None = None,
    secret_key: str | None = None,
    base_url: str = "https://mock-api.roostoo.com",
    timeout_s: float = 10.0,
  ):
    self._api_key = api_key
    self._secret_key = secret_key
    self._client = httpx.Client(base_url=base_url, timeout=timeout_s)

  def close(self) -> None:
    self._client.close()

  def __enter__(self) -> "RoostooClient":
    return self

  def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
    self.close()

  def server_time(self) -> ServerTime:
    """GET /v3/serverTime (RCL_NoVerification)."""

    payload = self._get_json("/v3/serverTime")
    return _parse_server_time(payload)

  def exchange_info(self) -> ExchangeInfo:
    """GET /v3/exchangeInfo (RCL_NoVerification)."""

    payload = self._get_json("/v3/exchangeInfo")
    return _parse_exchange_info(payload)

  def ticker(self, *, pair: str | None = None) -> TickerResponse:
    """
    GET /v3/ticker (RCL_TSCheck).

    - pair: optional symbol like "BTC/USD". When omitted, returns all tickers.
    """

    params: dict[str, str] = {"timestamp": _timestamp_ms()}
    if pair:
      params["pair"] = pair
    payload = self._get_json("/v3/ticker", params=params)
    return _parse_ticker_response(payload)

  def balance(self) -> BalanceResponse:
    """GET /v3/balance (RCL_TopLevelCheck / SIGNED)."""

    headers, params, _ = self._signed_payload({})
    payload = self._get_json("/v3/balance", headers=headers, params=params)
    return _parse_balance_response(payload)

  def pending_count(self) -> PendingCountResponse:
    """GET /v3/pending_count (RCL_TopLevelCheck / SIGNED)."""

    headers, params, _ = self._signed_payload({})
    payload = self._get_json("/v3/pending_count", headers=headers, params=params)
    return _parse_pending_count_response(payload)

  def place_order(
    self,
    *,
    pair: str,
    side: Side,
    order_type: OrderType,
    quantity: str | float,
    price: str | float | None = None,
  ) -> PlaceOrderResponse:
    """
    POST /v3/place_order (RCL_TopLevelCheck / SIGNED).

    - pair: symbol like "BTC/USD"
    - side: BUY or SELL
    - order_type: MARKET or LIMIT (LIMIT requires price)
    - quantity/price are serialized to strings and sent as form-encoded parameters
    """

    if order_type == OrderType.LIMIT and price is None:
      raise ValueError("LIMIT orders require price")

    payload: dict[str, str] = {
      "pair": pair,
      "side": side.value,
      "type": order_type.value,
      "quantity": str(quantity),
    }
    if price is not None:
      payload["price"] = str(price)

    headers, _, total_params = self._signed_payload(payload)
    headers = {**headers, "Content-Type": "application/x-www-form-urlencoded"}
    resp_payload = self._post_form("/v3/place_order", headers=headers, body=total_params)
    return _parse_place_order_response(resp_payload)

  def query_order(
    self,
    *,
    order_id: str | int | None = None,
    pair: str | None = None,
    offset: int | None = None,
    limit: int | None = None,
    pending_only: bool | None = None,
  ) -> QueryOrderResponse:
    """
    POST /v3/query_order (RCL_TopLevelCheck / SIGNED).

    Per DOCS.md:
    - If order_id is provided, no other optional parameter is allowed.
    - Otherwise, you can filter by pair, offset/limit, and pending_only.
    """

    payload: dict[str, str] = {}
    if order_id is not None:
      payload["order_id"] = str(order_id)
    else:
      if pair is not None:
        payload["pair"] = pair
      if offset is not None:
        payload["offset"] = str(offset)
      if limit is not None:
        payload["limit"] = str(limit)
      if pending_only is not None:
        payload["pending_only"] = "TRUE" if pending_only else "FALSE"

    headers, _, total_params = self._signed_payload(payload)
    headers = {**headers, "Content-Type": "application/x-www-form-urlencoded"}
    resp_payload = self._post_form("/v3/query_order", headers=headers, body=total_params)
    return _parse_query_order_response(resp_payload)

  def cancel_order(
    self,
    *,
    order_id: str | int | None = None,
    pair: str | None = None,
  ) -> CancelOrderResponse:
    """
    POST /v3/cancel_order (RCL_TopLevelCheck / SIGNED).

    Cancels pending orders:
    - Provide order_id OR pair (not both), or neither to cancel all pending.
    """

    payload: dict[str, str] = {}
    if order_id is not None and pair is not None:
      raise ValueError("Only one of order_id or pair is allowed")
    if order_id is not None:
      payload["order_id"] = str(order_id)
    if pair is not None:
      payload["pair"] = pair

    headers, _, total_params = self._signed_payload(payload)
    headers = {**headers, "Content-Type": "application/x-www-form-urlencoded"}
    resp_payload = self._post_form("/v3/cancel_order", headers=headers, body=total_params)
    return _parse_cancel_order_response(resp_payload)

  def _signed_payload(self, payload: Mapping[str, str]) -> tuple[dict[str, str], dict[str, str], str]:
    if not self._api_key or not self._secret_key:
      raise ValueError("api_key and secret_key are required for signed endpoints")

    params: dict[str, str] = dict(payload)
    params["timestamp"] = _timestamp_ms()
    total_params = "&".join(f"{k}={params[k]}" for k in sorted(params.keys()))
    signature = hmac.new(
      self._secret_key.encode("utf-8"), total_params.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    headers = {"RST-API-KEY": self._api_key, "MSG-SIGNATURE": signature}
    return headers, params, total_params

  def _get_json(
    self,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, str] | None = None,
  ) -> Any:
    try:
      resp = self._client.get(path, headers=headers, params=params)
    except httpx.HTTPError as e:
      raise RoostooHTTPError(str(e)) from e

    if resp.status_code >= 400:
      raise RoostooHTTPError(
        f"HTTP {resp.status_code} for GET {path}", status_code=resp.status_code, body=resp.text
      )
    try:
      return resp.json()
    except ValueError as e:
      raise RoostooParseError(f"Invalid JSON for GET {path}") from e

  def _post_form(self, path: str, *, headers: dict[str, str], body: str) -> Any:
    try:
      resp = self._client.post(path, headers=headers, content=body)
    except httpx.HTTPError as e:
      raise RoostooHTTPError(str(e)) from e

    if resp.status_code >= 400:
      raise RoostooHTTPError(
        f"HTTP {resp.status_code} for POST {path}", status_code=resp.status_code, body=resp.text
      )
    try:
      return resp.json()
    except ValueError as e:
      raise RoostooParseError(f"Invalid JSON for POST {path}") from e