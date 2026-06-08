#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


INSIDER_PLAUSIBLE_CATEGORIES = {
    "ELECTION",
    "EARNINGS",
    "POLICY",
    "GEOPOLITICAL",
    "APPOINTMENT",
    "LEGAL",
    "PRODUCT_LAUNCH",
    "OTHER_ANNOUNCEMENT",
}

NON_INSIDER_CATEGORIES = {
    "CRYPTO_PRICE",
    "SPORTS",
    "WEATHER",
    "ENTERTAINMENT",
    "SCIENCE",
    "GRADUAL",
    "OTHER",
}

ALL_CATEGORIES = INSIDER_PLAUSIBLE_CATEGORIES | NON_INSIDER_CATEGORIES

DEFAULT_MODEL = "gpt-5.4-nano"
DEFAULT_BATCH_SIZE = 20
DEFAULT_MAX_CONCURRENCY = 1
CHECKPOINT_INTERVAL = 5
SCHEMA_VERSION = "1.0"

SYSTEM_PROMPT = """You are a financial markets classifier. Categorize prediction markets based on whether insider trading is structurally plausible.

A market is "insider-plausible" if its resolution depends on a discrete, announceable event where some party could know the outcome before public announcement.

Categories (choose exactly one):
- ELECTION: Elections, referendums, voting outcomes, political appointments
- EARNINGS: Company earnings, quarterly reports, financial disclosures
- POLICY: Central bank decisions (Fed/ECB), tariffs, regulations, legislation
- GEOPOLITICAL: Wars, military actions, invasions, diplomatic events, treaties
- APPOINTMENT: CEO/executive appointments, papal conclaves, judicial nominations
- LEGAL: Court rulings, verdicts, indictments, regulatory enforcement actions
- PRODUCT_LAUNCH: Product announcements, release dates, feature launches
- OTHER_ANNOUNCEMENT: Other discrete privately-knowable events not above
- CRYPTO_PRICE: Cryptocurrency price targets (public feed, NOT insider-plausible)
- SPORTS: Game outcomes, championships, player statistics
- WEATHER: Weather events, climate predictions
- ENTERTAINMENT: Award shows, TV ratings, box office results
- SCIENCE: Scientific discoveries, space events, research publications
- GRADUAL: Slow-burn outcomes without clear announcement moment
- OTHER: Cannot categorize or doesn't fit above

Respond with JSON only."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json_atomic(payload: Dict[str, Any], path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
    tmp_path.replace(output_path)


def _build_failure_entry(market: Dict[str, Any], error: str) -> Dict[str, Any]:
    return {
        "market_slug": str(market.get("market_slug", "")),
        "category": None,
        "insider_plausible": False,
        "confidence": 0.0,
        "reasoning": "",
        "status": "failed",
        "error": str(error),
    }


def _build_success_entry(market: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "market_slug": str(market.get("market_slug", "")),
        "category": result["category"],
        "confidence": result["confidence"],
        "reasoning": result["reasoning"],
        "insider_plausible": derive_insider_plausible(result["category"]),
    }


def _is_failed_entry(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return True
    if entry.get("status") == "failed":
        return True
    category = entry.get("category")
    if category is None:
        return True
    return str(category).strip().upper() not in ALL_CATEGORIES


def _normalize_market_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def _get_prompt_text(row: pd.Series) -> str:
    for col in ("question", "title", "description"):
        if col in row.index:
            text = _normalize_market_text(row[col])
            if text:
                return text
    return ""


def _coerce_market_id(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _parse_iso_date(value: Optional[str]) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid date: {value}")
    return ts


def _extract_content_text(response: Any) -> str:
    if not getattr(response, "choices", None):
        raise ValueError("OpenAI response had no choices")

    message = response.choices[0].message
    content = getattr(message, "content", None)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
            elif hasattr(item, "text"):
                text_parts.append(str(getattr(item, "text", "")))
        joined = "".join(text_parts).strip()
        if joined:
            return joined

    raise ValueError("Unable to extract text content from OpenAI response")


def _extract_results(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in ("classifications", "results", "items", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return value

    raise ValueError("Response JSON did not contain a classifications array")


def _normalize_result(result: Any) -> Dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError(f"Classification result must be an object, got {type(result).__name__}")

    raw_category = str(result.get("category", "OTHER")).strip().upper()
    category = raw_category if raw_category in ALL_CATEGORIES else "OTHER"
    market_id = _coerce_market_id(result.get("id", result.get("market_id")))

    try:
        confidence = float(result.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    reasoning_present = "reasoning" in result
    reasoning = str(result.get("reasoning", "") or "").strip()
    if not reasoning and not reasoning_present:
        reasoning = "No reasoning provided by model."

    return {
        "id": market_id,
        "category": category,
        "confidence": confidence,
        "reasoning": reasoning,
    }


def _align_results_to_markets(
    raw_results: List[Dict[str, Any]],
    markets: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    normalized_results = [_normalize_result(result) for result in raw_results]
    expected_ids = [int(market["id"]) for market in markets]
    expected_id_set = set(expected_ids)

    matched_by_id: Dict[int, Dict[str, Any]] = {}
    id_items_seen = 0
    ignored_items = 0

    for result in normalized_results:
        result_id = result.get("id")
        if result_id is None:
            continue
        id_items_seen += 1
        if result_id not in expected_id_set or result_id in matched_by_id:
            ignored_items += 1
            continue
        matched_by_id[result_id] = result

    if len(matched_by_id) == len(expected_ids):
        if ignored_items:
            logging.debug(
                "Aligned batch by market ID and ignored %s extra/duplicate item(s)",
                ignored_items,
            )
        return [matched_by_id[market_id] for market_id in expected_ids]

    if id_items_seen == 0 and len(normalized_results) == len(markets):
        return normalized_results

    missing_ids = [market_id for market_id in expected_ids if market_id not in matched_by_id]
    raise ValueError(
        "Model response did not align to expected market IDs. "
        f"expected={len(markets)} raw={len(raw_results)} matched={len(matched_by_id)} "
        f"missing_ids={missing_ids[:10]}"
    )


def _batched(items: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    return [items[idx : idx + batch_size] for idx in range(0, len(items), batch_size)]


def load_markets(data_dir: str = "data") -> pd.DataFrame:
    """Load markets.csv and return DataFrame with columns: id, market_slug, answer1, answer2."""
    markets_path = Path(data_dir) / "markets.csv"
    if not markets_path.exists():
        raise FileNotFoundError(f"markets.csv not found at {markets_path}")

    df = pd.read_csv(markets_path)
    required_columns = {"id", "market_slug", "answer1", "answer2"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"markets.csv missing required columns: {', '.join(missing_columns)}")

    keep_columns = ["id", "market_slug", "answer1", "answer2"]
    for optional_col in ("question", "title", "description", "createdAt", "closedTime"):
        if optional_col in df.columns:
            keep_columns.append(optional_col)

    markets_df = df[keep_columns].copy()
    markets_df = markets_df.dropna(subset=["id"]).copy()
    markets_df["id"] = markets_df["id"].astype(int)

    for col in keep_columns:
        if col == "id":
            continue
        markets_df[col] = markets_df[col].map(_normalize_market_text)

    markets_df["prompt_text"] = markets_df.apply(_get_prompt_text, axis=1)
    markets_df = markets_df.sort_values("id").reset_index(drop=True)
    return markets_df


def load_existing_classifications(path: str) -> Dict[str, Any]:
    """Load existing classifications from JSON, return empty dict if not exists."""
    path_obj = Path(path)
    if not path_obj.exists():
        return {}

    try:
        with open(path_obj, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        logging.warning("Failed to load existing classifications from %s: %s", path, exc)
        return {}

    if isinstance(payload, dict) and isinstance(payload.get("classifications"), dict):
        return payload["classifications"]
    if isinstance(payload, dict):
        return payload

    logging.warning("Unexpected classification payload in %s; ignoring existing file", path)
    return {}


def save_classifications(classifications: Dict[str, Any], metadata: Dict[str, Any], path: str) -> None:
    """Save classifications to JSON with metadata."""
    _write_json_atomic(
        {
            "metadata": metadata,
            "classifications": classifications,
        },
        path,
    )


def save_checkpoint(classifications: Dict[str, Any], checkpoint_path: str) -> None:
    """Save progress checkpoint."""
    checkpoint_metadata = {
        "created_at": _utc_now_iso(),
        "schema_version": SCHEMA_VERSION,
        "classified_entries": len(classifications),
        "checkpoint": True,
    }
    save_classifications(classifications, checkpoint_metadata, checkpoint_path)


def format_batch_prompt(markets: List[Dict[str, Any]], include_reasoning: bool = True) -> str:
    """Format a batch of markets into the user prompt."""
    lines = [
        f"Classify the following {len(markets)} Polymarket markets in order.",
        f'Return a JSON object with key "classifications" containing exactly {len(markets)} items.',
        'Each item must contain "id", "category", "confidence", and "reasoning".',
        "Use each provided market ID exactly once.",
        "Do not add extra items or any text outside the JSON object.",
        (
            'Keep "reasoning" very short (max 8 words).'
            if include_reasoning
            else 'Set "reasoning" to "" for every item.'
        ),
        "",
        "Markets:",
    ]

    for idx, market in enumerate(markets, start=1):
        parts = [
            f'ID: {int(market["id"])}',
            f'Slug: {json.dumps(str(market.get("market_slug", "")), ensure_ascii=True)}',
        ]
        prompt_text = _normalize_market_text(market.get("prompt_text"))
        if prompt_text:
            parts.append(f'Question: {json.dumps(prompt_text, ensure_ascii=True)}')
        parts.append(
            "Outcomes: "
            f'{json.dumps(str(market.get("answer1", "")), ensure_ascii=True)} vs '
            f'{json.dumps(str(market.get("answer2", "")), ensure_ascii=True)}'
        )
        lines.append(f"{idx}. " + " | ".join(parts))

    return "\n".join(lines)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def classify_batch(
    client: OpenAI,
    markets: List[Dict[str, Any]],
    model: str,
    include_reasoning: bool = True,
) -> List[Dict[str, Any]]:
    """Classify a batch of markets via OpenAI API, return list of classification dicts."""
    user_prompt = format_batch_prompt(markets, include_reasoning=include_reasoning)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        logging.warning(
            "Structured JSON response_format failed for batch of %s markets; retrying without it: %s",
            len(markets),
            exc,
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )

    raw_content = _extract_content_text(response)
    try:
        payload = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model returned invalid JSON: {exc}") from exc

    results = _extract_results(payload)
    return _align_results_to_markets(results, markets)


def _classify_batch_with_fallback(
    batch_idx: int,
    total_batches: int,
    batch: List[Dict[str, Any]],
    model: str,
    include_reasoning: bool,
) -> tuple[int, Dict[str, Any], bool]:
    client = OpenAI()
    entries: Dict[str, Any] = {}

    try:
        results = classify_batch(
            client=client,
            markets=batch,
            model=model,
            include_reasoning=include_reasoning,
        )
        for market, result in zip(batch, results):
            entries[str(int(market["id"]))] = _build_success_entry(market, result)
        return batch_idx, entries, False
    except Exception as exc:
        logging.warning(
            "Batch %s/%s failed for %s markets; falling back to single-market classification: %s",
            batch_idx,
            total_batches,
            len(batch),
            exc,
        )

        for market in batch:
            market_id = str(int(market["id"]))
            try:
                result = classify_batch(
                    client=client,
                    markets=[market],
                    model=model,
                    include_reasoning=include_reasoning,
                )[0]
                entries[market_id] = _build_success_entry(market, result)
            except Exception as item_exc:
                entries[market_id] = _build_failure_entry(market, str(item_exc))
                logging.error(
                    "Failed to classify market_id=%s slug=%s: %s",
                    market["id"],
                    market.get("market_slug", ""),
                    item_exc,
                )
        return batch_idx, entries, True


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    for logger_name in ("openai", "openai._base_client", "httpx", "httpcore"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def derive_insider_plausible(category: str) -> bool:
    """Return True if category is in INSIDER_PLAUSIBLE_CATEGORIES."""
    return category in INSIDER_PLAUSIBLE_CATEGORIES


def print_summary(classifications: Dict[str, Any]) -> None:
    """Print summary statistics to console."""
    successful_entries = [
        entry
        for entry in classifications.values()
        if isinstance(entry, dict) and not _is_failed_entry(entry)
    ]
    failed_entries = [
        entry for entry in classifications.values() if isinstance(entry, dict) and _is_failed_entry(entry)
    ]

    category_counts = Counter(str(entry.get("category")).upper() for entry in successful_entries)
    insider_count = sum(
        1
        for entry in successful_entries
        if derive_insider_plausible(str(entry.get("category", "")).upper())
    )

    print("\nClassification Summary")
    print(f"  Successful:          {len(successful_entries):,}")
    print(f"  Failed:              {len(failed_entries):,}")
    print(f"  Insider-plausible:   {insider_count:,}")
    print(f"  Not insider-plausible: {len(successful_entries) - insider_count:,}")
    if category_counts:
        print("  Categories:")
        for category, count in category_counts.most_common():
            print(f"    {category}: {count:,}")


def main() -> None:
    """Main entry point with argparse CLI."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Classify Polymarket markets by announcement-sensitivity"
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output", type=str, default="data/market_classifications.json")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="data/market_classifications_checkpoint.json",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Inclusive ISO date filter on markets.csv closedTime",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Inclusive ISO date filter on markets.csv closedTime",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_MAX_CONCURRENCY,
        help="Number of batch requests to run in parallel",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--market-ids", type=int, nargs="+", help="Classify specific market IDs only")
    parser.add_argument("--reclassify-failed", action="store_true")
    parser.add_argument(
        "--no-reasoning",
        action="store_true",
        help='Store "" as reasoning to reduce output tokens',
    )
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without API calls")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of markets (for testing)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be > 0")
    if args.max_concurrency <= 0:
        raise SystemExit("--max-concurrency must be > 0")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be > 0 when provided")

    _configure_logging(verbose=args.verbose)

    markets_df = load_markets(args.data_dir)
    source_total_markets = len(markets_df)
    start_ts = _parse_iso_date(args.start_date)
    end_ts = _parse_iso_date(args.end_date)

    if start_ts is not None or end_ts is not None:
        if "closedTime" not in markets_df.columns:
            raise SystemExit("markets.csv does not contain closedTime, so timeframe filtering is unavailable.")
        markets_df = markets_df.copy()
        markets_df["closed_dt"] = pd.to_datetime(markets_df["closedTime"], utc=True, errors="coerce")
        mask = markets_df["closed_dt"].notna()
        if start_ts is not None:
            mask &= markets_df["closed_dt"] >= start_ts
        if end_ts is not None:
            mask &= markets_df["closed_dt"] <= end_ts
        before_timeframe_count = len(markets_df)
        markets_df = markets_df.loc[mask].copy()
        logging.info(
            "Applied closedTime filter: %s -> %s markets (start=%s, end=%s)",
            before_timeframe_count,
            len(markets_df),
            args.start_date,
            args.end_date,
        )

    if args.market_ids:
        requested_market_ids = {int(market_id) for market_id in args.market_ids}
        selected_df = markets_df[markets_df["id"].isin(requested_market_ids)].copy()
        found_ids = set(int(mid) for mid in selected_df["id"].tolist())
        missing_ids = sorted(requested_market_ids - found_ids)
        if missing_ids:
            logging.warning("Requested market IDs not found in markets.csv: %s", missing_ids)
    else:
        selected_df = markets_df.copy()

    if args.limit is not None:
        selected_df = selected_df.head(args.limit).copy()

    selected_markets = selected_df.to_dict(orient="records")
    logging.info(
        "Loaded %s target markets from %s (%s total rows in source markets.csv)",
        len(selected_markets),
        args.data_dir,
        source_total_markets,
    )

    classifications: Dict[str, Any] = {}
    existing_output = load_existing_classifications(args.output)
    if existing_output:
        classifications.update(existing_output)
        logging.info("Loaded %s existing classifications from %s", len(existing_output), args.output)

    if args.resume:
        checkpoint_classifications = load_existing_classifications(args.checkpoint)
        if checkpoint_classifications:
            classifications.update(checkpoint_classifications)
            logging.info(
                "Loaded %s classifications from checkpoint %s",
                len(checkpoint_classifications),
                args.checkpoint,
            )

    markets_to_classify: List[Dict[str, Any]] = []
    reused_count = 0
    for market in selected_markets:
        market_id = str(int(market["id"]))
        existing_entry = classifications.get(market_id)
        if existing_entry is None:
            markets_to_classify.append(market)
            continue
        if args.reclassify_failed and _is_failed_entry(existing_entry):
            markets_to_classify.append(market)
            continue
        reused_count += 1

    logging.info(
        "Classification plan: %s pending, %s reused, batch_size=%s, max_concurrency=%s",
        len(markets_to_classify),
        reused_count,
        args.batch_size,
        args.max_concurrency,
    )

    if args.dry_run:
        batches = _batched(markets_to_classify, args.batch_size)
        if not batches:
            print("No markets require classification.")
            return
        for batch_idx, batch in enumerate(batches, start=1):
            print(f"\n--- Batch {batch_idx}/{len(batches)} ({len(batch)} markets) ---")
            print(format_batch_prompt(batch, include_reasoning=not args.no_reasoning))
        return

    selected_market_ids = {str(int(market["id"])) for market in selected_markets}
    if not markets_to_classify:
        selected_entries = {
            market_id: entry
            for market_id, entry in classifications.items()
            if market_id in selected_market_ids
        }
        selected_classified_count = sum(
            1 for entry in selected_entries.values() if not _is_failed_entry(entry)
        )
        selected_failed_count = sum(
            1 for entry in selected_entries.values() if _is_failed_entry(entry)
        )
        stored_classified_count = sum(
            1 for entry in classifications.values() if not _is_failed_entry(entry)
        )
        stored_failed_count = sum(
            1 for entry in classifications.values() if _is_failed_entry(entry)
        )
        metadata = {
            "created_at": _utc_now_iso(),
            "model": args.model,
            "total_markets": len(selected_markets),
            "source_total_markets": source_total_markets,
            "classified": selected_classified_count,
            "failed": selected_failed_count,
            "stored_classified": stored_classified_count,
            "stored_failed": stored_failed_count,
            "schema_version": SCHEMA_VERSION,
            "data_dir": args.data_dir,
            "batch_size": args.batch_size,
            "max_concurrency": args.max_concurrency,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "resume": args.resume,
            "reclassify_failed": args.reclassify_failed,
            "no_reasoning": args.no_reasoning,
            "limit": args.limit,
            "market_ids": args.market_ids,
            "duration_seconds": 0.0,
        }
        save_classifications(classifications, metadata, args.output)
        logging.info("No markets required classification; refreshed metadata at %s", args.output)
        print_summary(classifications)
        return

    if markets_to_classify and not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Add it to your environment or .env file.")

    batches = _batched(markets_to_classify, args.batch_size)
    started_at = time.time()
    total_batches = len(batches)
    completed_batches = 0
    fallback_batches = 0
    next_batch_idx = 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
        in_flight: Dict[concurrent.futures.Future[tuple[int, Dict[str, Any], bool]], int] = {}

        while next_batch_idx <= total_batches and len(in_flight) < args.max_concurrency:
            batch = batches[next_batch_idx - 1]
            future = executor.submit(
                _classify_batch_with_fallback,
                next_batch_idx,
                total_batches,
                batch,
                args.model,
                not args.no_reasoning,
            )
            in_flight[future] = next_batch_idx
            next_batch_idx += 1

        while in_flight:
            done, _ = concurrent.futures.wait(
                in_flight.keys(),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )

            for future in done:
                batch_idx = in_flight.pop(future)
                _, batch_entries, used_fallback = future.result()
                classifications.update(batch_entries)
                completed_batches += 1
                if used_fallback:
                    fallback_batches += 1

                if (
                    completed_batches % CHECKPOINT_INTERVAL == 0
                    or completed_batches == total_batches
                ):
                    save_checkpoint(classifications, args.checkpoint)
                    logging.info("Checkpoint saved: %s entries", len(classifications))

                logging.info(
                    "Processed batch %s/%s | total_saved=%s | fallback_batches=%s",
                    batch_idx,
                    total_batches,
                    len(classifications),
                    fallback_batches,
                )

                if next_batch_idx <= total_batches:
                    batch = batches[next_batch_idx - 1]
                    next_future = executor.submit(
                        _classify_batch_with_fallback,
                        next_batch_idx,
                        total_batches,
                        batch,
                        args.model,
                        not args.no_reasoning,
                    )
                    in_flight[next_future] = next_batch_idx
                    next_batch_idx += 1

    selected_entries = {
        market_id: entry
        for market_id, entry in classifications.items()
        if market_id in selected_market_ids
    }
    classified_count = sum(1 for entry in selected_entries.values() if not _is_failed_entry(entry))
    failed_count = sum(1 for entry in selected_entries.values() if _is_failed_entry(entry))
    stored_classified_count = sum(
        1 for entry in classifications.values() if not _is_failed_entry(entry)
    )
    stored_failed_count = sum(1 for entry in classifications.values() if _is_failed_entry(entry))

    metadata = {
        "created_at": _utc_now_iso(),
        "model": args.model,
        "total_markets": len(selected_markets),
        "source_total_markets": source_total_markets,
        "classified": classified_count,
        "failed": failed_count,
        "stored_classified": stored_classified_count,
        "stored_failed": stored_failed_count,
        "schema_version": SCHEMA_VERSION,
        "data_dir": args.data_dir,
        "batch_size": args.batch_size,
        "max_concurrency": args.max_concurrency,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "resume": args.resume,
        "reclassify_failed": args.reclassify_failed,
        "no_reasoning": args.no_reasoning,
        "limit": args.limit,
        "market_ids": args.market_ids,
        "duration_seconds": round(time.time() - started_at, 2),
    }
    save_classifications(classifications, metadata, args.output)
    logging.info("Saved classifications to %s", args.output)
    print_summary(classifications)


if __name__ == "__main__":
    main()
