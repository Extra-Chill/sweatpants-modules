"""Mediavine Revenue Reports module for Sweatpants.

Pulls per-period, per-URL revenue from Mediavine using its REAL publisher API
(reverse-engineered + tested 2026-06-21). This is pure HTTP — no browser, no
Playwright. Every call goes out through sweatpants' rotating proxy via the core
``proxied_request`` primitive.

Architecture (chubes.net runs sweatpants — NOT EC prod):

    extrachill-analytics (EC prod) --signed job request--> sweatpants @ chubes.net
                                                            mediavine-reports module
                                                            (HTTP -> api-publishers.mediavine.com)
                                   <--HMAC-signed callback (revenue rows)--+

EC prod stays clean (no scraper deps); chubes.net does the API pull; results
return via the proven signed-callback pattern.

Depends on sweatpants#26: the signed-callback SENDER is a core SDK primitive
(``Module.send_signed_callback``) — this module does NOT hand-roll HMAC, it
calls the shared primitive.

============================================================================
THE REAL MEDIAVINE API (all endpoints tested 2026-06-21)
============================================================================
1) LOGIN — POST https://api-publishers.mediavine.com/graphql
   GraphQL mutation ``unidashSignIn`` with {email, password}. Returns
   ``accessToken`` (Bearer), ``refreshToken``, ``expiresIn``, and a
   ``twoFactorRequired`` flag. 2FA is not supported here — if the account
   demands it we raise a clear error.

2) PER-PAGE REVENUE CSV (report_type "pages", the primary pull) —
   GET https://api-publishers.mediavine.com/reports/{SITE_ID}/pages.csv
       ?startDate={MM/DD/YYYY}&endDate={MM/DD/YYYY}&perPage=100000&useMonetizable=false
   Authorization: Bearer {accessToken}. Returns text/csv with columns
   slug,views,revenue,rpm,cpm,viewability,fillRate,impressionsPerPageview.
   Date format is MM/DD/YYYY (url-encoded by httpx).

3) AGGREGATE METRICS (report_type "summary", optional) —
   POST https://api-publishers.mediavine.com/graphql with Bearer auth.
   ``metricsSummary`` query returns a single site-level row (earnings,
   pageviews, sessions, cpm, sessionRpm, pageRpm, paidImpressions). NOTE this
   one uses ISO timestamps, NOT MM/DD/YYYY.
============================================================================
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime
from typing import Any, AsyncIterator, Optional

from sweatpants import Module, proxied_request


# Mediavine publisher API. These hosts/paths are the tested, stable surface.
API_BASE = "https://api-publishers.mediavine.com"
GRAPHQL_URL = f"{API_BASE}/graphql"

# Default site identifier for Extra Chill — base64 of "InternalSite:11476".
# Exposed as a module input so the same module serves other sites/accounts.
DEFAULT_SITE_ID = "SW50ZXJuYWxTaXRlOjExNDc2"

# Per-request timeout (seconds). A full-year pages.csv with perPage=100000 can
# be a large payload, so give it generous headroom.
REQUEST_TIMEOUT = 120.0

# GraphQL login mutation (publisher dashboard sign-in).
LOGIN_MUTATION = (
    "mutation LoginFormMutation($data: UnidashSignInInput!) { "
    "unidashSignIn(data: $data) { "
    "accessToken expiresIn refreshToken tokenType twoFactorRequired userId } }"
)

# GraphQL aggregate-metrics query (site-level totals for report_type "summary").
METRICS_SUMMARY_QUERY = (
    "query MiniStatusTrackerQuery($data: MetricsSummaryInput!) { "
    "metricsSummary(data: $data) { summary { "
    "earnings pageviews sessions cpm sessionRpm pageRpm paidImpressions } } }"
)


class MediavineReports(Module):
    """Pull per-period Mediavine revenue rows via the real HTTP API and callback the result."""

    async def run(
        self, inputs: dict[str, Any], settings: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        """Execute the revenue pull.

        Args:
            inputs: periods | (period_start, period_end[, period]), report_type,
                site_id, callback_url, callback_secret, callback_issuer,
                callback_user_id
            settings: mediavine_email, mediavine_password
        """
        report_type = inputs.get("report_type", "pages")
        if report_type not in ("pages", "summary"):
            raise ValueError(
                f"report_type {report_type!r} is not supported; "
                "use 'pages' (per-URL CSV) or 'summary' (site-level aggregate)"
            )

        site_id = inputs.get("site_id") or DEFAULT_SITE_ID

        periods = self._normalize_periods(inputs)
        if not periods:
            raise ValueError(
                "Provide either `periods` (a list) or both `period_start` and `period_end`"
            )

        email = settings.get("mediavine_email")
        password = settings.get("mediavine_password")
        if not email or not password:
            raise ValueError(
                "mediavine_email and mediavine_password settings are required"
            )

        callback_url = inputs.get("callback_url")
        callback_secret = inputs.get("callback_secret")
        callback_issuer = inputs.get("callback_issuer", "sweatpants")
        callback_user_id = inputs.get("callback_user_id")

        # Resume support: skip periods already completed in a prior run.
        completed = set(self.get_checkpoint("completed_periods", []) or [])

        await self.log(
            f"Mediavine revenue pull: {len(periods)} period(s), "
            f"report_type={report_type}, site_id={site_id}"
        )
        await self.save_checkpoint(
            stage="started",
            report_type=report_type,
            total_periods=len(periods),
        )

        # Authenticate once for the whole job; the Bearer token is reused across
        # every period request.
        access_token = await self._login(email, password)

        collected: list[dict[str, Any]] = []

        for spec in periods:
            label = spec["period"]
            if label in completed:
                await self.log(f"Skipping already-completed period: {label}")
                continue

            await self.log(
                f"Pulling period {label} "
                f"({spec['period_start']} → {spec['period_end']})"
            )

            rows = await self._fetch_period(
                access_token,
                site_id=site_id,
                report_type=report_type,
                period_start=spec["period_start"],
                period_end=spec["period_end"],
            )

            # Tag every row with its period so the receiver can key on it.
            for row in rows:
                row["period"] = label

            period_result = {
                "period": label,
                "period_start": spec["period_start"],
                "period_end": spec["period_end"],
                "report_type": report_type,
                "rows": rows,
                "row_count": len(rows),
            }
            collected.append(period_result)

            # Checkpoint so a restart resumes after the last completed period,
            # and yield incrementally so a multi-month pull surfaces progress.
            completed.add(label)
            await self.save_checkpoint(
                stage="period_complete",
                completed_periods=sorted(completed),
                last_period=label,
            )
            yield {
                "status": "period_complete",
                "period": label,
                "row_count": len(rows),
                "rows": rows,
            }

        final_result = {
            "status": "complete",
            "report_type": report_type,
            "periods": collected,
            "period_count": len(collected),
            "total_rows": sum(p["row_count"] for p in collected),
        }
        await self.save_checkpoint(stage="complete")
        yield final_result

        # Fire the completion callback (best-effort) via the core SDK primitive
        # (sweatpants#26). Signing + delivery + job_id injection live in core;
        # this module does NOT hand-roll the HMAC POST.
        if callback_url:
            await self.send_signed_callback(
                callback_url,
                final_result,
                callback_secret,
                issuer=callback_issuer,
                user_id=callback_user_id,
            )

    # -----------------------------------------------------------------
    # Input handling
    # -----------------------------------------------------------------

    @staticmethod
    def _normalize_periods(inputs: dict[str, Any]) -> list[dict[str, str]]:
        """Normalize inputs into a list of {period, period_start, period_end}.

        Accepts either a `periods` list (each item a dict with period_start +
        period_end and an optional period label) or a single
        period_start/period_end pair. The `periods` form wins when both are set.
        """
        raw = inputs.get("periods")
        if raw:
            if isinstance(raw, str):
                # Inputs may arrive JSON-encoded depending on the transport.
                raw = json.loads(raw)
            normalized: list[dict[str, str]] = []
            for item in raw:
                start = item["period_start"]
                end = item["period_end"]
                label = item.get("period") or f"{start}..{end}"
                normalized.append(
                    {"period": label, "period_start": start, "period_end": end}
                )
            return normalized

        start = inputs.get("period_start")
        end = inputs.get("period_end")
        if start and end:
            label = inputs.get("period") or f"{start}..{end}"
            return [{"period": label, "period_start": start, "period_end": end}]

        return []

    # -----------------------------------------------------------------
    # Date conversion — the two Mediavine endpoints want different formats.
    # pages.csv wants MM/DD/YYYY; metricsSummary wants an ISO timestamp.
    # Inputs always arrive as YYYY-MM-DD.
    # -----------------------------------------------------------------

    @staticmethod
    def _to_mmddyyyy(date_ymd: str) -> str:
        """Convert a YYYY-MM-DD date to MM/DD/YYYY for the pages.csv endpoint."""
        dt = datetime.strptime(date_ymd, "%Y-%m-%d")
        return dt.strftime("%m/%d/%Y")

    @staticmethod
    def _to_iso(date_ymd: str, *, end_of_day: bool = False) -> str:
        """Convert a YYYY-MM-DD date to an ISO-8601 UTC timestamp.

        metricsSummary uses ISO timestamps. The sample query used
        ``2023-01-01T05:00:00.000Z`` (US-Eastern midnight expressed in UTC).
        We anchor the day boundaries at UTC midnight, which is the safe,
        unambiguous interpretation of a date-only input; ``end_of_day`` pushes
        the end bound to the final millisecond so the whole day is inclusive.
        """
        dt = datetime.strptime(date_ymd, "%Y-%m-%d")
        if end_of_day:
            return dt.strftime("%Y-%m-%dT23:59:59.999Z")
        return dt.strftime("%Y-%m-%dT00:00:00.000Z")

    # -----------------------------------------------------------------
    # Mediavine HTTP API — login + per-period fetch
    # -----------------------------------------------------------------

    async def _login(self, email: str, password: str) -> str:
        """Authenticate against the Mediavine publisher GraphQL API.

        POSTs the ``unidashSignIn`` mutation and returns the Bearer access
        token. Raises on 2FA-required accounts (not supported here) or any
        non-success response.
        """
        await self.log("Authenticating with Mediavine publisher API")

        body = {
            "query": LOGIN_MUTATION,
            "operationName": "LoginFormMutation",
            "variables": {"data": {"email": email, "password": password}},
        }

        response = await proxied_request(
            "POST",
            GRAPHQL_URL,
            headers={"Content-Type": "application/json"},
            json=body,
            timeout=REQUEST_TIMEOUT,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Mediavine login failed: HTTP {response.status_code} — "
                f"{response.text[:500]!r}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"Mediavine login returned non-JSON response: {response.text[:500]!r}"
            ) from exc

        # GraphQL transports errors in a top-level `errors` array even on HTTP 200.
        if data.get("errors"):
            raise RuntimeError(f"Mediavine login GraphQL errors: {data['errors']!r}")

        sign_in = (data.get("data") or {}).get("unidashSignIn") or {}

        if sign_in.get("twoFactorRequired"):
            raise RuntimeError(
                "Mediavine account requires two-factor authentication — the 2FA "
                "login path is not supported by this module. Disable 2FA for the "
                "reporting account or implement the 2FA flow."
            )

        access_token = sign_in.get("accessToken")
        if not access_token:
            raise RuntimeError(
                "Mediavine login succeeded but no accessToken was returned: "
                f"{sign_in!r}"
            )

        # refreshToken/expiresIn are returned too; not needed for a single
        # short-lived job but captured in the checkpoint for future use.
        await self.save_checkpoint(
            stage="authenticated",
            token_expires_in=sign_in.get("expiresIn"),
        )
        await self.log("Mediavine authentication complete")
        return access_token

    async def _fetch_period(
        self,
        access_token: str,
        *,
        site_id: str,
        report_type: str,
        period_start: str,
        period_end: str,
    ) -> list[dict[str, Any]]:
        """Fetch revenue rows for one date range.

        For ``report_type == "pages"`` this GETs the per-URL pages.csv and parses
        it into {slug, views, revenue, rpm, ...} rows. For
        ``report_type == "summary"`` it POSTs the metricsSummary GraphQL query
        and returns the single site-level aggregate row.

        Returns row dicts; the caller tags each with its period label.
        """
        if report_type == "summary":
            return await self._fetch_summary(
                access_token,
                site_id=site_id,
                period_start=period_start,
                period_end=period_end,
            )
        return await self._fetch_pages(
            access_token,
            site_id=site_id,
            period_start=period_start,
            period_end=period_end,
        )

    async def _fetch_pages(
        self,
        access_token: str,
        *,
        site_id: str,
        period_start: str,
        period_end: str,
    ) -> list[dict[str, Any]]:
        """GET the per-URL pages.csv report and parse it into revenue rows."""
        url = f"{API_BASE}/reports/{site_id}/pages.csv"
        params = {
            "startDate": self._to_mmddyyyy(period_start),
            "endDate": self._to_mmddyyyy(period_end),
            "perPage": "100000",
            "useMonetizable": "false",
        }

        await self.log(
            f"GET pages.csv for {site_id} "
            f"({params['startDate']} → {params['endDate']})"
        )
        response = await proxied_request(
            "GET",
            url,
            headers={"Authorization": f"Bearer {access_token}"},
            params=params,
            timeout=REQUEST_TIMEOUT,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Mediavine pages.csv request failed: HTTP {response.status_code} — "
                f"{response.text[:500]!r}"
            )

        return self._parse_revenue_csv(response.text)

    async def _fetch_summary(
        self,
        access_token: str,
        *,
        site_id: str,
        period_start: str,
        period_end: str,
    ) -> list[dict[str, Any]]:
        """POST the metricsSummary GraphQL query and return the aggregate row.

        Returns a single-element list (one site-level aggregate row) so the
        per-period plumbing in ``run`` is uniform with the CSV path.
        """
        body = {
            "query": METRICS_SUMMARY_QUERY,
            "operationName": "MiniStatusTrackerQuery",
            "variables": {
                "data": {
                    "siteId": site_id,
                    "startDate": self._to_iso(period_start),
                    "endDate": self._to_iso(period_end, end_of_day=True),
                }
            },
        }

        await self.log(f"POST metricsSummary for {site_id}")
        response = await proxied_request(
            "POST",
            GRAPHQL_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {access_token}",
            },
            json=body,
            timeout=REQUEST_TIMEOUT,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Mediavine metricsSummary request failed: HTTP {response.status_code} — "
                f"{response.text[:500]!r}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(
                "Mediavine metricsSummary returned non-JSON response: "
                f"{response.text[:500]!r}"
            ) from exc

        if data.get("errors"):
            raise RuntimeError(
                f"Mediavine metricsSummary GraphQL errors: {data['errors']!r}"
            )

        summary = (
            ((data.get("data") or {}).get("metricsSummary") or {}).get("summary")
            or {}
        )
        if not summary:
            await self.log(
                "metricsSummary returned an empty summary for this period",
                level="WARNING",
            )
            return []

        # Normalize onto the same {slug, views, revenue, rpm} shape the revenue
        # import expects, with the extra aggregate fields carried alongside. The
        # synthetic slug marks this as a site-level total, not a per-URL row.
        revenue = self._num(summary.get("earnings"))
        views = int(self._num(summary.get("pageviews")))
        row = {
            "slug": "__site_total__",
            "views": views,
            "revenue": revenue,
            "rpm": self._num(summary.get("pageRpm")),
            "sessions": int(self._num(summary.get("sessions"))),
            "cpm": self._num(summary.get("cpm")),
            "session_rpm": self._num(summary.get("sessionRpm")),
            "page_rpm": self._num(summary.get("pageRpm")),
            "paid_impressions": int(self._num(summary.get("paidImpressions"))),
        }
        return [row]

    @staticmethod
    def _num(value: Any) -> float:
        """Coerce a CSV/JSON value to float, tolerating $, commas, blanks, None."""
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        cleaned = str(value).replace("$", "").replace(",", "").strip()
        if not cleaned:
            return 0.0
        try:
            return float(cleaned)
        except ValueError:
            return 0.0

    def _parse_revenue_csv(self, csv_text: str) -> list[dict[str, Any]]:
        """Parse a Mediavine pages.csv response into normalized revenue rows.

        The tested column set is:
            slug,views,revenue,rpm,cpm,viewability,fillRate,impressionsPerPageview

        Maps onto the shape the extrachill-analytics revenue import expects
        ({slug, views, revenue, rpm}) and carries the remaining ad-quality
        metrics alongside. Header lookup is case-insensitive and tolerant of
        minor renames; rows without a slug are skipped.
        """
        rows: list[dict[str, Any]] = []
        reader = csv.DictReader(io.StringIO(csv_text))

        # Candidate header aliases (lowercased) per logical field.
        slug_keys = ("slug", "url", "page", "page url", "path")
        views_keys = ("views", "pageviews", "page views")
        revenue_keys = ("revenue", "earnings", "total revenue", "est. revenue")
        rpm_keys = ("rpm", "page rpm", "session rpm")
        cpm_keys = ("cpm",)
        viewability_keys = ("viewability",)
        fill_rate_keys = ("fillrate", "fill rate")
        ipp_keys = ("impressionsperpageview", "impressions per pageview")

        def _pick(record: dict[str, str], keys: tuple[str, ...]) -> Optional[str]:
            lowered = {(k or "").strip().lower(): v for k, v in record.items()}
            for key in keys:
                if key in lowered and lowered[key] not in (None, ""):
                    return lowered[key]
            return None

        for record in reader:
            slug = _pick(record, slug_keys)
            if not slug:
                continue
            rows.append(
                {
                    "slug": slug.strip(),
                    "views": int(self._num(_pick(record, views_keys))),
                    "revenue": self._num(_pick(record, revenue_keys)),
                    "rpm": self._num(_pick(record, rpm_keys)),
                    "cpm": self._num(_pick(record, cpm_keys)),
                    "viewability": self._num(_pick(record, viewability_keys)),
                    "fill_rate": self._num(_pick(record, fill_rate_keys)),
                    "impressions_per_pageview": self._num(_pick(record, ipp_keys)),
                }
            )

        return rows
