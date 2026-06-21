# mediavine-reports

Pulls **per-period, per-URL Mediavine revenue** via Mediavine's real publisher
HTTP API (reverse-engineered + tested), and POSTs the rows back to a receiver
(e.g. `extrachill-analytics`) using an HMAC-signed callback. Pure HTTP — no
browser — and every request goes out through sweatpants' rotating proxy.

## The real Mediavine API

Mediavine **does** have a publisher API. It's undocumented but stable and
tested (2026-06-21). Three endpoints, all under
`https://api-publishers.mediavine.com`:

1. **Login** — `POST /graphql`, the `unidashSignIn` mutation with
   `{email, password}` → returns a Bearer `accessToken` (plus `refreshToken`
   and `expiresIn`). Accounts with 2FA enabled return `twoFactorRequired:true`;
   that path is **not** supported and raises a clear error.
2. **Per-page revenue CSV** (primary, `report_type: "pages"`) —
   `GET /reports/{site_id}/pages.csv?startDate=MM/DD/YYYY&endDate=MM/DD/YYYY&perPage=100000&useMonetizable=false`
   with `Authorization: Bearer {token}` → `text/csv` with columns
   `slug,views,revenue,rpm,cpm,viewability,fillRate,impressionsPerPageview`.
3. **Aggregate metrics** (optional, `report_type: "summary"`) —
   `POST /graphql`, the `metricsSummary` query → a single site-level row
   (`earnings, pageviews, sessions, cpm, sessionRpm, pageRpm, paidImpressions`).
   This one uses **ISO timestamps**, not `MM/DD/YYYY`.

This is a clean, deterministic HTTP integration — no Playwright, no selectors,
no scrape fragility. Keeping it on sweatpants (chubes.net) — **not** baked into
a WordPress plugin — means EC prod stays clean (no scraper/HTTP-auth deps) and
the proxy rotation lives in the tool built for it.

## Architecture

```
extrachill-analytics (EC prod) --signed job request--> sweatpants @ chubes.net
                                                        mediavine-reports module
                                                        (HTTP -> api-publishers.mediavine.com)
                               <--HMAC-signed callback (revenue rows)--+
```

The WP glue (companion: `extrachill-analytics#65`) mints a signed sweatpants
job-request token, POSTs to the chubes.net `/jobs` endpoint, then receives the
HMAC-signed callback and feeds the rows into the existing
`extrachill analytics revenue import` path (period-keyed). The revenue store +
import logic already exist — this module only **feeds** them; it does not
reimplement revenue parsing/rollup.

## Inputs

| input | required | description |
|-------|----------|-------------|
| `periods` | one of | JSON list of `{period, period_start, period_end}` (YYYY-MM-DD). Use for multi-bucket backfill. Wins over the single-period inputs. |
| `period_start` / `period_end` | one of | A single reporting period (YYYY-MM-DD). |
| `period` | no | Label for the single-period case. Defaults to `<start>..<end>`. |
| `report_type` | no | `pages` (per-URL revenue CSV, default) or `summary` (site-level aggregate). |
| `site_id` | no | Mediavine site id (base64 of `InternalSite:<id>`). Defaults to Extra Chill (`SW50ZXJuYWxTaXRlOjExNDc2`). |
| `callback_url` | no | HTTPS receiver for the completed revenue rows. |
| `callback_secret` | no | Shared HMAC secret. Signs the callback (sweatpants core format → `wp_native_auth_verify_external_token`). |
| `callback_issuer` | no | `iss` claim. Default `sweatpants`. |
| `callback_user_id` | no | `sub` claim — the receiving WP user_id. |

## Settings (secrets — stay on sweatpants, never on WP)

| setting | required | description |
|---------|----------|-------------|
| `mediavine_email` | yes | Publisher account login email. |
| `mediavine_password` | yes | Publisher account login password. |

## Date handling

Inputs are always `YYYY-MM-DD`. The module converts per endpoint:

- **`pages.csv`** wants `MM/DD/YYYY` (httpx url-encodes the `/`).
- **`metricsSummary`** wants ISO timestamps; the module anchors the range at
  UTC midnight (`...T00:00:00.000Z`) and pushes the end bound to the final
  millisecond of the day so the period is inclusive.

## Output / callback shape

Each completed period yields incrementally, and on completion the module POSTs:

```json
{
  "job_id": "<sweatpants job id>",
  "status": "complete",
  "report_type": "pages",
  "period_count": 5,
  "total_rows": 1234,
  "periods": [
    {
      "period": "2022",
      "period_start": "2022-01-01",
      "period_end": "2022-12-31",
      "report_type": "pages",
      "row_count": 247,
      "rows": [
        {"slug": "/some-post", "views": 10000, "revenue": 123.45, "rpm": 12.3, "period": "2022"}
      ]
    }
  ]
}
```

Row shape (`slug`, `views`, `revenue`, `rpm`, `period`) matches what the
`extrachill analytics revenue import` path expects. The `pages` rows carry the
extra ad-quality metrics (`cpm`, `viewability`, `fill_rate`,
`impressions_per_pageview`) alongside. A `summary` pull returns a single
`__site_total__` row per period with the aggregate fields.

## Resume / checkpointing

The module checkpoints `completed_periods` after each period, so a multi-bucket
backfill resumes after a restart without re-pulling finished periods.

## Operating guidance

- This automates **our own** publisher dashboard export of **our own** revenue —
  low frequency, our account only. Keep it that way: a backfill run, then
  monthly. Be mindful of ToS.
- First milestone: a **backfill** pulling yearly buckets 2022→2026 (5 periods) —
  that alone draws the revenue curve (2022 peak → Sept-2023 HCU cliff → now).
  Monthly granularity around the 2023 cliff is a follow-up.

## Dependencies

Depends on **sweatpants#26** — the signed-callback SENDER is a core SDK
primitive (`Module.send_signed_callback`). This module does not hand-roll HMAC.

No module-specific runtime deps; uses the stdlib (`csv`, `io`, `json`,
`datetime`) plus sweatpants core primitives (`Module`, `proxied_request`).
