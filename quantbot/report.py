"""Builds a single self-contained HTML report you can open in any web browser."""
from __future__ import annotations

import html
from datetime import datetime

import numpy as np
import pandas as pd

PCT = {"Total return", "CAGR", "Volatility", "Max drawdown", "Positive months", "Best month",
       "Worst month", "Alpha (yearly)", "Cost drag (yearly)", "Turnover (yearly)"}

CSS = """
:root{--bg:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--line:#e6e5e1;--accent:#2a78d6;
--good:#0a7d32;--bad:#c23b3a;color-scheme:light}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:1040px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:28px;margin:0 0 4px}h2{font-size:19px;margin:40px 0 8px;padding-top:8px;
border-top:1px solid var(--line)}p.sub{color:var(--ink2);margin:0 0 20px}
.note{color:var(--ink2);font-size:14px;margin:4px 0 12px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:16px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.card .k{color:var(--ink2);font-size:13px}.card .v{font-size:24px;font-weight:650;margin-top:2px}
.verdict{background:#eef4fc;border:1px solid #cde2fb;border-radius:10px;padding:14px 16px;margin:16px 0}
.wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px;margin:8px 0}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{color:var(--ink2);font-weight:600;background:#f6f5f2}
td.pos{color:var(--good)}td.neg{color:var(--bad)}img{width:100%;height:auto;margin:8px 0;
border:1px solid var(--line);border-radius:10px}
.pill{display:inline-block;padding:1px 8px;border-radius:99px;background:#f0efec;color:var(--ink2);font-size:12px}
"""


def fmt(name: str, v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    if name in PCT:
        return f"{v*100:,.1f}%" if name != "Turnover (yearly)" else f"{v:,.1f}× a year"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, (float, np.floating)):
        return f"{v:,.2f}"
    return html.escape(str(v))


def table(df: pd.DataFrame, pct_cols: set[str] = frozenset(), signed: set[str] = frozenset(),
          decimals: dict[str, int] | None = None) -> str:
    decimals = decimals or {}
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in [df.index.name or ""] + list(df.columns))
    rows = []
    for idx, r in df.iterrows():
        cells = [f"<td>{html.escape(str(idx))}</td>"]
        for c in df.columns:
            v = r[c]
            cls = ""
            if isinstance(v, (float, np.floating)) and not np.isnan(v):
                if c in signed:
                    cls = " class='pos'" if v > 0 else (" class='neg'" if v < 0 else "")
                if c in pct_cols:
                    s = f"{v*100:+.2f}%" if c in signed else f"{v*100:.1f}%"
                else:
                    d = decimals.get(c, 3)
                    s = f"{v:+.{d}f}" if c in signed else f"{v:.{d}f}"
            else:
                s = html.escape(str(v))
            cells.append(f"<td{cls}>{s}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='wrap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def build_report(ctx: dict) -> str:
    labels = ctx["labels"]
    m = ctx["metrics"]               # {strategy: {metric: value}}
    live = ctx["live_strategy"]
    best = m.get(live, {})
    bench = m.get("benchmark", {})

    # ---- verdict in plain English
    lines = []
    if best:
        beat = best["CAGR"] > bench.get("CAGR", -1)
        lines.append(f"<b>{labels[live]}</b> grew {fmt('CAGR', best['CAGR'])} a year after costs, "
                     f"versus {fmt('CAGR', bench.get('CAGR'))} for simply holding the benchmark "
                     f"({'beat' if beat else 'did NOT beat'} it).")
        lines.append(f"Its worst fall was {fmt('Max drawdown', best['Max drawdown'])} "
                     f"(benchmark {fmt('Max drawdown', bench.get('Max drawdown'))}), and its Sharpe ratio was "
                     f"{best['Sharpe']:.2f} (benchmark {bench.get('Sharpe', float('nan')):.2f}).")
    hm = ctx.get("holdout_metrics", {}).get(live)
    hb = ctx.get("holdout_metrics", {}).get("benchmark")
    if hm and hb:
        lines.append(f"On the final holdout year it never saw during model selection: "
                     f"{fmt('Total return', hm['Total return'])} vs benchmark "
                     f"{fmt('Total return', hb['Total return'])}.")
    if ctx.get("demo"):
        lines.append("<b>These results use made-up DEMO data.</b> They only prove the program works.")
    verdict = "<div class='verdict'>" + " ".join(lines) + "</div>"

    cards = ""
    if best:
        for k in ("CAGR", "Sharpe", "Max drawdown", "Turnover (yearly)"):
            cards += f"<div class='card'><div class='k'>{labels[live]} · {k}</div><div class='v'>{fmt(k, best.get(k))}</div></div>"

    mt = pd.DataFrame(m).rename(columns=labels)
    mt.index.name = "Metric"
    mt_html = "<div class='wrap'><table><thead><tr><th>Metric</th>" + "".join(
        f"<th>{html.escape(c)}</th>" for c in mt.columns) + "</tr></thead><tbody>"
    for k, row in mt.iterrows():
        mt_html += f"<tr><td>{k}</td>" + "".join(f"<td>{fmt(k, v)}</td>" for v in row) + "</tr>"
    mt_html += "</tbody></table></div>"

    ev = ctx["eval_summary"].copy()
    ev["model"] = ev["model"].map(ctx["model_labels"])
    ev = ev.set_index("model")
    ev.index.name = "Model"
    ev = ev.rename(columns={"period": "Period", "folds": "Folds", "auc": "AUC", "accuracy": "Accuracy",
                            "ic": "IC", "ic_tstat": "IC t-stat", "ic_positive_days": "IC>0 days",
                            "weekly_spread": "Top-bottom weekly"})
    ev_html = table(ev, pct_cols={"Accuracy", "IC>0 days", "Top-bottom weekly"},
                    signed={"IC", "Top-bottom weekly"}, decimals={"IC t-stat": 2, "Folds": 0})

    folds_html = "<br>".join(html.escape(str(f)) for f in ctx["folds"])

    picks_html = ""
    for k, w in ctx["latest_picks"].items():
        w = w[w.abs() > 1e-9].sort_values(ascending=False)
        chips = " ".join(f"<span class='pill'>{t} {v*100:.1f}%</span>" for t, v in w.items())
        picks_html += f"<p><b>{labels[k]}</b> ({ctx['latest_date']}): {chips or 'no picks'}</p>"

    yr = ctx["yearly"].rename(columns=labels)
    yr.index.name = "Year"
    yr_html = table(yr, pct_cols=set(yr.columns), signed=set(yr.columns))

    img = lambda b: f"<img alt='' src='data:image/png;base64,{b}'>"  # noqa: E731
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Strategy Report</title>
<style>{CSS}</style></head><body><main>
<h1>Strategy research report</h1>
<p class="sub">Created {datetime.now():%d %b %Y %H:%M} · data: {html.escape(ctx['source'])} ·
{ctx['n_stocks']} candidate stocks · backtest {ctx['bt_start']} to {ctx['bt_end']} ·
costs {ctx['cost_bps']} bps per trade</p>
{verdict}
<div class="cards">{cards}</div>

<h2>Step 8 · Portfolio performance</h2>
<p class="note">All strategies are tested over the same out-of-sample period, after trading costs.
The shaded area is the holdout year that was never used to choose the model.</p>
{img(ctx['charts']['equity'])}
{img(ctx['charts']['drawdown'])}
{mt_html}
{img(ctx['charts']['yearly'])}
{yr_html}

<h2>Step 7 · Backtest settings</h2>
<p class="note">Rebalance every {ctx['rebalance']} trading days · hold top {ctx['top_n']} ·
{'long/short' if ctx['long_short'] else 'long only'} · weighting: {ctx['weighting']} ·
max {ctx['max_weight']*100:.0f}% per stock · trade {ctx['lag']} day(s) after the signal.</p>

<h2>Step 6 · Latest signals (what each strategy would hold now)</h2>
{picks_html}

<h2>Steps 4 &amp; 5 · Machine learning models and evaluation</h2>
<p class="note">Chosen model: <b>{ctx['model_labels'][ctx['best_model']]}</b> (highest average IC on the
development folds). AUC above 0.50 and a positive IC mean the model ranks stocks better than chance.
In stock picking even small edges (IC 0.02–0.05) can be valuable; very high numbers usually mean a bug.</p>
{ev_html}
{img(ctx['charts']['deciles'])}
{img(ctx['charts']['importance'])}

<h2>Step 3 · Walk-forward folds</h2>
<p class="note">Each fold trains only on the past, skips {ctx['horizon']} days (purge), then predicts the next block.</p>
<p class="note" style="font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12.5px">{folds_html}</p>

<h2>Steps 1 &amp; 2 · Universe and features</h2>
<p class="note">{html.escape(ctx['universe_text'])}</p>
<p class="note">{len(ctx['features'])} features: {', '.join(ctx['features'])}.</p>

<h2>Important caveats</h2>
<p class="note">Backtests are optimistic. The stock list is today's large companies (survivorship bias),
Yahoo prices are free and occasionally wrong, and the past does not guarantee the future.
Paper trade for several months before considering real money. This is educational software, not financial advice.</p>
</main></body></html>"""
