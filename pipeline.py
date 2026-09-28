"""서울 25개구 저평가·고평가 지수: 수집 → 계산 → latest.json (파일 1개 버전, R-ONE 키 1개만 필요).

환경변수: RONE_API_KEY
산출물(같은 폴더): latest.json, rone_sale.csv, rone_jeonse.csv, rone_tables.json
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import pathlib
import re
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
KST = dt.timezone(dt.timedelta(hours=9))
BASE = "https://www.reb.or.kr/r-one/openapi/"
DEFAULT_SALE_TABLE = "T244183132827305"   # (주) 매매가격지수
START_YEAR = 2012
PSIZE = 1000

GU_NAMES = ["종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구", "성북구", "강북구",
            "도봉구", "노원구", "은평구", "서대문구", "마포구", "양천구", "강서구", "구로구", "금천구",
            "영등포구", "동작구", "관악구", "서초구", "강남구", "송파구", "강동구"]


def short(n: str) -> str:
    return n[:-1] if n.endswith("구") and len(n) > 2 else n


def env(name, default=None):
    v = os.environ.get(name)
    return v.strip() if v and v.strip() else default


# ----------------------------------------------------------------- HTTP / R-ONE
def http_get(url, params=None, retries=4, timeout=60):
    if params:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "seoul-valuation/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (401, 403, 404):
                raise
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(2 ** i)
    raise RuntimeError(f"GET 실패: {url.split('KEY=')[0]}... ({last})")


def find_rows(obj):
    if isinstance(obj, dict):
        if isinstance(obj.get("row"), list):
            return [r for r in obj["row"] if isinstance(r, dict)]
        for v in obj.values():
            r = find_rows(v)
            if r:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_rows(v)
            if r:
                return r
    return []


_shown = {"n": 0}


def call(endpoint, key, **params):
    q = {"KEY": key, "Type": "json", "pSize": PSIZE, **params}
    raw = http_get(BASE + endpoint, q)
    try:
        obj = json.loads(raw.decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"R-ONE 응답 해석 실패: {raw[:300]!r}") from e
    rows = find_rows(obj)
    if not rows and _shown["n"] < 3:           # 진단용: 빈 응답이면 원문 일부를 로그에 남긴다
        _shown["n"] += 1
        print("[진단] 빈 응답:", endpoint, json.dumps(obj, ensure_ascii=False)[:500])
    elif rows and _shown["n"] < 1:
        _shown["n"] += 1
        print("[진단] 응답 필드:", list(rows[0].keys()), "예:", json.dumps(rows[0], ensure_ascii=False)[:300])
    return rows


def list_tables(key):
    out = []
    for p in range(1, 20):
        got = call("SttsApiTbl.do", key, pIndex=p)
        out += got
        if len(got) < PSIZE:
            break
    return out


def discover_tables(key):
    p = ROOT / "rone_tables.json"
    try:
        cfg = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        cfg = {}
    sale = env("RONE_STATBL_SALE") or cfg.get("sale") or DEFAULT_SALE_TABLE
    jeonse = env("RONE_STATBL_JEONSE") or cfg.get("jeonse")
    if not jeonse:
        try:
            for r in list_tables(key):
                nm = str(r.get("STATBL_NM", ""))
                if re.search(r"\(주\).*전세.*가격지수", nm) and "변동" not in nm and "증감" not in nm:
                    jeonse = r.get("STATBL_ID")
                    break
        except Exception as e:  # noqa: BLE001
            print("[경고] 통계표 목록 조회 실패:", e)
    out = {"sale": sale, "jeonse": jeonse}
    p.write_text(json.dumps(out), encoding="utf-8")
    return out


def norm_region(row):
    full = str(row.get("CLS_FULLNM") or row.get("CLS_NM") or "").strip()
    if not full:
        return None
    if row.get("CLS_FULLNM") and "서울" not in full:
        return None                       # 부산 중구, 대구 서구 등 제외
    tokens = [t for t in re.split(r"[>/\s]+", full) if t]
    for t in reversed(tokens):
        if t in GU_NAMES:
            return t
    return None


def week_to_date(w):
    try:
        return dt.date.fromisocalendar(int(w[:4]), int(w[4:6]), 1)
    except Exception:  # noqa: BLE001
        return None


def fetch_year(key, statbl, year):
    out, page = [], 1
    while True:
        rows = call("SttsApiTblData.do", key, STATBL_ID=statbl, DTACYCLE_CD="WK",
                    START_WRTTIME=f"{year}01", END_WRTTIME=f"{year}53", pIndex=page)
        items = {str(r.get("ITM_NM", "")) for r in rows if r.get("ITM_NM")}
        for r in rows:
            if len(items) > 1 and "지수" not in str(r.get("ITM_NM", "")):
                continue
            gu = norm_region(r)
            wk = str(r.get("WRTTIME_IDTFR_ID", ""))
            try:
                val = float(str(r.get("DTA_VAL", "")).replace(",", ""))
            except ValueError:
                continue
            if gu and re.fullmatch(r"\d{6}", wk):
                out.append((wk, gu, val))
        if len(rows) < PSIZE or page > 60:
            break
        page += 1
    return out


def pivot(df):
    df = df.drop_duplicates(["week", "gu"], keep="last").copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["date"] = pd.to_datetime(df["week"].map(week_to_date))
    df = df.dropna(subset=["date", "value"])
    return df.pivot(index="date", columns="gu", values="value").sort_index()


def load_cached(kind):
    p = ROOT / f"rone_{kind}.csv"
    return pivot(pd.read_csv(p, dtype={"week": str})) if p.exists() else None


def update(key, kind, statbl):
    p = ROOT / f"rone_{kind}.csv"
    old = pd.read_csv(p, dtype={"week": str}) if p.exists() else pd.DataFrame(columns=["week", "gu", "value"])
    start = max(START_YEAR, int(str(old["week"].max())[:4])) if len(old) else START_YEAR
    new = []
    for y in range(start, dt.date.today().year + 1):
        new += fetch_year(key, statbl, y)
        print(f"  {kind} {y}: 누적 신규 {len(new)}행")
    merged = pd.concat([old, pd.DataFrame(new, columns=["week", "gu", "value"])], ignore_index=True)
    merged = merged.drop_duplicates(["week", "gu"], keep="last").sort_values(["week", "gu"])
    if merged.empty:
        raise RuntimeError(f"{kind}: 수신 데이터 없음(통계표 {statbl}). 위 [진단] 로그를 확인하세요.")
    merged.to_csv(p, index=False)
    return pivot(merged)


# ----------------------------------------------------------------- 지수 계산
WIN, MIN_HIST, FLAT = 520, 156, 0.3
WEIGHTS = {"sale_jeonse": 15, "trend_dev": 20, "drawdown": 15, "relative": 20}
LABEL = {"jeonse_ratio": "전세가율", "yield_spread": "임대수익률-금리", "sale_jeonse": "매매/전세 지수비",
         "trend_dev": "장기추세 괴리", "drawdown": "고점 대비 위치", "relative": "구 간 상대가치"}
ALL_KEYS = ["jeonse_ratio", "yield_spread", "sale_jeonse", "trend_dev", "drawdown", "relative"]


def own_pct(df):
    def f(x):
        last = x[-1]
        if np.isnan(last):
            return np.nan
        v = x[~np.isnan(x)]
        return float((v <= last).mean() * 100) if len(v) else np.nan
    return df.rolling(WIN, min_periods=MIN_HIST).apply(f, raw=True)


def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else round(x, 2)


def quadrant(v, arrow):
    if v < 35:
        return {"up": "저평가·반등 조짐", "flat": "저평가·횡보", "down": "저평가·하락 지속(관망)"}[arrow]
    if v > 65:
        return {"up": "고평가·과열", "flat": "고평가·횡보", "down": "고평가·조정 중"}[arrow]
    return {"up": "중립·상승", "flat": "중립·횡보", "down": "중립·하락"}[arrow]


def compute(sale, jeonse=None, history_weeks=104):
    sale = sale.reindex(columns=GU_NAMES).dropna(how="all").ffill(limit=2)
    weeks = sale.index
    lnS = np.log(sale)
    comp = {}
    trend = lnS - lnS.rolling(260, min_periods=MIN_HIST).mean()
    dd = lnS - lnS.rolling(WIN, min_periods=MIN_HIST).max()
    rel = lnS.sub(lnS.mean(axis=1), axis=0)
    comp["trend_dev"], comp["drawdown"], comp["relative"] = own_pct(trend), own_pct(dd), own_pct(rel)
    lnJ = None
    if jeonse is not None and not jeonse.empty:
        jeonse = jeonse.reindex(index=weeks, columns=GU_NAMES).ffill(limit=2)
        lnJ = np.log(jeonse)
        comp["sale_jeonse"] = own_pct(lnS - lnJ)

    num = pd.DataFrame(0.0, index=weeks, columns=GU_NAMES)
    den = num.copy()
    cnt = pd.DataFrame(0, index=weeks, columns=GU_NAMES)
    for k, df in comp.items():
        ok = df.notna()
        num += df.fillna(0) * WEIGHTS[k]
        den += ok * WEIGHTS[k]
        cnt += ok.astype(int)
    value = (num / den.where(den > 0)).where(cnt >= 3)

    m4, m12 = lnS.diff(4) * 100, lnS.diff(12) * 100
    j12 = lnJ.diff(12) * 100 if lnJ is not None else None
    last = weeks[-1]
    pct = lambda s: s.rank(pct=True) * 100  # noqa: E731
    mom, wsum = 0.5 * pct(m12.loc[last]) + 0.3 * pct(m4.loc[last]), 0.8
    if j12 is not None and j12.loc[last].notna().any():
        mom, wsum = mom + 0.2 * pct(j12.loc[last]), 1.0
    mom = mom / wsum

    gus = []
    for gu in GU_NAMES:
        v = value.at[last, gu]
        if pd.isna(v):
            continue
        m12v = _f(m12.at[last, gu])
        arrow = "flat" if m12v is None or abs(m12v) <= FLAT else ("up" if m12v > 0 else "down")
        hist = value[gu].dropna().iloc[-history_weeks:]
        gus.append({
            "name": short(gu), "full": gu, "value": round(float(v), 1),
            "label": "저평가" if v < 30 else ("고평가" if v > 70 else "보통"),
            "momentum": round(float(mom[gu]), 1), "arrow": arrow, "quadrant": quadrant(float(v), arrow),
            "components": {k: _f(comp[k].at[last, gu]) for k in comp},
            "metrics": {"m4": _f(m4.at[last, gu]), "m12": m12v,
                        "j12": _f(j12.at[last, gu]) if j12 is not None else None,
                        "trend_dev": _f(trend.at[last, gu] * 100), "drawdown": _f(dd.at[last, gu] * 100),
                        "relative": _f(rel.at[last, gu] * 100)},
            "history": [[d.strftime("%Y-%m-%d"), round(float(x), 1)] for d, x in hist.items()],
        })
    return {"asof": last.strftime("%Y-%m-%d"),
            "components_used": [k for k in ALL_KEYS if k in comp],
            "components_missing": [k for k in ALL_KEYS if k not in comp],
            "labels": LABEL, "gu": gus}


# ----------------------------------------------------------------- 실행
def main():
    now = dt.datetime.now(KST)
    key = env("RONE_API_KEY")
    status, warnings = {}, []
    sale = jeonse = None
    try:
        if not key:
            raise RuntimeError("RONE_API_KEY 미설정 (GitHub Secrets 확인)")
        tbl = discover_tables(key)
        print("통계표:", tbl)
        sale = update(key, "sale", tbl["sale"])
        status["rone_sale"] = {"ok": True, "last": sale.index.max().strftime("%Y-%m-%d")}
        if tbl.get("jeonse"):
            try:
                jeonse = update(key, "jeonse", tbl["jeonse"])
                status["rone_jeonse"] = {"ok": True, "last": jeonse.index.max().strftime("%Y-%m-%d")}
            except Exception as e:  # noqa: BLE001
                jeonse = load_cached("jeonse")
                status["rone_jeonse"] = {"ok": False, "error": str(e)[:200]}
                warnings.append("전세지수 수집 실패: 저장된 값을 사용합니다.")
        else:
            status["rone_jeonse"] = {"ok": False, "error": "전세 통계표를 찾지 못함"}
            warnings.append("주간 전세지수 통계표를 찾지 못해 '매매/전세 지수비' 항목을 뺐습니다.")
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        sale, jeonse = load_cached("sale"), load_cached("jeonse")
        status["rone_sale"] = {"ok": False, "error": str(e)[:200],
                               "last": sale.index.max().strftime("%Y-%m-%d") if sale is not None else None}
        warnings.append("부동산원 수집 실패: 저장된 지수로 계산했습니다." if sale is not None else "부동산원 수집 실패, 저장된 데이터 없음.")

    out = ROOT / "latest.json"
    if sale is None or sale.empty:
        try:
            prev = json.loads(out.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            prev = {}
        prev.update({"status": status, "warnings": warnings + ["계산 불가: 이전 결과를 그대로 표시합니다."],
                     "checked_at": now.isoformat(timespec="seconds")})
        out.write_text(json.dumps(prev, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        return 1

    res = compute(sale, jeonse)
    try:
        prev = json.loads(out.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        prev = {}
    fresh = status.get("rone_sale", {}).get("ok", False)
    res.update({"generated_at": now.isoformat(timespec="seconds") if fresh else prev.get("generated_at"),
                "checked_at": now.isoformat(timespec="seconds"), "is_sample": False,
                "status": status, "warnings": warnings})
    out.write_text(json.dumps(res, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"완료: 기준주 {res['asof']}, 구 {len(res['gu'])}개, 사용항목 {res['components_used']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
