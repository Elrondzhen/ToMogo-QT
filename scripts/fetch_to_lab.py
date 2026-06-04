"""
用 akshare 抓取 A 股日线数据，转换为 tomogoqt.alpha 的 AlphaLab 兼容格式。

产出（与 tomogoqt/alpha/lab.py 的 AlphaLab 文件布局一致）：
  {lab_path}/daily/{vt_symbol}.parquet   逐只股票日线
  {lab_path}/component/{index_symbol}    指数成分股（shelve）
  {lab_path}/contract.json               合约费率/乘数/最小变动价

数据源：akshare（公开行情接口，免费、无需账号）。
数值行情全部来自数据接口，保证精确可复现；本脚本只做取数与格式转换。
"""

from __future__ import annotations

import argparse
import json
import shelve
from datetime import datetime, date, timedelta
from pathlib import Path

import pandas as pd
import polars as pl

# akshare / baostock 均为惰性导入：用到哪个数据源才导入哪个，
# 避免只安装其中一个时脚本无法运行。


def to_vt_symbol(code: str) -> str:
    """把纯数字代码转成 tomogoqt 的 vt_symbol（如 600519 -> 600519.SSE）。"""
    code = code.zfill(6)
    # 沪市：60/68(科创)/11(可转债)/51(ETF) 开头；深市：00/30(创业)/12/15 开头
    if code[0] == "6" or code.startswith(("68", "51", "11")):
        return f"{code}.SSE"
    return f"{code}.SZSE"


def _norm_index_code(index_symbol: str) -> str:
    """从 000300.SSE 或 000300 提取 6 位指数代码。"""
    return index_symbol.split(".")[0].zfill(6)


def to_baostock_symbol(vt_symbol: str) -> str:
    """vt_symbol 转 baostock 代码（600519.SSE -> sh.600519）。"""
    code, exchange = vt_symbol.split(".")
    prefix = "sh" if exchange == "SSE" else "sz"
    return f"{prefix}.{code}"


def _fmt_dash(yyyymmdd: str) -> str:
    """YYYYMMDD -> YYYY-MM-DD（baostock 需要带横线的日期）。"""
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"


def fetch_one_daily(
    vt_symbol: str, start: str, end: str, adjust: str = "qfq", source: str = "akshare"
) -> pl.DataFrame | None:
    """抓取单只股票日线，返回 AlphaLab parquet 所需的列。按 source 分发。"""
    if source == "baostock":
        return _fetch_one_daily_baostock(vt_symbol, start, end, adjust)
    return _fetch_one_daily_akshare(vt_symbol.split(".")[0], start, end, adjust)


def _fetch_one_daily_akshare(code: str, start: str, end: str, adjust: str) -> pl.DataFrame | None:
    """akshare 数据源：code 为 6 位纯数字，start/end 为 YYYYMMDD。"""
    import akshare as ak

    raw: pd.DataFrame = ak.stock_zh_a_hist(
        symbol=code, period="daily", start_date=start, end_date=end, adjust=adjust,
    )
    if raw is None or raw.empty:
        return None

    df = pl.from_pandas(raw)
    # akshare 列名为中文；成交量单位为"手"(100股)，换算为股
    return df.select(
        pl.col("日期").cast(pl.Datetime).alias("datetime"),
        pl.col("开盘").cast(pl.Float64).alias("open"),
        pl.col("最高").cast(pl.Float64).alias("high"),
        pl.col("最低").cast(pl.Float64).alias("low"),
        pl.col("收盘").cast(pl.Float64).alias("close"),
        (pl.col("成交量").cast(pl.Float64) * 100).alias("volume"),
        pl.col("成交额").cast(pl.Float64).alias("turnover"),
        pl.lit(0.0).alias("open_interest"),
    ).sort("datetime")


def _fetch_one_daily_baostock(vt_symbol: str, start: str, end: str, adjust: str) -> pl.DataFrame | None:
    """baostock 数据源。需调用方已 bs.login()。volume 单位为股、amount 为元。"""
    import baostock as bs

    # adjust 映射：qfq->2(前复权), hfq->1(后复权), ""->3(不复权)
    adjustflag = {"qfq": "2", "hfq": "1", "": "3"}.get(adjust, "2")
    bs_code = to_baostock_symbol(vt_symbol)

    rs = bs.query_history_k_data_plus(
        bs_code, "date,open,high,low,close,volume,amount",
        start_date=_fmt_dash(start), end_date=_fmt_dash(end),
        frequency="d", adjustflag=adjustflag,
    )
    if rs.error_code != "0":
        raise RuntimeError(f"baostock 查询失败: {rs.error_msg}")

    rows: list[list[str]] = []
    while rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return None

    # 直接用行列表构造 polars（绕开 pandas，避免 pyarrow 依赖）
    cols = ["date", "open", "high", "low", "close", "volume", "amount"]
    df = pl.DataFrame(rows, schema=cols, orient="row")
    # 停牌日 volume/amount 可能为空串，置 0；价格空串置 null
    return df.select(
        pl.col("date").str.to_datetime().alias("datetime"),
        pl.col("open").cast(pl.Float64, strict=False).alias("open"),
        pl.col("high").cast(pl.Float64, strict=False).alias("high"),
        pl.col("low").cast(pl.Float64, strict=False).alias("low"),
        pl.col("close").cast(pl.Float64, strict=False).alias("close"),
        pl.col("volume").cast(pl.Float64, strict=False).fill_null(0.0).alias("volume"),
        pl.col("amount").cast(pl.Float64, strict=False).fill_null(0.0).alias("turnover"),
        pl.lit(0.0).alias("open_interest"),
    ).sort("datetime")


def save_daily(lab_path: Path, vt_symbol: str, df: pl.DataFrame) -> None:
    """写入 {lab_path}/daily/{vt_symbol}.parquet，存在则按 datetime 去重合并。"""
    daily_dir = lab_path / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    file_path = daily_dir / f"{vt_symbol}.parquet"

    if file_path.exists():
        old = pl.read_parquet(file_path)
        df = pl.concat([old, df]).unique(subset=["datetime"]).sort("datetime")

    df.write_parquet(file_path)


def fetch_index_daily(
    index_symbol: str, lab_path: Path, start: str, end: str,
) -> bool:
    """抓取指数日线（如沪深300 000300.SSE）写入 daily/，作市场代理用。

    指数无复权概念，用 adjustflag=3（adjust=""）。写入 {lab_path}/daily/{index_symbol}.parquet
    （存在则按 datetime 去重合并）。指数文件会被 pipeline._list_local_symbols() 排除，
    不会泄漏进训练 universe，仅供 regime/市场代理读取。返回是否成功抓到数据。
    """
    import baostock as bs

    lg = bs.login()
    if lg.error_code != "0":
        return False
    try:
        df = _fetch_one_daily_baostock(index_symbol, start, end, adjust="")
    except Exception:                                       # noqa: BLE001
        df = None
    finally:
        bs.logout()
    if df is None or df.is_empty():
        return False
    save_daily(lab_path, index_symbol, df)
    return True


def fetch_ondemand_batch(
    vt_symbols: list[str], cache_dir: Path, start: str, end: str, adjust: str = "qfq",
) -> set[str]:
    """按需批量抓取若干股票日线到独立缓存目录（池外票评估用，一次登录抓多只）。

    写入 cache_dir/{vt_symbol}.parquet（存在则按 datetime 去重合并），与训练池 daily/ 隔离，
    防止池外票泄漏进训练 universe。返回成功抓到数据的 vt_symbol 集合。
    用 baostock 数据源（与回训一致，前复权口径相同）。
    """
    import baostock as bs

    cache_dir.mkdir(parents=True, exist_ok=True)
    ok: set[str] = set()
    lg = bs.login()
    if lg.error_code != "0":
        return ok
    try:
        for vt in vt_symbols:
            try:
                df = _fetch_one_daily_baostock(vt, start, end, adjust)
            except Exception:                                   # noqa: BLE001
                continue
            if df is None or df.is_empty():
                continue
            fp = cache_dir / f"{vt}.parquet"
            if fp.exists():
                old = pl.read_parquet(fp)
                df = pl.concat([old, df]).unique(subset=["datetime"]).sort("datetime")
            df.write_parquet(fp)
            ok.add(vt)
    finally:
        bs.logout()
    return ok


def from_baostock_symbol(bs_code: str) -> str:
    """baostock 代码转 vt_symbol（sh.600519 -> 600519.SSE）。"""
    prefix, code = bs_code.split(".")
    exchange = "SSE" if prefix == "sh" else "SZSE"
    return f"{code}.{exchange}"


def is_a_share(bs_code: str) -> bool:
    """判定 baostock code 是否为 A 股个股（排除指数/ETF/债券/B股）。

    沪市A股：600/601/603/605 主板 + 688 科创板；
    深市A股：000/001/002/003 主板中小板 + 300/301 创业板。
    """
    try:
        prefix, num = bs_code.split(".")
    except ValueError:
        return False
    if prefix == "sh":
        return num.startswith(("600", "601", "603", "605", "688"))
    if prefix == "sz":
        return num.startswith(("000", "001", "002", "003", "300", "301"))
    return False


def fetch_all_a_share_names(day: str | None = None) -> dict[str, str]:
    """用 baostock query_all_stock 一次性取全 A 股个股 代码->名称。需调用方已 bs.login()。

    一次查询返回全部证券（含指数/ETF/B股），用 is_a_share 过滤出 A 股个股（约 5200 只）。
    day=None 时自动从今天往前找最近一个有数据的交易日（最多回退 7 天）。
    返回 {vt_symbol: name}；失败返回 {}。
    """
    import baostock as bs

    if day:
        candidate_days = [day]
    else:
        candidate_days = [
            (datetime.today() - timedelta(days=b)).strftime("%Y-%m-%d") for b in range(7)
        ]

    for d in candidate_days:
        rs = bs.query_all_stock(day=d)
        if rs.error_code != "0":
            continue
        names: dict[str, str] = {}
        while rs.next():
            row = rs.get_row_data()        # [code, tradeStatus, code_name]
            if len(row) >= 3 and is_a_share(row[0]):
                names[from_baostock_symbol(row[0])] = row[2]
        if names:
            return names
    return {}



def fetch_index_components(index_symbol: str, source: str = "akshare") -> list[str]:
    """抓取指数最新成分股，返回 vt_symbol 列表。按 source 分发。"""
    if source == "baostock":
        return _fetch_index_components_baostock(index_symbol)
    return _fetch_index_components_akshare(_norm_index_code(index_symbol))


def _fetch_index_components_akshare(index_code: str) -> list[str]:
    """akshare 成分股接口。只给"当前"成分，无历史快照。"""
    import akshare as ak

    raw: pd.DataFrame = ak.index_stock_cons(symbol=index_code)
    if raw is None or raw.empty:
        return []

    code_col = next((c for c in raw.columns if "代码" in c), None)
    if code_col is None:
        return []
    return [to_vt_symbol(str(c)) for c in raw[code_col].tolist()]


def _fetch_index_components_baostock(index_symbol: str) -> list[str]:
    """baostock 成分股。仅支持沪深300/中证500/上证50 三个常用指数。需已 login。"""
    import baostock as bs

    code = _norm_index_code(index_symbol)
    query_map = {
        "000300": bs.query_hs300_stocks,    # 沪深300
        "000905": bs.query_zz500_stocks,    # 中证500
        "000016": bs.query_sz50_stocks,     # 上证50
    }
    query_fn = query_map.get(code)
    if query_fn is None:
        raise SystemExit(
            f"baostock 仅支持指数 000300/000905/000016，不支持 {index_symbol}。"
            "可改用 --symbols 指定股票，或用 akshare 数据源。"
        )

    rs = query_fn()
    if rs.error_code != "0":
        raise RuntimeError(f"baostock 成分股查询失败: {rs.error_msg}")

    vt_symbols: list[str] = []
    while rs.next():
        row = rs.get_row_data()
        # 行格式: [updateDate, code, weight]，code 形如 sh.600519
        bs_code = row[1] if len(row) > 1 else ""
        if bs_code:
            vt_symbols.append(from_baostock_symbol(bs_code))
    return vt_symbols


def save_components(lab_path: Path, index_symbol: str, vt_symbols: list[str]) -> None:
    """把成分股写入 shelve，与 AlphaLab.save_component_data 格式一致。

    key 为日期字符串，value 为该日成分 vt_symbol 列表。
    用今天作为快照日期。
    """
    comp_dir = lab_path / "component"
    comp_dir.mkdir(parents=True, exist_ok=True)
    file_path = comp_dir / index_symbol

    today = date.today().strftime("%Y-%m-%d")
    with shelve.open(str(file_path)) as db:
        db[today] = vt_symbols


def save_contract_settings(lab_path: Path, vt_symbols: list[str]) -> None:
    """写入合约配置 contract.json，与 AlphaLab.add_contract_setting 格式一致。

    费率参考下载示例：买 5bp / 卖 10bp，size=1，pricetick=0.0001。
    """
    file_path = lab_path / "contract.json"
    contracts: dict = {}
    if file_path.exists():
        with open(file_path, encoding="UTF-8") as f:
            contracts = json.load(f)

    for vt_symbol in vt_symbols:
        contracts[vt_symbol] = {
            "long_rate": 5 / 10000,
            "short_rate": 10 / 10000,
            "size": 1,
            "pricetick": 0.0001,
        }

    with open(file_path, mode="w+", encoding="UTF-8") as f:
        json.dump(contracts, f, indent=4, ensure_ascii=False)


def run(
    lab_path: str,
    index_symbol: str | None,
    symbols: list[str] | None,
    start: str,
    end: str,
    adjust: str,
    source: str = "akshare",
) -> None:
    """编排：确定股票池 -> 逐只取数写盘 -> 写成分股与合约配置。"""
    lab = Path(lab_path)
    lab.mkdir(parents=True, exist_ok=True)

    # baostock 需要先登录会话（整个流程共用一次登录）
    bs = None
    if source == "baostock":
        import baostock as bs
        lg = bs.login()
        if lg.error_code != "0":
            raise SystemExit(f"baostock 登录失败: {lg.error_msg}")
        print(f"      baostock 登录成功（数据源={source}）")

    try:
        # 1. 确定股票池
        vt_symbols: list[str] = []
        if index_symbol:
            print(f"[1/3] 抓取指数 {index_symbol} 成分股...")
            vt_symbols = fetch_index_components(index_symbol, source)
            print(f"      获取到 {len(vt_symbols)} 只成分股")
            if vt_symbols:
                save_components(lab, index_symbol, vt_symbols)
                vt_symbols_with_index = vt_symbols + [index_symbol]
            else:
                vt_symbols_with_index = []
        elif symbols:
            vt_symbols = [s if "." in s else to_vt_symbol(s) for s in symbols]
            vt_symbols_with_index = vt_symbols
            print(f"[1/3] 使用指定股票池：{len(vt_symbols)} 只")
        else:
            raise SystemExit("必须指定 --index 或 --symbols 之一")

        # 2. 逐只取数写盘
        print(f"[2/3] 抓取日线 {start} ~ {end} (adjust={adjust}, source={source})...")
        ok, fail = 0, []
        total = len(vt_symbols_with_index)
        for i, vt_symbol in enumerate(vt_symbols_with_index, 1):
            try:
                df = fetch_one_daily(vt_symbol, start, end, adjust, source)
                if df is not None and not df.is_empty():
                    save_daily(lab, vt_symbol, df)
                    ok += 1
                else:
                    fail.append(vt_symbol)
            except Exception as e:                              # noqa: BLE001
                fail.append(f"{vt_symbol}({e})")
            if i % 20 == 0 or i == total:
                print(f"      进度 {i}/{total}，成功 {ok}，失败 {len(fail)}")

        # 3. 写合约配置
        print("[3/3] 写入合约配置 contract.json...")
        save_contract_settings(lab, vt_symbols)

        print(f"\n完成。成功 {ok} 只，失败 {len(fail)} 只。数据目录：{lab.resolve()}")
        if fail:
            print(f"失败列表（前20）：{fail[:20]}")
    finally:
        if bs is not None:
            bs.logout()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="用 akshare/baostock 抓取 A 股日线到 AlphaLab 格式")
    p.add_argument("--lab-path", required=True, help="lab 数据目录，如 ./lab/csi300")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--index", dest="index_symbol", help="指数 vt_symbol，如 000300.SSE")
    g.add_argument("--symbols", nargs="+", help="指定股票代码列表，如 600519 000001")
    p.add_argument("--start", default="20200101", help="起始日 YYYYMMDD")
    p.add_argument(
        "--end",
        default=datetime.today().strftime("%Y%m%d"),
        help="结束日 YYYYMMDD，默认今天（确保最新）",
    )
    p.add_argument("--adjust", default="qfq", choices=["qfq", "hfq", ""], help="复权方式")
    p.add_argument(
        "--source", default="akshare", choices=["akshare", "baostock"],
        help="数据源。akshare(默认)依赖东方财富；baostock 用自有服务器，网络受限环境更稳",
    )
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(
        lab_path=args.lab_path,
        index_symbol=args.index_symbol,
        symbols=args.symbols,
        start=args.start,
        end=args.end,
        adjust=args.adjust,
        source=args.source,
    )



