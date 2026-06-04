"""
ToMogo-QT 量化推荐流程封装。

封装：抓最新行情 -> 计算 Alpha158 因子 -> 训练 LightGBM -> 生成信号 -> 推荐 Top-N。
以已验证的 demo_alpha_run.py 为蓝本。

研究演示用途，非实盘投资建议。
持有天数为策略轮动规则推导的启发式估计，非模型预测。
"""

import sys
import math
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

# 让本地 tomogoqt 与 scripts 可导入
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "scripts"))

from tomogoqt.trader.constant import Interval                                   # noqa: E402
from tomogoqt.alpha import AlphaLab, Segment                                    # noqa: E402
from tomogoqt.alpha.dataset import process_drop_na, process_cs_norm            # noqa: E402
from tomogoqt.alpha.dataset.datasets.alpha_158 import Alpha158                  # noqa: E402
from tomogoqt.alpha.dataset.utility import calculate_by_expression             # noqa: E402
from tomogoqt.alpha.model.models.lgb_model import LgbModel                      # noqa: E402

LAB_PATH = str(BASE_DIR / "lab" / "demo")
INDEX_SYMBOL = "000300.SSE"
ARTIFACT_NAME = "tomogo"                       # z-score 排名模型/信号：Top3 与持仓评估排名专用（零回归）
HORIZON_SIGNAL = "tomogo_horizons"             # 多视野原始收益预测宽表信号：datetime,vt_symbol,ret_h2..ret_h20
NAMES_PATH = BASE_DIR / "lab" / "demo" / "names.json"   # 股票名称本地缓存(代码→中文名)，去除每次实时依赖
ONDEMAND_DIR = Path(LAB_PATH) / "daily_ondemand"        # 池外票按需抓取的日线缓存（独立于 daily/，防止泄漏进训练池）

# 策略参数（用于持有天数启发式 + 一致性）
TOP_K = 20
N_DROP = 2
MIN_DAYS = 3

# 多视野持有天数模型参数
HORIZONS = [2, 5, 10, 20]                      # 预测视野（交易日），各训一个原始收益 LightGBM
HOLD_SWITCH_MARGIN = 0.10                      # 选窗滞后阈值：更长窗口日均收益需超当前最优 10% 才切换，抑制 argmax 抖动
ROTATION_DAYS = math.ceil(TOP_K / N_DROP)      # =10，策略轮动周期，用于"模型最优窗口越界"标注
_RISK_DENOM = "linear"                         # 跨视野可比口径："linear"=日均收益 ret/h（严惩长窗口，默认）；"sqrt"=Sharpe 口径 ret/√h


def _label_expr(h: int) -> str:
    """视野 h 的原始前向收益标签：t+1 买入、t+(h+1) 卖出，持有 h 个交易日的累计收益。"""
    return f"ts_delay(close, -{h + 1}) / ts_delay(close, -1) - 1"

# 多因子建议出手价参数（替代"现价×固定倍数"的粗糙口径）
ATR_WINDOW = 14         # ATR 计算窗口（真实波幅，价格单位）
VOL_WINDOW = 20         # 日收益波动率窗口
MAX_DEVIATION = 0.095   # 出手价相对现价的最大偏离（贴近 A 股 10% 涨跌停，留 0.5% 余量）
TICK = 0.01             # A 股最小报价单位


def holding_days_range(top_k: int = TOP_K, n_drop: int = N_DROP, min_days: int = MIN_DAYS) -> tuple[int, int]:
    """建议持有天数上下限（启发式，作为多视野模型缺信号时的兜底）。

    下限 = 策略最短持有期 min_days；
    上限 = 轮动周期 ceil(top_k / n_drop)，即一只票从入选到被全部轮换掉的平均天数。
    """
    lower = min_days
    upper = math.ceil(top_k / n_drop)
    return lower, max(upper, lower)


def suggest_hold_days(rets: dict[int, float]) -> dict:
    """根据多视野原始预测收益推算建议持有天数（真·模型推测，替代固定启发式）。

    rets: {2: ret2, 5: ret5, 10: ret10, 20: ret20}，各视野的原始预测累计收益（持有 h 个交易日）。
    口径：
      1. 日均收益 daily_h = ret_h / h（_RISK_DENOM='linear'，严惩长窗口，默认）；
         或 ret_h/√h（'sqrt'，Sharpe 口径，偏爱长窗口，备选）。
      2. 选窗带滞后：从最短窗口起，仅当更长窗口日均收益"显著更优"（超当前最优 HOLD_SWITCH_MARGIN 的幅度）
         才切换，抑制 4 个独立模型噪声导致的 argmax 逐日抖动。
      3. 区间 = best_h 与相邻窗口（best=10 → 5~20；best=2 → 2~5；best=20 → 10~20）。
      4. 模型全部看跌（4 视野预测均≤0）→ lower=upper=最短窗口 + bearish 标注。
      5. best_h > ROTATION_DAYS → over_rotation 标注（最优窗口超出策略轮动周期，与策略节奏不一致）。
    返回 {lower, upper, best, over_rotation, bearish, basis}。
    """
    # 仅保留有效预测（非 None / 非 NaN）的视野，按窗口升序
    valid: dict[int, float] = {}
    for h, r in rets.items():
        if r is None:
            continue
        try:
            rf = float(r)
        except (TypeError, ValueError):
            continue
        if rf != rf:                                          # NaN
            continue
        valid[int(h)] = rf

    if not valid:
        return {"lower": None, "upper": None, "best": None,
                "over_rotation": False, "bearish": False,
                "basis": "无有效视野预测，无法估计持有天数"}

    horizons_sorted = sorted(valid.keys())
    shortest = horizons_sorted[0]

    # 退化场景：模型全面看跌 → 尽快离场
    if all(v <= 0 for v in valid.values()):
        return {"lower": shortest, "upper": shortest, "best": shortest,
                "over_rotation": False, "bearish": True,
                "basis": f"各视野预测累计收益均为负（模型看跌），建议尽快离场（最短 {shortest} 天）"}

    # 日均/风险调整收益（跨视野可比口径）
    def _daily(h: int, r: float) -> float:
        if _RISK_DENOM == "sqrt":
            return r / (h ** 0.5)
        return r / h

    daily = {h: _daily(h, valid[h]) for h in horizons_sorted}

    # 选窗带滞后：仅当更长窗口日均收益超当前最优 HOLD_SWITCH_MARGIN 幅度才切换。
    # 用 margin*|daily_best| 的符号安全写法：正值时等价于计划的 ×(1+margin)，负值混入时也能正确判定"显著更优"。
    best_h = horizons_sorted[0]
    for h in horizons_sorted[1:]:
        margin = HOLD_SWITCH_MARGIN * abs(daily[best_h])
        if daily[h] - daily[best_h] > margin:
            best_h = h

    idx = horizons_sorted.index(best_h)
    lower = horizons_sorted[idx - 1] if idx > 0 else best_h
    upper = horizons_sorted[idx + 1] if idx < len(horizons_sorted) - 1 else best_h
    over_rotation = best_h > ROTATION_DAYS

    denom_txt = "日均收益(ret/h)" if _RISK_DENOM != "sqrt" else "风险调整收益(ret/√h)"
    basis = (f"按{denom_txt}选最优持有窗口 = {best_h} 天"
             f"（{best_h}天累计预测 {valid[best_h] * 100:+.2f}%，日均 {daily[best_h] * 100:+.3f}%）；"
             f"建议区间 {lower}~{upper} 天")
    if over_rotation:
        basis += f"；注意：超出策略轮动周期(~{ROTATION_DAYS}天)，与策略节奏不一致"

    return {"lower": lower, "upper": upper, "best": best_h,
            "over_rotation": over_rotation, "bearish": False,
            "basis": basis}


def _list_local_symbols() -> list[str]:
    """本地已有日线数据的 vt_symbol 列表（排除指数本身）。

    只认训练池目录 daily/，不含 daily_ondemand/，确保按需抓取的池外票
    永不泄漏进训练 universe。
    """
    daily_dir = Path(LAB_PATH) / "daily"
    syms = [p.stem for p in daily_dir.glob("*.parquet")]
    return [s for s in syms if s != INDEX_SYMBOL]


def _bar_path(vt_symbol: str) -> Path | None:
    """查找某票的日线 parquet：优先训练池 daily/，回退按需缓存 daily_ondemand/。

    供现价/波动统计等读取逻辑统一寻址，使池外票也能取到现价与 ATR。返回 None 表示两处都没有。
    """
    p = Path(LAB_PATH) / "daily" / f"{vt_symbol}.parquet"
    if p.exists():
        return p
    p2 = ONDEMAND_DIR / f"{vt_symbol}.parquet"
    return p2 if p2.exists() else None


def split_periods(df: pl.DataFrame, full: bool = False) -> tuple:
    """切分 train/valid/test 时间段。

    full=False（样本外回测模式）：60/20/20 严格按时间切分，test 是模型未见过的纯样本外数据。
    full=True（全量训练模式）：train 覆盖全部有标签数据（含近期行情），valid 用最近一段切片
        仅供 LightGBM 早停（会与 train 重叠），test 取最近窗口以便对最新交易日出信号。
        代价：失去诚实的样本外回测能力——此时模型已见过近乎全部数据，回测会虚高。
    """
    dts = df["datetime"].unique().sort()
    n = len(dts)
    fmt = "%Y-%m-%d"

    if full:
        # train 覆盖全部；valid/test 取最近 15% 切片（valid 仅用于早停，与 train 重叠）
        i = int(n * 0.85)
        train = (dts[0].strftime(fmt), dts[n - 1].strftime(fmt))
        valid = (dts[i].strftime(fmt), dts[n - 1].strftime(fmt))
        test = (dts[i].strftime(fmt), dts[n - 1].strftime(fmt))
        return train, valid, test

    i1, i2 = int(n * 0.6), int(n * 0.8)
    train = (dts[0].strftime(fmt), dts[i1 - 1].strftime(fmt))
    valid = (dts[i1].strftime(fmt), dts[i2 - 1].strftime(fmt))
    test = (dts[i2].strftime(fmt), dts[n - 1].strftime(fmt))
    return train, valid, test


def get_current_price(vt_symbol: str) -> float | None:
    """从本地 parquet 末行取现价（qfq 末值=真实现价）。训练池/按需缓存皆可。"""
    fp = _bar_path(vt_symbol)
    if fp is None:
        return None
    df = pl.read_parquet(fp).sort("datetime")
    if df.is_empty():
        return None
    return round(float(df["close"][-1]), 4)


def _round_tick(x: float) -> float:
    """对齐 A 股最小报价单位（0.01 元）。"""
    return round(round(x / TICK) * TICK, 2)


def _calc_stats(vt_symbol: str) -> dict | None:
    """从日线算波动统计：ATR(真实波幅)、日收益波动率、近 20 日高/低。

    用于把"出手价"建立在个股真实波动之上，而非对所有股票一刀切乘固定倍数。
    返回 None 表示数据不足（无法稳健推算）。
    """
    fp = _bar_path(vt_symbol)
    if fp is None:
        return None
    df = pl.read_parquet(fp).sort("datetime")
    if df.height < 2:
        return None

    # 真实波幅 TR = max(high-low, |high-prev_close|, |low-prev_close|)，取 ATR_WINDOW 均值
    win = df.tail(ATR_WINDOW + 1)
    highs = win["high"].to_list()
    lows = win["low"].to_list()
    closes = win["close"].to_list()
    trs = []
    for i in range(1, len(closes)):
        pc = closes[i - 1]
        trs.append(max(highs[i] - lows[i], abs(highs[i] - pc), abs(lows[i] - pc)))
    atr = sum(trs) / len(trs) if trs else None

    # 日收益波动率 σ（近 VOL_WINDOW 日），用于展示与持有期投影
    vwin = df.tail(VOL_WINDOW + 1)["close"].to_list()
    rets = [vwin[i] / vwin[i - 1] - 1 for i in range(1, len(vwin)) if vwin[i - 1]]
    if len(rets) >= 2:
        mean = sum(rets) / len(rets)
        sigma = (sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)) ** 0.5
    else:
        sigma = None

    band = df.tail(VOL_WINDOW)
    return {
        "atr": atr,
        "sigma": sigma,
        "high_20": float(band["high"].max()),
        "low_20": float(band["low"].min()),
    }


def _suggest_order_price(advice: str, price: float, stats: dict | None, urgent: bool = False) -> tuple[float | None, str | None, str | None]:
    """按"波动率 + 排名紧迫度 + 支撑/阻力"多因子推算建议出手价。

    思路（非固定倍数）：
      - 偏移幅度 = k × ATR：个股真实波幅越大，挂单越远，自适应波动；k 随紧迫度变化。
      - 方向/紧迫度由建议决定：
          减仓(urgent) = 末段紧迫离场，挂【低于现价】可成交卖单(−0.5×ATR)，牺牲少量价格换确定性成交；
          减仓(普通)   = 从容减仓，挂【高于现价】卖单(+0.5×ATR)，逢强卖出博取更优价；
          加仓        = 挂【略高于现价】可成交买单(+0.3×ATR)，适度溢价确保买进而不过度追高。
      - 钳制：现价上方挂单(买/从容减仓)以涨停带封顶；紧迫减仓以近 20 日低点(支撑)与跌停带托底，
              避免挂出非理性价位。
    返回 (order_price, order_side, basis_text)；持仓/无统计时返回 (None, None, None)。
    """
    if price is None or stats is None or stats.get("atr") is None:
        return None, None, None

    atr = stats["atr"]
    lo = stats["low_20"]
    lo_bound = price * (1 - MAX_DEVIATION)
    hi_bound = price * (1 + MAX_DEVIATION)
    atr_pct = atr / price * 100 if price else 0

    if advice == "减仓" and urgent:
        raw = price - 0.5 * atr                       # 低于现价：可成交卖单，优先确定性离场
        op = _round_tick(max(raw, lo, lo_bound))
        basis = (f"末段紧迫离场：现价 −0.5×ATR(¥{atr:.2f}/{atr_pct:.1f}%)，"
                 f"挂低于现价的可成交卖单优先保成交，下探不破近20日低 ¥{lo:.2f} 与跌停带")
        return op, "卖出", basis
    if advice == "减仓":
        raw = price + 0.5 * atr                       # 高于现价：逢强卖出，博取更优价
        op = _round_tick(min(raw, hi_bound))
        basis = (f"从容减仓：现价 +0.5×ATR(¥{atr:.2f}/{atr_pct:.1f}%)，"
                 f"挂高于现价卖单逢强卖出，上探不破涨停带")
        return op, "卖出", basis
    if advice == "加仓":
        raw = price + 0.3 * atr                       # 略高于现价：适度溢价确保买进
        op = _round_tick(min(raw, hi_bound))
        basis = (f"加仓：现价 +0.3×ATR(¥{atr:.2f}/{atr_pct:.1f}%)，"
                 f"挂略高于现价的可成交买单确保买进而不过度追高，上探不破涨停带")
        return op, "买入", basis
    # 持仓：不挂单
    return None, None, None


def _load_names() -> dict[str, str]:
    """读取本地股票名称缓存(代码→中文名)。文件缺失或损坏时返回空字典。"""
    import json
    if not NAMES_PATH.exists():
        return {}
    try:
        with open(NAMES_PATH, encoding="UTF-8") as f:
            data = json.load(f)
        return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}
    except (ValueError, OSError):
        return {}


def resolve_symbol(raw: str) -> dict:
    """把用户输入的代码归一化为 vt_symbol 并查本地名称缓存（检索层，供前端校验/补全）。

    接受 "600519" / "600519.SSE" / " sh600519 " 等形式，统一成 vt_symbol。
    返回 {ok, vt_symbol, name, known}：
      known=True 表示在全A股名称索引中找到（合法代码）；False 表示缓存里没有
      （可能是非法代码，也可能是名称索引尚未抓取——此时 vt_symbol 仍照常返回）。
    """
    from fetch_to_lab import to_vt_symbol

    s = str(raw or "").strip().upper()
    if not s:
        return {"ok": False, "vt_symbol": None, "name": None, "known": False}
    # 容错：去掉 baostock/行情软件风格前缀（SH./SZ./SH/SZ）只留数字，再统一走 to_vt_symbol
    if s.startswith(("SH.", "SZ.")):
        s = s[3:]
    elif (s.startswith(("SH", "SZ")) and s[2:].isdigit()):
        s = s[2:]
    if "." in s:
        vt = s
    else:
        vt = to_vt_symbol(s)
    names = _load_names()
    name = names.get(vt)
    return {"ok": True, "vt_symbol": vt, "name": name or vt, "known": name is not None}



def _save_names(names: dict[str, str]) -> None:
    """写入本地股票名称缓存。"""
    import json
    NAMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NAMES_PATH, mode="w", encoding="UTF-8") as f:
        json.dump(names, f, ensure_ascii=False, indent=2, sort_keys=True)


def refresh_names(vt_symbols: list[str] | None = None) -> dict[str, str]:
    """用 baostock 一次性拉取全 A 股名称并合并进本地缓存，返回最新全量缓存。

    在回训时调用一次（此时 baostock 会话本就需要打开），把全市场 代码->名称 固化到本地，使：
      1. /api/recommend 不再每次实时依赖 baostock，避免限流时名称退化成代码；
      2. 持仓评估可检索/校验任意 A 股代码（检索层全覆盖）。
    用 query_all_stock 一次取全市场（~5200 只），远优于逐只 query_stock_basic。
    全量失败时回退对传入的 vt_symbols 逐只补名（至少保证训练池名称新鲜）。
    """
    import baostock as bs
    from fetch_to_lab import fetch_all_a_share_names, to_baostock_symbol

    names = _load_names()
    lg = bs.login()
    if lg.error_code != "0":
        return names
    try:
        all_names = fetch_all_a_share_names()
        if all_names:
            names.update(all_names)                     # 全市场一次性合并
        elif vt_symbols:
            # 全量失败兜底：逐只补训练池名称
            for vt in vt_symbols:
                rs = bs.query_stock_basic(code=to_baostock_symbol(vt))
                if rs.error_code == "0" and rs.next():
                    row = rs.get_row_data()
                    if len(row) > 1 and row[1]:
                        names[vt] = row[1]
    finally:
        bs.logout()
    _save_names(names)
    return names


def fetch_valuation_and_names(vt_symbols: list[str]) -> dict[str, dict]:
    """用 baostock 取每只票的最新估值(PE/PB/PS)与名称。需调用方处理异常。

    名称优先用本地缓存，baostock 取到新名称时顺带回写缓存。
    """
    import baostock as bs
    from fetch_to_lab import to_baostock_symbol

    name_cache = _load_names()
    result: dict[str, dict] = {}
    lg = bs.login()
    if lg.error_code != "0":
        # 登录失败：估值留空，但名称仍可用本地缓存
        return {vt: {"pe": None, "pb": None, "ps": None, "name": name_cache.get(vt, vt)}
                for vt in vt_symbols}
    cache_dirty = False
    try:
        end = datetime.today().strftime("%Y-%m-%d")
        start = (datetime.today() - timedelta(days=15)).strftime("%Y-%m-%d")
        for vt in vt_symbols:
            bs_code = to_baostock_symbol(vt)
            info: dict = {"pe": None, "pb": None, "ps": None, "name": name_cache.get(vt, vt)}
            # 估值
            rs = bs.query_history_k_data_plus(
                bs_code, "date,close,peTTM,pbMRQ,psTTM",
                start_date=start, end_date=end, frequency="d", adjustflag="2",
            )
            rows = []
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())
            if rows:
                last = rows[-1]
                info["pe"] = _to_float(last[2])
                info["pb"] = _to_float(last[3])
                info["ps"] = _to_float(last[4])
            # 名称（取到则回写缓存）
            rs2 = bs.query_stock_basic(code=bs_code)
            if rs2.error_code == "0" and rs2.next():
                row = rs2.get_row_data()
                if len(row) > 1 and row[1]:
                    info["name"] = row[1]
                    if name_cache.get(vt) != row[1]:
                        name_cache[vt] = row[1]
                        cache_dirty = True
            result[vt] = info
    finally:
        bs.logout()
    if cache_dirty:
        _save_names(name_cache)
    return result


def _to_float(s: str) -> float | None:
    try:
        v = float(s)
        return round(v, 2)
    except (ValueError, TypeError):
        return None


def run_retrain(progress=None, fetch: bool = True, full: bool = True) -> dict:
    """完整回训流程：抓最新行情 -> 因子 -> 训练 -> 生成并保存信号。

    progress: 可选回调 progress(stage: str, msg: str)，用于上报进度。
    fetch: 是否先抓最新数据（False 则用本地已有数据直接重训）。
    full: True=全量训练模式（默认，训练覆盖全部历史，模型更强但失去样本外回测能力）；
          False=60/20/20 样本外回测模式。
    返回 {"ok": bool, "last_dt": str, "n_symbols": int, ...}
    """
    def report(stage: str, msg: str) -> None:
        if progress:
            progress(stage, msg)

    lab = AlphaLab(LAB_PATH)

    # 1. 抓最新行情（增量去重合并）
    if fetch:
        report("fetching", "正在用 baostock 增量抓取最新行情...")
        try:
            from fetch_to_lab import run as fetch_run
            symbols = _list_local_symbols()
            # 增量：只抓最近一段时间补空缺（save_daily 会按日期去重合并）。
            # 不重抓全部历史，否则量大且易触发 baostock 会话限流。
            fetch_start = (datetime.today() - timedelta(days=30)).strftime("%Y%m%d")
            fetch_run(
                lab_path=LAB_PATH, index_symbol=None, symbols=symbols,
                start=fetch_start, end=datetime.today().strftime("%Y%m%d"),
                adjust="qfq", source="baostock",
            )
        except Exception as e:                                          # noqa: BLE001
            report("error", f"抓取行情出错（将用已有数据继续）：{e}")

        # 补抓沪深300指数日线作市场代理（regime 计算用）。指数单独存 daily/，
        # 被 _list_local_symbols() 排除，不进训练 universe；全量去重合并（仅一只，开销小）。
        try:
            from fetch_to_lab import fetch_index_daily
            fetch_index_daily(INDEX_SYMBOL, Path(LAB_PATH),
                              start="20180101", end=datetime.today().strftime("%Y%m%d"))
        except Exception as e:                                          # noqa: BLE001
            report("computing", f"指数抓取失败（不影响训练）：{e}")

        # 固化股票名称到本地缓存，使推荐接口不再每次实时依赖 baostock
        try:
            refresh_names(_list_local_symbols())
        except Exception as e:                                          # noqa: BLE001
            report("computing", f"名称缓存更新失败（不影响训练）：{e}")

    # 2. 加载行情
    report("computing", "加载行情数据并计算 Alpha158 因子...")
    vt_symbols = _list_local_symbols()
    df = lab.load_bar_df(vt_symbols, Interval.DAILY, "2018-01-01",
                         datetime.today().strftime("%Y-%m-%d"), extended_days=100)

    # 3. 因子计算
    mode_txt = "全量训练模式（模型更强，但无样本外回测）" if full else "样本外回测模式（60/20/20）"
    report("computing", f"加载行情并计算 Alpha158 因子... [{mode_txt}]")
    train_p, valid_p, test_p = split_periods(df, full=full)
    dataset = Alpha158(df, train_period=train_p, valid_period=valid_p, test_period=test_p)
    dataset.add_processor("learn", lambda df, names=["label"]: process_drop_na(df, names=names))
    dataset.add_processor("learn", lambda df: process_cs_norm(df, names=["label"], method="zscore"))
    dataset.prepare_data(filters=None, max_workers=4)
    dataset.process_data()

    # 4. 训练排名模型（z-score，不变，Top3/评估排名专用，零回归）
    report("training", "训练模型 1/5：排名模型（z-score）...")
    model = LgbModel(seed=42)
    model.fit(dataset)
    lab.save_model(ARTIFACT_NAME, model)

    # 5. 生成排名信号（用 TEST 段，含最新交易日）
    pre = model.predict(dataset, Segment.TEST)
    df_t = dataset.fetch_infer(Segment.TEST).sort(["datetime", "vt_symbol"])
    df_t = df_t.with_columns(pl.Series(pre).alias("signal"))
    signal = df_t.select(["datetime", "vt_symbol", "signal"])
    lab.save_signal(ARTIFACT_NAME, signal)

    # 6. 训练多视野持有天数模型（原始收益，不做 z-score，跨窗口可比）。
    #    复用 dataset.raw_df 已算好的 158 特征（特征计算是耗时大头），仅替换 label 列 → 不重算特征。
    feature_base = dataset.raw_df.drop("label")          # [datetime, vt_symbol, <158 特征>]，raw_df 始终无伤
    horizon_cols: list[pl.DataFrame] = []
    for i, h in enumerate(HORIZONS, 1):
        report("training", f"训练模型 {i + 1}/5：{h} 天视野收益模型...")
        # 现算视野 h 的原始前向收益标签 → [datetime, vt_symbol, label]
        label_df = (
            calculate_by_expression(df, _label_expr(h))
            .rename({"data": "label"})
        )
        with_label = feature_base.join(label_df, on=["datetime", "vt_symbol"], how="left")
        # 替换 dataset 的 learn/infer 视图：learn 仅 drop_na（保留原始收益单位，不做 cs_norm）；
        # infer 末列为 label，predict 会自动丢弃，只用特征。
        dataset.infer_df = with_label.sort(["datetime", "vt_symbol"])
        dataset.learn_df = process_drop_na(dataset.infer_df, names=["label"])

        hmodel = LgbModel(seed=42)
        hmodel.fit(dataset)
        lab.save_model(f"{ARTIFACT_NAME}_h{h}", hmodel)

        pred = hmodel.predict(dataset, Segment.TEST)
        ht = dataset.fetch_infer(Segment.TEST).sort(["datetime", "vt_symbol"])
        ht = ht.with_columns(pl.Series(pred).alias(f"ret_h{h}"))
        horizon_cols.append(ht.select(["datetime", "vt_symbol", f"ret_h{h}"]))

    # 7. 汇总 4 个视野预测为宽表信号（各视野同源同 TEST 段，键完全一致，left 即可）
    wide = horizon_cols[0]
    for col in horizon_cols[1:]:
        wide = wide.join(col, on=["datetime", "vt_symbol"], how="left")
    lab.save_signal(HORIZON_SIGNAL, wide.sort(["datetime", "vt_symbol"]))

    # 8. 训练 GNN 波动率风险叠加层（含持续性基线诚实对比）。失败不影响主流程（排名信号已存）。
    gnn_info: dict = {"ok": False, "msg": "未训练"}
    try:
        report("training", "训练模型 6/6：GNN 波动率风险叠加层...")
        import gnn_vol
        gnn_info = gnn_vol.train_gnn_vol(LAB_PATH, INDEX_SYMBOL, vt_symbols, progress=progress)
    except Exception as e:                                          # noqa: BLE001
        gnn_info = {"ok": False, "msg": f"GNN 波动率训练失败（不影响推荐/排名）：{e}"}
        report("computing", gnn_info["msg"])

    last_dt = signal["datetime"].max()
    report("done", f"完成。信号最新日期 {last_dt}，共 {len(vt_symbols)} 只股票。模式：{mode_txt}")
    return {
        "ok": True,
        "last_dt": last_dt.strftime("%Y-%m-%d"),
        "n_symbols": len(vt_symbols),
        "full_train": full,
        "horizons": HORIZONS,
        "train_period": train_p, "valid_period": valid_p, "test_period": test_p,
        "gnn_vol": gnn_info,
    }


def _load_horizon_predictions() -> tuple[dict[str, dict[int, float]], object]:
    """读取多视野宽表信号，返回 {vt_symbol: {h: ret_h}} （最新交易日）与该日期。

    宽表列：datetime, vt_symbol, ret_h2, ret_h5, ret_h10, ret_h20。
    信号缺失（未训练视野模型）时返回 ({}, None)，调用方回退到启发式区间。
    """
    lab = AlphaLab(LAB_PATH)
    wide = lab.load_signal(HORIZON_SIGNAL)
    if wide is None or wide.is_empty():
        return {}, None

    last_dt = wide["datetime"].max()
    day = wide.filter(pl.col("datetime") == last_dt)
    preds: dict[str, dict[int, float]] = {}
    for row in day.iter_rows(named=True):
        rets: dict[int, float] = {}
        for h in HORIZONS:
            col = f"ret_h{h}"
            if col in row and row[col] is not None:
                rets[h] = row[col]
        preds[row["vt_symbol"]] = rets
    return preds, last_dt


def _hold_fields(vt: str, horizon_preds: dict[str, dict[int, float]]) -> dict:
    """根据多视野预测组装某票持有天数字段；无预测时回退启发式区间。"""
    rets = horizon_preds.get(vt)
    if rets:
        s = suggest_hold_days(rets)
        return {
            "hold_lower": s["lower"], "hold_upper": s["upper"], "hold_best": s["best"],
            "hold_over_rotation": s["over_rotation"], "hold_bearish": s["bearish"],
            "hold_basis": s["basis"],
        }
    lower, upper = holding_days_range()
    return {
        "hold_lower": lower, "hold_upper": upper, "hold_best": None,
        "hold_over_rotation": False, "hold_bearish": False,
        "hold_basis": f"无多视野模型预测，回退策略轮动启发式区间 {lower}~{upper} 天",
    }


# 持仓评估表里的策略真实离场口径说明（ATR 出手价为研究参考，非策略本身的卖点）
STRATEGY_EXIT_NOTE = "策略真实离场：当排名跌出末尾时挂收盘价×0.95 保成交，无目标价概念；下方出手价为 ATR 多因子研究参考"


def get_recommendations(top_n: int = 3) -> dict:
    """读取最新信号，组装 Top-N 推荐（含现价、估值、多视野模型持有天数）。"""
    lab = AlphaLab(LAB_PATH)
    signal = lab.load_signal(ARTIFACT_NAME)
    if signal is None or signal.is_empty():
        return {"ok": False, "msg": "尚无信号数据，请先点击数据回训。", "items": []}

    last_dt = signal["datetime"].max()
    top = (
        signal.filter(pl.col("datetime") == last_dt)
        .sort("signal", descending=True)
        .head(top_n)
    )
    vt_list = list(top["vt_symbol"])

    # 估值与名称：优先 baostock 实时估值（失败不致命）；名称恒以本地缓存兜底
    name_cache = _load_names()
    try:
        meta = fetch_valuation_and_names(vt_list)
    except Exception:                                                   # noqa: BLE001
        meta = {}

    # 持有天数：多视野模型预测（缺失则逐票回退启发式）
    horizon_preds, _ = _load_horizon_predictions()
    fb_lower, fb_upper = holding_days_range()

    items = []
    for row in top.iter_rows(named=True):
        vt = row["vt_symbol"]
        m = meta.get(vt, {})
        price = get_current_price(vt)
        # 建议出手价：推荐均为买入候选(=加仓口径)，按 ATR+支撑/阻力多因子推算，非固定倍数
        stats = _calc_stats(vt)
        order_price, order_side, order_basis = _suggest_order_price("加仓", price, stats)
        item = {
            "vt_symbol": vt,
            "name": m.get("name") or name_cache.get(vt, vt),
            "signal": round(float(row["signal"]), 4),
            "price": price,
            "order_price": order_price,
            "order_side": order_side or "买入",
            "order_basis": order_basis,
            "atr_pct": round(stats["atr"] / price * 100, 2) if (stats and stats.get("atr") and price) else None,
            "pe": m.get("pe"), "pb": m.get("pb"), "ps": m.get("ps"),
        }
        item.update(_hold_fields(vt, horizon_preds))
        items.append(item)

    return {
        "ok": True,
        "as_of": last_dt.strftime("%Y-%m-%d"),
        "items": items,
        "hold_lower": fb_lower, "hold_upper": fb_upper,    # 顶层兜底区间（前端无逐票字段时回退）
    }


# 按需评估：池外票需足够历史才能算出 60 窗口特征（roc_60/corr_60 等回看 60 个交易日）
ONDEMAND_MIN_BARS = 65
ONDEMAND_FETCH_DAYS = 500          # 抓取回看日历天数（~340 交易日，足够 60 窗口 + 停牌冗余）


def _load_ondemand_bar_df(vt_symbol: str) -> pl.DataFrame | None:
    """读取池外票日线并复刻 AlphaLab.load_bar_df 的预处理，使特征口径与训练完全一致。

    复刻步骤（缺一不可，否则特征值与训练分布不符）：
      1. vwap = turnover / volume（用归一化前的原始量价）；
      2. close_0 归一：open/high/low/close 同除以窗口首个收盘价（Alpha158 全为比值特征，
         本身对 close_0 不敏感，但仍严格复刻以防口径漂移）；
      3. 停牌日（数值列整行求和为 0）置 NaN；
      4. 追加 vt_symbol 列。
    数据不足/首收无效时返回 None。
    """
    fp = _bar_path(vt_symbol)
    if fp is None:
        return None
    df = pl.read_parquet(fp).sort("datetime")
    if df.is_empty():
        return None

    df = df.with_columns((pl.col("turnover") / pl.col("volume")).alias("vwap"))

    close_0 = df.select(pl.col("close")).item(0, 0)
    if close_0 is None or close_0 == 0 or close_0 != close_0:    # None/0/NaN 首收无法归一
        return None
    df = df.with_columns(
        (pl.col("open") / close_0).alias("open"),
        (pl.col("high") / close_0).alias("high"),
        (pl.col("low") / close_0).alias("low"),
        (pl.col("close") / close_0).alias("close"),
    )

    numeric_columns = df.columns[1:]                            # 除 datetime 外的数值列
    mask = df[numeric_columns].sum_horizontal() == 0            # 整行为 0 = 停牌
    df = df.with_columns(
        [pl.when(mask).then(float("nan")).otherwise(pl.col(c)).alias(c) for c in numeric_columns]
    )
    return df.with_columns(pl.lit(vt_symbol).alias("vt_symbol"))


def _score_out_of_universe(
    vt_symbol: str,
    rank_model,
    horizon_models: dict[int, object],
    pool_signals: list[float],
) -> dict | None:
    """对池外票独立打分（外推）：复刻特征 -> 同一 booster 预测 -> 插入池内分布定位排名。

    数学依据：Alpha158 全为 ts_* 个股时序特征（无截面依赖），predict 为逐行映射且特征不做归一化，
    故单只票独立算出的 signal 与训练池 signal 同尺度、可直接比较排名。
    返回 {signal, rank, total, rets, last_dt}；数据不足/无法预测时返回 None。
    """
    bar = _load_ondemand_bar_df(vt_symbol)
    if bar is None or bar.height < ONDEMAND_MIN_BARS:
        return None

    # 用 Alpha158 仅取与训练完全一致的特征表达式列表（不调用 prepare_data，避免 spawn 进程池）
    ds = Alpha158(bar, train_period=("2018-01-01", "2018-01-01"),
                  valid_period=("2018-01-01", "2018-01-01"),
                  test_period=("2018-01-01", "2018-01-01"))
    feat_names: list[str] = []
    cols: list[pl.Series] = []
    for name, expr in ds.feature_expressions.items():           # 串行计算 158 特征（单票，量小）
        cols.append(calculate_by_expression(bar, expr)["data"].alias(name))
        feat_names.append(name)
    feat_df = bar.with_columns(cols).sort("datetime")

    # 取最新交易日一行特征 → 矩阵（列序 = add_feature 插入序 = booster 期望序）
    last = feat_df.tail(1)
    last_dt = last["datetime"][0]
    matrix = last.select(feat_names).fill_null(float("nan")).to_numpy()

    # 排名模型打分（与池内 signal 同尺度）
    sig = float(rank_model.model.predict(matrix)[0])

    # 多视野收益预测（用于持有天数估计；模型缺失则留空回退启发式）
    rets: dict[int, float] = {}
    for h, hm in horizon_models.items():
        if hm is not None and hm.model is not None:
            rets[h] = float(hm.model.predict(matrix)[0])

    # 插入池内 signal 分布定位排名：rank = 池内打分更高的数量 + 1，total = 池内数 + 1
    higher = sum(1 for v in pool_signals if v > sig)
    rank = higher + 1
    total = len(pool_signals) + 1
    return {"signal": round(sig, 4), "rank": rank, "total": total,
            "rets": rets, "last_dt": last_dt}


def _advice_by_rank(rank: int, total: int) -> tuple[str, str, bool]:
    """按因子打分排名映射三档建议（启发式，沿用策略 Top-K 轮动口径）。

    三档：加仓 / 持仓 / 减仓。rank: 1-based 排名（1=打分最高）；total: 当日参与排名总数。
    返回 (建议, 理由, urgent)；urgent 仅用于出手价：末段票走紧迫离场挂单口径，但建议标签仍统一为"减仓"。
    """
    if rank <= TOP_K:
        return "加仓", f"排名第 {rank}/{total}，处于策略买入区(前 {TOP_K})，因子打分强", False
    if rank <= int(total * 0.6):
        return "持仓", f"排名第 {rank}/{total}，中上水平，建议持有观望", False
    if rank <= int(total * 0.85):
        return "减仓", f"排名第 {rank}/{total}，因子打分偏弱，建议逢高减仓", False
    return "减仓", f"排名第 {rank}/{total}，处于末段，策略轮动会将其淘汰，建议尽快减仓", True


def _predicted_vol(vt: str, vol_map: dict) -> tuple[float | None, str | None]:
    """取某票前向波动率预测：池内用 GNN，池外/无 GNN 退回 trailing 已实现波动率（诚实兜底）。

    返回 (年化波动率, 来源)；来源 "gnn"/"trailing"，无数据返回 (None, None)。
    """
    if vt in vol_map:
        return vol_map[vt]["gnn"], "gnn"
    fp = _bar_path(vt)
    if fp is None:
        return None, None
    try:
        import gnn_vol
        tv = gnn_vol.trailing_vol(fp.parent, vt)
    except Exception:                                           # noqa: BLE001
        return None, None
    return (tv, "trailing") if tv is not None else (None, None)


def _risk_level(vol: float | None, pool_vols: list[float]) -> tuple[str | None, float | None]:
    """按预测波动率在池内分布的分位定风险档（相对口径，跨市场更稳健）。

    分位 <0.33 低 / 0.33~0.67 中 / ≥0.67 高。无 vol 或无参考分布返回 (None, None)。
    """
    if vol is None or vol != vol or not pool_vols:
        return None, None
    pct = sum(1 for v in pool_vols if v < vol) / len(pool_vols)
    if pct < 0.33:
        return "低", round(pct, 3)
    if pct < 0.67:
        return "中", round(pct, 3)
    return "高", round(pct, 3)


def _gate_advice_by_regime(advice: str, risk_level: str | None, regime: str) -> tuple[str, bool, str]:
    """regime 调节建议强度：仅在「市场高波动 + 个股高波动」时下调加仓为持仓，绝不上调。

    诚实约束：波动率不预测收益方向，故只用于控险（收紧加仓），不会把减仓改成加仓。
    返回 (调节后建议, 是否被下调, 说明)。
    """
    if regime == "stress" and risk_level == "高" and advice == "加仓":
        return "持仓", True, "市场处于高波动状态且该股波动居高，已将加仓下调为持仓以控制风险"
    return advice, False, ""


def evaluate_holdings(holdings: list[dict]) -> dict:
    """评估用户持仓：按因子排名给增减建议 + 盈亏展示。

    holdings: [{"vt_symbol": "600519.SSE", "buy_price": 1400, "volume": 100}, ...]
    增减建议仅来自模型因子排名（不依赖买入价）；买入价+股数仅用于盈亏展示。
    池外票（不在 116 训练池）走按需抓取 + 外推打分：把其因子打分插入训练池 signal 分布定位排名，
    并标注 out_of_universe=True，供前端显示"外推"。
    """
    lab = AlphaLab(LAB_PATH)
    signal = lab.load_signal(ARTIFACT_NAME)
    if signal is None or signal.is_empty():
        return {"ok": False, "msg": "尚无信号数据，请先点击数据回训。", "items": []}

    last_dt = signal["datetime"].max()
    day = signal.filter(pl.col("datetime") == last_dt).sort("signal", descending=True)
    total = day.height
    # 构造 排名 + 分值 查询表
    rank_map: dict[str, tuple[int, float]] = {}
    for i, row in enumerate(day.iter_rows(named=True), 1):
        rank_map[row["vt_symbol"]] = (i, round(float(row["signal"]), 4))

    # 持有天数：多视野模型预测（缺失则逐票回退启发式）
    horizon_preds, _ = _load_horizon_predictions()

    # 风险叠加层：GNN 前向波动率预测 + 市场 regime（缺失则降级，不影响排名建议）
    try:
        import gnn_vol
        vol_map = gnn_vol.predict_latest_vol(LAB_PATH, INDEX_SYMBOL)
        regime_info = gnn_vol.compute_regime(Path(LAB_PATH) / "daily", INDEX_SYMBOL)
    except Exception as e:                                       # noqa: BLE001
        vol_map, regime_info = {}, {"ok": False, "regime": "unknown", "msg": f"风险层不可用：{e}"}
    regime = regime_info.get("regime", "unknown")
    # 池内预测波动率分布，作个股风险分档的相对参考
    pool_vols = [v["gnn"] for v in vol_map.values() if v.get("gnn") is not None]

    # 先归一化所有持仓代码，区分池内/池外
    from fetch_to_lab import to_vt_symbol
    parsed: list[dict] = []
    for h in holdings:
        vt = str(h.get("vt_symbol", "")).strip().upper()
        if not vt:
            continue                                    # 跳过完全空白行
        if "." not in vt:                               # 容错：用户可能只填数字代码
            vt = to_vt_symbol(vt)
        parsed.append({"vt": vt, "buy_price": _to_float(h.get("buy_price")),
                       "volume": _to_float(h.get("volume"))})

    # 池外票：按需批量抓取（一次 baostock 登录）+ 载入模型，供外推打分
    out_syms = list(dict.fromkeys(p["vt"] for p in parsed if p["vt"] not in rank_map))
    rank_model = None
    horizon_models: dict[int, object] = {}
    pool_signals: list[float] = []
    if out_syms:
        try:
            from fetch_to_lab import fetch_ondemand_batch
            start = (datetime.today() - timedelta(days=ONDEMAND_FETCH_DAYS)).strftime("%Y%m%d")
            end = datetime.today().strftime("%Y%m%d")
            fetch_ondemand_batch(out_syms, ONDEMAND_DIR, start, end, adjust="qfq")
        except Exception:                               # noqa: BLE001
            pass                                        # 抓取失败仍尝试用已有缓存评估
        rank_model = lab.load_model(ARTIFACT_NAME)
        horizon_models = {h: lab.load_model(f"{ARTIFACT_NAME}_h{h}") for h in HORIZONS}
        pool_signals = [float(v) for v in day["signal"].to_list()]

    items = []
    for p in parsed:
        vt, buy_price, volume = p["vt"], p["buy_price"], p["volume"]
        price = get_current_price(vt)
        in_signal = vt in rank_map

        # 盈亏：需现价 + 买入价 + 股数齐全才计算；缺任一项只跳过盈亏，不影响建议
        pnl = pnl_pct = market_value = cost = None
        if price is not None and buy_price is not None and volume is not None:
            cost = round(buy_price * volume, 2)
            market_value = round(price * volume, 2)
            pnl = round((price - buy_price) * volume, 2)
            pnl_pct = round((price / buy_price - 1) * 100, 2) if buy_price else None

        if in_signal:
            rank, sig = rank_map[vt]
            advice, reason, urgent = _advice_by_rank(rank, total)
            cur_total, out_of_universe, extra_reason = total, False, ""
            cur_horizon_preds = horizon_preds
        else:
            # 池外票：按需打分（外推）
            scored = None
            if rank_model is not None:
                scored = _score_out_of_universe(vt, rank_model, horizon_models, pool_signals)
            if scored is None:
                items.append({
                    "vt_symbol": vt, "buy_price": buy_price, "volume": volume,
                    "price": price, "cost": cost, "market_value": market_value,
                    "pnl": pnl, "pnl_pct": pnl_pct,
                    "advice": "无法评估", "rank": None, "signal": None,
                    "reason": "无法获取该代码足够的历史行情（代码可能无效或数据源暂不可用），无法外推打分",
                    "evaluable": False, "out_of_universe": True,
                })
                continue
            rank, cur_total, sig = scored["rank"], scored["total"], scored["signal"]
            advice, reason, urgent = _advice_by_rank(rank, cur_total)
            out_of_universe = True
            extra_reason = "（池外票：将因子打分插入训练池分布定位排名，属外推估计）"
            cur_horizon_preds = {vt: scored["rets"]}    # 用按需算出的多视野预测估持有天数

        # 风险叠加：前向波动率预测（池内 GNN / 池外 trailing）→ 相对分档 → regime 调节建议强度
        pred_vol, vol_src = _predicted_vol(vt, vol_map)
        risk_level, risk_pct = _risk_level(pred_vol, pool_vols)
        advice, gated, gate_note = _gate_advice_by_regime(advice, risk_level, regime)
        if gated:
            reason += f"；{gate_note}"
            urgent = False                              # 被下调为持仓后不再走紧迫离场挂单口径

        # 建议出手价：按 ATR(真实波幅)+排名紧迫度+支撑/阻力多因子推算，非固定倍数
        stats = _calc_stats(vt)
        order_price, order_side, order_basis = _suggest_order_price(advice, price, stats, urgent)
        item = {
            "vt_symbol": vt, "buy_price": buy_price, "volume": volume,
            "price": price, "cost": cost, "market_value": market_value,
            "pnl": pnl, "pnl_pct": pnl_pct,
            "advice": advice, "rank": rank, "signal": sig, "total_ranked": cur_total,
            "order_price": order_price, "order_side": order_side,
            "order_basis": order_basis,
            "strategy_exit_note": STRATEGY_EXIT_NOTE,       # 策略真实离场口径（排名跌出挂收盘×0.95），与 ATR 出手价区分
            "atr_pct": round(stats["atr"] / price * 100, 2) if (stats and stats.get("atr") and price) else None,
            "reason": reason + extra_reason, "evaluable": True,
            "out_of_universe": out_of_universe,
            # 风险层字段：前向年化波动率预测、来源(gnn/trailing)、相对分档、是否被 regime 下调
            "pred_vol": round(pred_vol, 4) if pred_vol is not None else None,
            "pred_vol_pct": round(pred_vol * 100, 1) if pred_vol is not None else None,
            "vol_source": vol_src,
            "risk_level": risk_level, "risk_pct": risk_pct,
            "advice_gated": gated,
        }
        item.update(_hold_fields(vt, cur_horizon_preds))
        items.append(item)

    return {
        "ok": True, "as_of": last_dt.strftime("%Y-%m-%d"),
        "total_ranked": total, "items": items,
        "regime": regime, "regime_msg": regime_info.get("msg", ""),
        "vol_available": bool(vol_map),     # 前端据此决定是否展示波动率风险列
    }


if __name__ == "__main__":
    # 命令行直接跑一次回训+推荐（便于离线验证）
    import json
    def _p(stage, msg):
        print(f"[{stage}] {msg}")
    info = run_retrain(progress=_p, fetch=False)
    print(json.dumps(info, ensure_ascii=False))
    print(json.dumps(get_recommendations(3), ensure_ascii=False, indent=2))
