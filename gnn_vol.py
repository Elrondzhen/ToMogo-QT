"""
GNN 波动率风险叠加层（参考 /Users/10298918/project/test2/gnn 论文，适配 116 只 A 股）。

定位（务必诚实）：
  本层只预测「前向已实现波动率」并作风险叠加，绝不当收益信号。论文实证表明：
    - GNN 波动率预测的「预测式多空策略失败」(论文 6.7)；
    - 图结构有时帮忙有时反而有害 (论文 6.4/6.5)；
    - 最优也只是最小方差组合 Sharpe≈0.98，而简单 HAR 基线就有 0.73。
  因此本模块强制：每次都与「持续性基线」(下期波动率≈本期 trailing RV) 对比 MSE，
  GNN 没赢过基线就如实标注，不夸大其价值。

与论文的差异（数据可得性，如实说明）：
  论文 macro 为美股专属(VIX/美债利差/信用利差)，A 股无现成免费源，故 macro 退化为可计算子集：
  {沪深300已实现波动率, 沪深300收益, 全池平均两两相关, 相关图密度}。

实现选择：
  - 目标：前向 H=5 交易日(≈论文周频) 已实现波动率（年化）。
  - 图：相关图(63d 滚动 Pearson，|ρ|≥0.30)，周频网格重算（116 节点稠密邻接，开销极小）。
  - 骨干：手写 GraphSAGE(3 层 / hidden 256 / dropout 0.3，论文超参)，纯 PyTorch，无需 torch_geometric。

研究演示用途，非实盘投资建议。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import polars as pl

# ---- 常量（窗口与论文超参） ----
HORIZON = 5                      # 前向预测视野（交易日），≈论文周频
RV_WINDOWS = [5, 10, 21, 63]    # 已实现波动率回看窗口
MOM_WINDOWS = [5, 20]           # 动量/成交量窗口
CORR_WINDOW = 63                # 相关图滚动窗口（论文 macro-conditioned 候选之一）
CORR_THRESHOLD = 0.30           # 边规则 |ρ|≥0.30（论文选定值）
WEEK_STRIDE = 5                 # 周频网格步长（每 5 个交易日取一个样本点）
TRADING_DAYS = 252              # 年化因子

# GraphSAGE 超参（论文 grid search 选定值）
GNN_HIDDEN = 256
GNN_LAYERS = 3
GNN_DROPOUT = 0.3
GNN_LR = 1e-3
GNN_EPOCHS = 200
GNN_PATIENCE = 30               # 早停耐心（验证 MSE 不降的最大 epoch 数）

MODEL_SUBDIR = "gnn_vol"        # 存于 {LAB_PATH}/model/gnn_vol/


# ==================== 数据 / panel 构造 ====================

def _load_panel_matrices(
    daily_dir: Path, vt_symbols: list[str], coverage: float = 0.9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]] | None:
    """读取池内各票收盘价 + 成交量，按日期交集对齐为矩阵。

    返回 (datetimes[T], close[T,N], volume[T,N], symbols[N])；不足两只或对齐后为空返回 None。
    用真实收盘价（未做 close_0 归一）：波动率是对收益的统计，归一只改尺度不改收益序列。

    关键：先按「历史覆盖度」筛票再做交集，否则单只新上市票（如仅 200 多行）会把
    全员交易日交集塌缩到它的上市日，导致样本骤减。只保留行数 ≥ coverage×最长历史的票
    （上市晚的少数票被剔除，叠加层会让它们退回 trailing 波动率基线），使长历史簇得以保留。
    """
    raw: list[tuple[str, pl.DataFrame]] = []
    for vt in vt_symbols:
        fp = daily_dir / f"{vt}.parquet"
        if not fp.exists():
            continue
        d = pl.read_parquet(fp).select(
            pl.col("datetime"),
            pl.col("close").alias(f"c_{vt}"),
            pl.col("volume").alias(f"v_{vt}"),
        ).sort("datetime")
        raw.append((vt, d))
    if len(raw) < 2:
        return None

    # 覆盖度筛选：剔除历史明显偏短的票（上市晚），避免交集塌缩
    max_rows = max(d.height for _, d in raw)
    kept_pairs = [(vt, d) for vt, d in raw if d.height >= coverage * max_rows]
    if len(kept_pairs) < 2:
        kept_pairs = raw                                        # 极端情况下不筛，保底可跑
    kept = [vt for vt, _ in kept_pairs]
    frames = [d for _, d in kept_pairs]

    wide = frames[0]
    for d in frames[1:]:
        wide = wide.join(d, on="datetime", how="inner")     # 交集对齐：仅保留全员都有的交易日
    wide = wide.sort("datetime").drop_nulls()
    if wide.is_empty():
        return None

    dts = wide["datetime"].to_numpy()
    close = wide.select([f"c_{vt}" for vt in kept]).to_numpy().astype(np.float64)
    volume = wide.select([f"v_{vt}" for vt in kept]).to_numpy().astype(np.float64)
    return dts, close, volume, kept


def _align_index_returns(daily_dir: Path, index_symbol: str, dts: np.ndarray) -> np.ndarray | None:
    """把指数日线对齐到 panel 的交易日序列 dts，返回指数对数收益数组[T]（首行 0）。

    指数缺文件或对齐后长度不符时返回 None（调用方降级为无 macro 的指数项）。
    """
    fp = daily_dir / f"{index_symbol}.parquet"
    if not fp.exists():
        return None
    idx = pl.read_parquet(fp).select(
        pl.col("datetime"), pl.col("close").alias("idx_close")
    ).sort("datetime")
    # 用 panel 日期左连接指数，保证逐日对齐；缺失日 forward fill
    base = pl.DataFrame({"datetime": dts}).join(idx, on="datetime", how="left")
    base = base.with_columns(pl.col("idx_close").forward_fill())
    close = base["idx_close"].to_numpy().astype(np.float64)
    if np.isnan(close).any():
        return None
    return _log_returns(close)


def _corr_adjacency(logret_win: np.ndarray) -> tuple[np.ndarray, float, float]:
    """由一段对数收益窗口[w,N] 算相关图邻接 A[N,N]（|ρ|≥阈值，无自环）。

    返回 (A, avg_abs_corr, density)：
      avg_abs_corr = 所有股票对 |ρ| 均值（含未连边的对，与论文一致）；
      density = 连边数 / 可能对数。
    """
    n = logret_win.shape[1]
    # 标准差为 0 的列（窗口内无波动）会让 corrcoef 出 NaN，预先置 0 相关
    c = np.corrcoef(logret_win, rowvar=False)
    c = np.nan_to_num(c, nan=0.0)
    np.fill_diagonal(c, 0.0)
    absc = np.abs(c)
    A = (absc >= CORR_THRESHOLD).astype(np.float64)
    np.fill_diagonal(A, 0.0)

    iu = np.triu_indices(n, k=1)
    avg_abs_corr = float(absc[iu].mean()) if iu[0].size else 0.0
    density = float(A[iu].mean()) if iu[0].size else 0.0
    return A, avg_abs_corr, density


# 个股特征名（10 个，与论文 stock-level 特征对齐）
NODE_FEATURES = [
    "rv_5", "rv_10", "rv_21", "rv_63", "rv_short_long",
    "mom_5", "mom_20", "logvol_5", "logvol_20", "vol_short_long",
]
# macro 特征名（4 个，A 股可计算子集；论文美股版含 VIX/利差/信用利差，此处如实退化）
MACRO_FEATURES = ["idx_rv_21", "idx_ret_5", "avg_abs_corr", "graph_density"]

START_OFFSET = max(max(RV_WINDOWS), CORR_WINDOW, max(MOM_WINDOWS))   # 首个可算样本的最小历史


def _node_features_at(close: np.ndarray, volume: np.ndarray, logret: np.ndarray, i: int) -> np.ndarray:
    """在时间索引 i 处计算每只股票的 10 维特征，返回 [N, 10]（用截至 i 的信息，无前瞻）。"""
    n = close.shape[1]
    feats = np.zeros((n, len(NODE_FEATURES)), dtype=np.float64)
    rv = {}
    for w in RV_WINDOWS:
        win = logret[i - w + 1: i + 1]                       # 最近 w 日对数收益
        rv[w] = np.std(win, axis=0, ddof=1) * math.sqrt(TRADING_DAYS)
    feats[:, 0] = rv[5]
    feats[:, 1] = rv[10]
    feats[:, 2] = rv[21]
    feats[:, 3] = rv[63]
    feats[:, 4] = rv[5] / (rv[63] + 1e-12)                   # 短长比：尖峰相对长期基线
    feats[:, 5] = close[i] / close[i - 5] - 1                # 5 日动量
    feats[:, 6] = close[i] / close[i - 20] - 1               # 20 日动量
    vol5 = volume[i - 5 + 1: i + 1].mean(axis=0)
    vol20 = volume[i - 20 + 1: i + 1].mean(axis=0)
    feats[:, 7] = np.log(vol5 + 1.0)
    feats[:, 8] = np.log(vol20 + 1.0)
    feats[:, 9] = vol5 / (vol20 + 1e-12)                     # 短长成交量比：异常放量
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def _forward_vol_target(logret: np.ndarray, i: int, horizon: int) -> np.ndarray:
    """前向 horizon 日已实现波动率目标 [N]：用 i+1..i+horizon 的对数收益（无前瞻泄漏）。"""
    win = logret[i + 1: i + 1 + horizon]
    return np.std(win, axis=0, ddof=1) * math.sqrt(TRADING_DAYS)


def build_samples(daily_dir: Path, index_symbol: str, vt_symbols: list[str]) -> dict | None:
    """周频网格构造完整训练样本。

    遍历时间索引 i ∈ [START_OFFSET, T-1-HORIZON]（步长 WEEK_STRIDE），每点产出：
      X[s, N, 10]  个股特征      M[s, 4] macro 特征    A[s, N, N] 相关图邻接
      Y[s, N]      前向波动率目标 P[s, N] 持续性基线(=本期 rv_5，下期≈本期)
    返回 dict 含上述数组 + symbols + sample_dts；数据不足返回 None。

    macro 的 idx_rv_21/idx_ret_5 来自指数（缺指数则置 0 并照常训练，如实退化）。
    """
    loaded = _load_panel_matrices(daily_dir, vt_symbols)
    if loaded is None:
        return None
    dts, close, volume, symbols = loaded
    t, n = close.shape
    if t < START_OFFSET + HORIZON + 2:
        return None

    logret = _log_returns(close)
    idx_ret = _align_index_returns(daily_dir, index_symbol, dts)   # None 时 macro 指数项置 0

    xs, ms, adjs, ys, ps, samp_dts = [], [], [], [], [], []
    last_i = t - 1 - HORIZON
    for i in range(START_OFFSET, last_i + 1, WEEK_STRIDE):
        X = _node_features_at(close, volume, logret, i)           # [N,10]
        corr_win = logret[i - CORR_WINDOW + 1: i + 1]             # [63,N]
        A, avg_abs_corr, density = _corr_adjacency(corr_win)

        if idx_ret is not None:
            idx_rv_21 = float(np.std(idx_ret[i - 21 + 1: i + 1], ddof=1) * math.sqrt(TRADING_DAYS))
            idx_ret_5 = float(np.sum(idx_ret[i - 5 + 1: i + 1]))  # 近 5 日累计对数收益≈区间收益
        else:
            idx_rv_21 = idx_ret_5 = 0.0
        M = np.array([idx_rv_21, idx_ret_5, avg_abs_corr, density], dtype=np.float64)

        Y = _forward_vol_target(logret, i, HORIZON)               # [N]
        P = X[:, 0].copy()                                        # 持续性基线 = 本期 rv_5（同年化口径）

        xs.append(X); ms.append(M); adjs.append(A); ys.append(Y); ps.append(P)
        samp_dts.append(dts[i])

    if len(xs) < 10:                                              # 样本太少不足以训练/切分
        return None
    return {
        "X": np.stack(xs), "M": np.stack(ms), "A": np.stack(adjs),
        "Y": np.stack(ys), "P": np.stack(ps),
        "symbols": symbols, "sample_dts": np.array(samp_dts),
    }


# ==================== 特征预处理 ====================

def _winsorize_zscore_week(X_week: np.ndarray) -> np.ndarray:
    """单周个股特征 [N,10] 截面 winsorize(1/99 分位) + z-score（论文口径）。"""
    out = X_week.copy()
    for j in range(out.shape[1]):
        col = out[:, j]
        lo, hi = np.percentile(col, 1), np.percentile(col, 99)
        col = np.clip(col, lo, hi)
        mu, sd = col.mean(), col.std()
        out[:, j] = (col - mu) / sd if sd > 1e-12 else 0.0
    return out


def _build_input_tensor(samples: dict, train_n: int) -> np.ndarray:
    """组装 GNN 输入 [S, N, 14]：10 个股特征(逐周截面标准化) ++ 4 macro(训练统计标准化，广播到每只)。

    macro 用训练段(前 train_n 个样本)的均值/方差标准化——市场级变量不做截面缩放（论文口径），
    且只用训练统计避免测试信息泄漏。
    """
    X, M = samples["X"], samples["M"]
    s, n, _ = X.shape

    # 个股特征：逐周截面 winsorize+zscore（每周独立，无跨期泄漏）
    Xn = np.stack([_winsorize_zscore_week(X[k]) for k in range(s)])

    # macro：训练段统计标准化
    m_mu = M[:train_n].mean(axis=0)
    m_sd = M[:train_n].std(axis=0)
    m_sd = np.where(m_sd > 1e-12, m_sd, 1.0)
    Mn = (M - m_mu) / m_sd                                        # [S,4]

    # macro 广播到每只股票并与个股特征拼接 → [S,N,14]
    Mb = np.repeat(Mn[:, None, :], n, axis=1)
    return np.concatenate([Xn, Mb], axis=2), (m_mu, m_sd)


# ==================== 手写 GraphSAGE（纯 PyTorch） ====================

def _import_torch():
    """惰性导入 torch（仅 GNN 路径需要，避免无 torch 环境下整模块不可用）。"""
    import torch
    import torch.nn as nn
    return torch, nn


def _make_model(in_dim: int):
    """构建 GraphSAGE 模型（3 层 / hidden 256 / dropout 0.3，论文超参）。

    mean 聚合：邻居均值 = (A @ X) / deg，与自身特征拼接后线性变换。
    稠密邻接 [B,N,N] 经 bmm 批量聚合（116 节点平凡，无需 torch_geometric）。
    """
    torch, nn = _import_torch()

    class SageLayer(nn.Module):
        def __init__(self, fin: int, fout: int) -> None:
            super().__init__()
            self.lin = nn.Linear(fin * 2, fout)               # 拼接自身 + 邻居均值

        def forward(self, x, a):                              # x:[B,N,F] a:[B,N,N]
            deg = a.sum(dim=-1, keepdim=True).clamp(min=1.0)  # 无邻居则除 1（=只用自身）
            neigh = torch.bmm(a, x) / deg                     # 邻居均值聚合
            return self.lin(torch.cat([x, neigh], dim=-1))

    class GraphSAGE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            dims = [in_dim] + [GNN_HIDDEN] * GNN_LAYERS
            self.layers = nn.ModuleList(
                [SageLayer(dims[k], dims[k + 1]) for k in range(GNN_LAYERS)]
            )
            self.drop = nn.Dropout(GNN_DROPOUT)
            self.head = nn.Linear(GNN_HIDDEN, 1)              # 输出每节点标量波动率预测

        def forward(self, x, a):
            for layer in self.layers:
                x = torch.relu(layer(x, a))
                x = self.drop(x)
            return self.head(x).squeeze(-1)                  # [B,N]

    return GraphSAGE()


# ==================== 训练（含诚实性基线对比） ====================

def train_gnn_vol(lab_path: str, index_symbol: str, vt_symbols: list[str],
                  progress=None) -> dict:
    """训练 GraphSAGE 波动率模型，时序 70/15/15 切分，早停于验证 MSE。

    强制与「持续性基线」(下期波动率≈本期 rv_5) 对比测试 MSE 并如实记录 beats_baseline。
    产出落盘到 {lab_path}/model/gnn_vol/：权重 state_dict、macro 归一统计、symbols、元信息。
    返回 {ok, test_mse_gnn, test_mse_persistence, beats_baseline, n_samples, ...}。
    """
    def report(msg: str) -> None:
        if progress:
            progress("training", msg)

    torch, nn = _import_torch()
    daily_dir = Path(lab_path) / "daily"
    samples = build_samples(daily_dir, index_symbol, vt_symbols)
    if samples is None:
        return {"ok": False, "msg": "样本不足，无法训练 GNN 波动率模型"}

    s = samples["X"].shape[0]
    n_tr = int(s * 0.70)
    n_va = int(s * 0.85)
    Xall, _stats = _build_input_tensor(samples, n_tr)            # [S,N,14]
    A, Y, P = samples["A"], samples["Y"], samples["P"]

    tX = torch.tensor(Xall, dtype=torch.float32)
    tA = torch.tensor(A, dtype=torch.float32)
    tY = torch.tensor(Y, dtype=torch.float32)

    model = _make_model(Xall.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=GNN_LR)
    lossf = nn.MSELoss()

    best_va = float("inf")
    best_state = None
    wait = 0
    report(f"GNN 波动率：样本 {s}（训练 {n_tr}/验证 {n_va - n_tr}/测试 {s - n_va}），开始训练...")
    for ep in range(GNN_EPOCHS):
        model.train()
        opt.zero_grad()
        pred = model(tX[:n_tr], tA[:n_tr])
        loss = lossf(pred, tY[:n_tr])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            va = lossf(model(tX[n_tr:n_va], tA[n_tr:n_va]), tY[n_tr:n_va]).item()
        if va < best_va - 1e-9:
            best_va, best_state, wait = va, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= GNN_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # 测试段：GNN vs 持续性基线 MSE（诚实对比）
    model.eval()
    with torch.no_grad():
        gnn_test = model(tX[n_va:], tA[n_va:]).numpy()
    y_test = Y[n_va:]
    mse_gnn = float(np.mean((gnn_test - y_test) ** 2))
    mse_base = float(np.mean((P[n_va:] - y_test) ** 2))

    _save_gnn(lab_path, model, _stats, samples["symbols"])
    beats = mse_gnn < mse_base
    report(f"GNN 波动率训练完成：测试 MSE GNN={mse_gnn:.5f} vs 持续性基线={mse_base:.5f}"
           f"（{'GNN 占优' if beats else 'GNN 未赢过基线，按基线口径使用'}）")
    return {
        "ok": True, "n_samples": s,
        "test_mse_gnn": round(mse_gnn, 6),
        "test_mse_persistence": round(mse_base, 6),
        "beats_baseline": beats,
    }


# ==================== 保存 / 读取 ====================

def _gnn_dir(lab_path: str) -> Path:
    d = Path(lab_path) / "model" / MODEL_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_gnn(lab_path: str, model, stats: tuple, symbols: list[str]) -> None:
    """保存 GNN 权重 + macro 归一统计 + symbols + 输入维度到 {lab}/model/gnn_vol/gnn.pt。"""
    torch, _ = _import_torch()
    m_mu, m_sd = stats
    payload = {
        "state_dict": model.state_dict(),
        "m_mu": m_mu, "m_sd": m_sd,
        "symbols": symbols,
        "in_dim": len(NODE_FEATURES) + len(MACRO_FEATURES),
        "horizon": HORIZON,
    }
    torch.save(payload, _gnn_dir(lab_path) / "gnn.pt")


def _load_gnn(lab_path: str):
    """读取 GNN 权重与元信息，重建模型。无文件返回 (None, None)。"""
    fp = _gnn_dir(lab_path) / "gnn.pt"
    if not fp.exists():
        return None, None
    torch, _ = _import_torch()
    payload = torch.load(fp, weights_only=False)
    model = _make_model(payload["in_dim"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


# ==================== 推断 / 风险叠加输入 ====================

def _build_latest_input(daily_dir: Path, index_symbol: str, symbols: list[str],
                        m_mu: np.ndarray, m_sd: np.ndarray) -> tuple | None:
    """组装「最新交易日」的 GNN 输入（无 target，用于预测未来波动率）。

    用训练时保存的 symbols 顺序与 macro 归一统计，保证与训练口径一致。
    返回 (X[1,N,14], A[1,N,N], rv5[N], dt)；数据不足或无指数对齐失败时仍照常（macro 指数项置 0）。
    """
    loaded = _load_panel_matrices(daily_dir, symbols)
    if loaded is None:
        return None
    dts, close, volume, kept = loaded
    if kept != symbols:                                          # 池构成变化则不强行预测（维度/顺序须一致）
        return None
    t = close.shape[0]
    if t < START_OFFSET + 1:
        return None

    logret = _log_returns(close)
    i = t - 1                                                    # 最新交易日
    X = _node_features_at(close, volume, logret, i)             # [N,10]
    A, avg_abs_corr, density = _corr_adjacency(logret[i - CORR_WINDOW + 1: i + 1])

    idx_ret = _align_index_returns(daily_dir, index_symbol, dts)
    if idx_ret is not None:
        idx_rv_21 = float(np.std(idx_ret[i - 21 + 1: i + 1], ddof=1) * math.sqrt(TRADING_DAYS))
        idx_ret_5 = float(np.sum(idx_ret[i - 5 + 1: i + 1]))
    else:
        idx_rv_21 = idx_ret_5 = 0.0
    M = np.array([idx_rv_21, idx_ret_5, avg_abs_corr, density], dtype=np.float64)

    Xn = _winsorize_zscore_week(X)
    m_sd_safe = np.where(m_sd > 1e-12, m_sd, 1.0)
    Mn = (M - m_mu) / m_sd_safe
    Mb = np.repeat(Mn[None, :], X.shape[0], axis=0)             # [N,4]
    feat = np.concatenate([Xn, Mb], axis=1)[None, :, :]        # [1,N,14]
    return feat, A[None, :, :], X[:, 0], dts[i]


def predict_latest_vol(lab_path: str, index_symbol: str) -> dict:
    """用已训 GNN 预测最新交易日各池内票的前向波动率。

    返回 {vt: {"gnn": 预测年化波动率, "trailing": 本期 rv_5 持续性基线}}；无模型/数据不足返回 {}。
    """
    model, payload = _load_gnn(lab_path)
    if model is None:
        return {}
    symbols = payload["symbols"]
    built = _build_latest_input(Path(lab_path) / "daily", index_symbol, symbols,
                                payload["m_mu"], payload["m_sd"])
    if built is None:
        return {}
    feat, A, rv5, _dt = built
    torch, _ = _import_torch()
    with torch.no_grad():
        pred = model(torch.tensor(feat, dtype=torch.float32),
                     torch.tensor(A, dtype=torch.float32)).numpy()[0]
    return {vt: {"gnn": float(pred[k]), "trailing": float(rv5[k])}
            for k, vt in enumerate(symbols)}


def trailing_vol(daily_dir: Path, vt_symbol: str, window: int = HORIZON) -> float | None:
    """任意单票的 trailing 已实现波动率（年化），作池外票/无 GNN 时的诚实兜底。

    用最近 window 日对数收益的年化标准差，与训练目标同口径。数据不足返回 None。
    """
    fp = daily_dir / f"{vt_symbol}.parquet"
    if not fp.exists():
        return None
    df = pl.read_parquet(fp).sort("datetime")
    if df.height < window + 1:
        return None
    close = df["close"].to_numpy().astype(np.float64)[-(window + 1):]
    logret = np.diff(np.log(close))
    if logret.size < 2:
        return None
    return float(np.std(logret, ddof=1) * math.sqrt(TRADING_DAYS))


# ==================== regime（市场状态，由沪深300推断） ====================

REGIME_LOOKBACK = 252           # regime 分位回看（约一年），用指数已实现波动率历史定位当前水平


def compute_regime(daily_dir: Path, index_symbol: str) -> dict:
    """由沪深300指数近 21 日已实现波动率在过去一年的分位定位市场状态。

    分位 <0.5 → calm（温和）；0.5~0.8 → normal（常态）；≥0.8 → stress（高波动）。
    regime 仅用于调节建议强度（stress 收紧加仓、放宽减仓），不改变排名本身。
    返回 {ok, regime, idx_vol, percentile, msg}；指数缺失返回 ok=False（调用方退化为不调节）。
    """
    fp = daily_dir / f"{index_symbol}.parquet"
    if not fp.exists():
        return {"ok": False, "regime": "unknown", "msg": "缺少沪深300指数数据，无法判定市场状态"}
    df = pl.read_parquet(fp).sort("datetime")
    if df.height < REGIME_LOOKBACK + 22:
        return {"ok": False, "regime": "unknown", "msg": "指数历史不足，无法判定市场状态"}

    close = df["close"].to_numpy().astype(np.float64)
    logret = np.diff(np.log(close))
    # 逐日滚动 21 日年化波动率序列（取最近 REGIME_LOOKBACK 个点定分位）
    vols = []
    for k in range(21, len(logret) + 1):
        vols.append(np.std(logret[k - 21:k], ddof=1) * math.sqrt(TRADING_DAYS))
    vols = np.array(vols)
    hist = vols[-REGIME_LOOKBACK:]
    cur = float(vols[-1])
    pct = float((hist < cur).mean())                            # 当前波动在过去一年中的分位

    if pct < 0.5:
        regime, txt = "calm", "温和"
    elif pct < 0.8:
        regime, txt = "normal", "常态"
    else:
        regime, txt = "stress", "高波动"
    return {
        "ok": True, "regime": regime,
        "idx_vol": round(cur, 4), "percentile": round(pct, 3),
        "msg": f"市场状态：{txt}（沪深300近21日年化波动 {cur * 100:.1f}%，处于近一年第 {pct * 100:.0f} 分位）",
    }







def _log_returns(close: np.ndarray) -> np.ndarray:
    """逐列对数收益（行=时间，列=股票）。首行收益置 0（无前值）。"""
    r = np.zeros_like(close)
    r[1:] = np.log(close[1:] / close[:-1])
    return r


def _realized_vol(logret_window: np.ndarray) -> float:
    """一段对数收益的年化已实现波动率：std(daily logret) * sqrt(252)。"""
    if logret_window.size < 2:
        return float("nan")
    return float(np.std(logret_window, ddof=1) * math.sqrt(TRADING_DAYS))

