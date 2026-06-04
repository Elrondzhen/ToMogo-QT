# ToMogo-QT 量化选股研究台

> **策略**：Alpha158 因子 + LightGBM 排名 · GNN 前向波动率风险叠加 · 基于沪深300成分股真实日线

基于Alpha158 因子 + LightGBM的多因子选股研究 Web 应用。功能：

- **当日推荐 Top3**：模型因子打分，含现价（前复权）、PE/PB/PS 估值、建议出手价（ATR 推算）、多视野建议持有天数。
- **我的持仓评估**：录入持仓（代码/买入价/股数）→ 模型按因子排名给出增/减建议、盈亏、风险档、持有天数；池外票按需抓取后外推排名。
- **GNN 前向波动率风险叠加**：高波动市场状态下收紧加仓建议（仅控险，不预测涨跌方向）。
- **数据回训**：baostock 增量抓数 → 重算因子 → 重训 5 个 LightGBM（1 排名 + 4 视野）+ GNN。

> **研究演示声明**：本工具为机器学习多因子选股的研究演示，所有输出**不构成任何投资建议**。"建议持有天数"是依据策略轮动规则推导的**启发式估计**，并非模型预测。当前数据集为 116 只沪深300成分股（受数据源限流，非全量），结果存在样本偏差。请勿据此进行实盘交易。

## 目录结构

```
ToMogo-QT/
├── tomogoqt/                #  tomogoqt.alpha 框架（alpha + trader 子集）
├── lab/demo/                # 数据中心：116只CSI300日线 / 模型(*.pkl) / 信号 / 名称缓存
├── scripts/fetch_to_lab.py  # baostock 取数脚本
├── pipeline.py              # 抓数→因子→训练→推荐→持仓评估 流程封装
├── gnn_vol.py               # GNN 前向波动率风险叠加层（PyTorch）
├── app.py                   # Flask 后端（仅绑定 127.0.0.1）
├── templates/index.html     # 前端界面
└── requirements.txt
```

## 环境要求

- **Python 3.10+**（建议 3.11）
- **TA-Lib C 库**（`ta-lib` Python 包的底层依赖，需单独安装）
- 约 2GB 磁盘（含 PyTorch）

---

## 安装

三个平台都遵循同一思路：① 装 TA-Lib C 库 → ② 建虚拟环境 → ③ `pip install -r requirements.txt`。区别只在第 ① 步和命令语法。

### macOS

```bash
brew install ta-lib                       # 提供 talib 所需的 C 库
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Linux（Debian/Ubuntu）

```bash
sudo apt-get install -y build-essential wget
wget https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz
tar -xzf ta-lib-0.6.4-src.tar.gz && cd ta-lib-0.6.4 && ./configure --prefix=/usr && make && sudo make install && cd ..
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows

Windows 安装 TA-Lib 的 C 库较麻烦，**推荐用 Conda**（自动处理 C 库 + Python 包），最省事：

**方式 A — Conda（推荐）**

用 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)，在 **Anaconda Prompt** 中执行：

```bat
conda create -n tomogoqt python=3.11
conda activate tomogoqt
conda install -c conda-forge ta-lib
pip install -r requirements.txt
```

**方式 B — 原生 venv + 预编译库**

1. 装 [Python 3.11](https://www.python.org/downloads/windows/)（勾选 *Add python.exe to PATH*）。
2. 装 TA-Lib C 库：从 <https://github.com/ta-lib/ta-lib/releases> 下载 `ta-lib-0.6.4-windows-x86_64.msi` 并安装（默认装到 `C:\Program Files\TA-Lib`）。
3. 建环境并安装依赖：

   - **CMD（命令提示符）**
     ```bat
     py -3.11 -m venv .venv
     .venv\Scripts\activate.bat
     pip install -r requirements.txt
     ```
   - **PowerShell**
     ```powershell
     py -3.11 -m venv .venv
     .venv\Scripts\Activate.ps1
     pip install -r requirements.txt
     ```
     > 若 PowerShell 报“禁止运行脚本”，先执行：
     > `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

   若 `pip install ta-lib` 仍报找不到 C 库，改用方式 A（Conda）最稳妥。

> 备注：`requirements.txt` 含 `torch`，Windows 默认从 PyPI 装 CPU 版即可，无需 CUDA。



---

## 启动

激活上面建好的环境后，在项目根目录执行：

| 平台 | 命令 |
|------|------|
| macOS / Linux | `python app.py` |
| Windows | `python app.py` |


浏览器访问 **http://127.0.0.1:5000**

> **macOS 用户务必用 `127.0.0.1`，不要用 `localhost`**——macOS 的 AirPlay 接收器会占用 IPv6 的 5000 端口。Windows/Linux 用 `localhost` 或 `127.0.0.1` 均可。
>
> 服务只绑定 `127.0.0.1`（本机回环），不对外网暴露。

## 使用

1. **当日推荐**：页面加载即显示因子打分 Top3。
2. **我的持仓评估**：点“+ 添加一行”录入代码（如 `600519` 或 `600519.SSE`）、买入价、股数 → 点“模型评估持仓”。买入价/股数仅用于盈亏展示，**不影响建议**。
3. **数据回训**：点“数据回训”→ 后台 baostock 抓最新行情 → 重算因子 → 重训模型 → 刷新推荐。耗时约 **6–9 分钟**，期间页面轮询进度。

## 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 前端页面 |
| POST | `/api/retrain` | 触发后台回训（body `{"fetch": true}`）|
| GET | `/api/status` | 轮询回训进度 |
| GET | `/api/recommend` | 获取 Top3 推荐 |
| GET | `/api/resolve?code=<代码>` | 归一化代码并查名称（前端录入即时校验）|
| POST | `/api/evaluate` | 持仓评估（body `{"holdings": [{"vt_symbol","buy_price","volume"}, ...]}`）|

## 持有天数口径

- 下限 = 策略最短持有期 `MIN_DAYS`（默认 3 天）
- 上限 = 轮动周期 `ceil(TOP_K / N_DROP)`（默认 20/2 = 10 天）
- 多视野模型（`HORIZONS = [2,5,10,20]` 交易日）给出日均收益最优窗口估计

可调参数集中在 `pipeline.py` 顶部：`TOP_K / N_DROP / MIN_DAYS / HORIZONS` 等。

## 常见问题

- **改了 `templates/index.html` 但页面没变**：服务以 `debug=False` 运行，模板被缓存——**重启 `app.py`** 后再硬刷新浏览器（macOS `Cmd+Shift+R` / Windows `Ctrl+F5`）。
- **端口 5000 被占用**：macOS 关掉“系统设置 → 通用 → 隔空播放接收器”，或改 `app.py` 末尾 `port=5000` 为其他端口。
- **`pip install ta-lib` 失败**：说明 TA-Lib C 库未就绪，按上方对应平台先装 C 库（Windows 推荐 Conda）。
- **回训报数据源限流/超时**：baostock 偶发限流，稍后重试即可。
