"""
ToMogo-QT Web 后端（Flask）。

仅绑定 127.0.0.1，作为本地研究工具，不做网络暴露。
回训耗时数分钟，用后台线程执行 + 前端轮询 /api/status。

研究演示用途，非实盘投资建议。
"""

import threading
from pathlib import Path

from flask import Flask, jsonify, request, render_template

import pipeline

app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))

# 全局回训状态（单用户本地工具，单一任务即可）
_state: dict = {"stage": "idle", "msg": "尚未开始", "running": False}
_lock = threading.Lock()


def _progress(stage: str, msg: str) -> None:
    with _lock:
        _state["stage"] = stage
        _state["msg"] = msg


def _retrain_worker(fetch: bool) -> None:
    try:
        pipeline.run_retrain(progress=_progress, fetch=fetch)
    except Exception as e:                                              # noqa: BLE001
        _progress("error", f"回训失败：{e}")
    finally:
        with _lock:
            _state["running"] = False


@app.route("/")
def index():
    return render_template("index.html")


@app.after_request
def add_no_cache(resp):
    # 本地研究工具：禁止浏览器缓存页面/接口，避免改版后仍加载旧 HTML/JS
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    with _lock:
        if _state["running"]:
            return jsonify({"ok": False, "msg": "回训已在进行中"}), 409
        _state["running"] = True
        _state["stage"] = "starting"
        _state["msg"] = "正在启动回训..."

    fetch = bool(request.json.get("fetch", True)) if request.is_json else True
    t = threading.Thread(target=_retrain_worker, args=(fetch,), daemon=True)
    t.start()
    return jsonify({"ok": True, "msg": "回训已开始"})


@app.route("/api/status")
def api_status():
    with _lock:
        return jsonify(dict(_state))


@app.route("/api/recommend")
def api_recommend():
    try:
        return jsonify(pipeline.get_recommendations(top_n=3))
    except Exception as e:                                              # noqa: BLE001
        return jsonify({"ok": False, "msg": f"读取推荐失败：{e}", "items": []}), 500


@app.route("/api/resolve")
def api_resolve():
    # 检索层：把用户输入的代码归一化并查本地名称缓存，供前端录入时即时校验/显示名称
    code = request.args.get("code", "")
    try:
        return jsonify(pipeline.resolve_symbol(code))
    except Exception as e:                                              # noqa: BLE001
        return jsonify({"ok": False, "vt_symbol": None, "name": None,
                        "known": False, "msg": f"解析失败：{e}"}), 500


@app.route("/api/evaluate", methods=["POST"])
def api_evaluate():
    if not request.is_json:
        return jsonify({"ok": False, "msg": "请求需为 JSON", "items": []}), 400
    holdings = request.json.get("holdings", [])
    if not isinstance(holdings, list) or not holdings:
        return jsonify({"ok": False, "msg": "请至少录入一条持仓", "items": []}), 400
    try:
        return jsonify(pipeline.evaluate_holdings(holdings))
    except Exception as e:                                              # noqa: BLE001
        return jsonify({"ok": False, "msg": f"评估失败：{e}", "items": []}), 500


if __name__ == "__main__":
    # 仅监听本地回环，避免网络暴露
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
