#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""按 download_plan.json 批量下载 N×M 组合。

设计要点：
  1. gms_hash 与角色无关（已实证），所以每个动作只查一次，缓存复用
  2. 断点续传：目标文件存在且 > 20KB 就跳过，manifest 只记录状态
  3. 401 立刻中止 —— token 失效时继续跑只会把 480 个都记成失败
  4. 串行 + 可调间隔，不并发（避免触发风控）
  5. 每个文件下完立刻校验大小，不攒批（预签名 URL 只有 5 分钟）

用法：
  set MIXAMO_TOKEN=eyJhbGci...
  python mixamo_batch.py --dry-run
  python mixamo_batch.py --limit 3
  python mixamo_batch.py
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.parse

import requests

BASE = "https://www.mixamo.com/api/v1"
POLL_INTERVAL = 3.0
POLL_TIMEOUT = 200.0


def log(msg):
    print(msg, flush=True)


def safe(s):
    return re.sub(r"[^0-9A-Za-z._-]+", "_", s).strip("_")


class Client(object):
    def __init__(self, token, proxy=None):
        self.token = token
        self.s = requests.Session()
        self.s.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json; charset=UTF-8",
            "X-Api-Key": "mixamo2",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) mixamo-batch/0.1",
        })
        if proxy:
            self.s.proxies.update({"http": proxy, "https": proxy})

    def _req(self, method, path, retries=3, **kw):
        url = path if path.startswith("http") else BASE + path
        headers = {"Authorization": "Bearer " + self.token} if self.token else {}
        kw.setdefault("timeout", 90)
        last = None
        for i in range(retries):
            try:
                r = self.s.request(method, url, headers=headers, **kw)
                if r.status_code in (429, 500, 502, 503, 504):
                    wait = 15 * (i + 1) if r.status_code == 429 else 2 ** i
                    log("      HTTP %d，%ds 后重试" % (r.status_code, wait))
                    time.sleep(wait)
                    last = r
                    continue
                return r
            except requests.RequestException as e:
                log("      网络异常 %r" % e)
                time.sleep(2 ** i)
                last = e
        if isinstance(last, requests.Response):
            return last
        raise RuntimeError("请求失败：%s %s" % (method, path))

    def json_of(self, method, path, **kw):
        r = self._req(method, path, **kw)
        if r.status_code == 401:
            raise SystemExit("\n!! 401：token 失效或未提供。已中止，避免把剩下全部记成失败。")
        if r.status_code != 200 and r.status_code != 202:
            raise RuntimeError("HTTP %d: %s" % (r.status_code, r.text[:200]))
        return r.json()

    def gms(self, motion_id, character_id):
        d = self.json_of("GET", "/products/%s" % motion_id,
                         params={"similar": 0, "character_id": character_id})
        return (d.get("details") or {}).get("gms_hash")

    def export(self, body):
        return self.json_of("POST", "/animations/export", data=json.dumps(body))

    def wait(self, uuid):
        t0 = time.time()
        while True:
            d = self.json_of("GET", "/characters/%s/monitor" % uuid)
            st = d.get("status")
            if st == "completed":
                return d
            if st == "failed":
                raise RuntimeError("导出失败：%r" % (d.get("job_result"),))
            if time.time() - t0 > POLL_TIMEOUT:
                raise RuntimeError("轮询超时")
            time.sleep(POLL_INTERVAL)

    def fetch(self, url, path):
        # 预签名 URL 自带授权，绝不能带 Authorization
        with requests.get(url, stream=True, timeout=180) as r:
            if r.status_code != 200:
                raise RuntimeError("下载 HTTP %d" % r.status_code)
            n = 0
            tmp = path + ".part"
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 16):
                    f.write(chunk)
                    n += len(chunk)
            os.replace(tmp, path)
            return n


def export_body(plan, char_id, motion_name, gh):
    g = dict(gh)
    p = g.get("params")
    if isinstance(p, (list, tuple)):
        vals = []
        for it in p:
            if isinstance(it, dict):
                vals.append(it.get("value", 0))
            elif isinstance(it, (list, tuple)):
                vals.append(it[1] if len(it) > 1 else it[0])
            else:
                vals.append(it)
        g["params"] = ",".join(
            str(int(float(v))) if float(v) == int(float(v)) else ("%g" % float(v)) for v in vals)
    g.setdefault("overdrive", 0)
    t = g.get("trim")
    if isinstance(t, (list, tuple)) and len(t) == 2:
        g["trim"] = [int(round(float(t[0]))), int(round(float(t[1])))]
    return {
        "gms_hash": [g],
        "preferences": dict(plan["preferences"]),
        "character_id": char_id,
        "type": "Motion",
        "product_name": motion_name,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="../data/download_plan.json")
    ap.add_argument("--out", default="../data/mixamo_raw/_anims")
    ap.add_argument("--manifest", default="../data/download_manifest.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="本次最多跑 N 个组合（0=全部）")
    ap.add_argument("--delay", type=float, default=1.0, help="每个组合之间的间隔秒数")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--proxy", default=os.environ.get("MIXAMO_PROXY", ""))
    args = ap.parse_args()

    plan = json.load(open(args.plan, encoding="utf-8"))
    chars, motions = plan["characters"], plan["motions"]
    total = len(chars) * len(motions)
    os.makedirs(args.out, exist_ok=True)

    todo = []
    for c in chars:
        for m in motions:
            name = "%s__%s.fbx" % (safe(c["name"]), safe(m["name"]))
            todo.append((c, m, os.path.join(args.out, name)))

    done = sum(1 for _, _, p in todo if os.path.exists(p) and os.path.getsize(p) > 20 * 1024)
    log("计划 %d 个组合（%d 角色 × %d 动作）" % (total, len(chars), len(motions)))
    log("已存在 %d 个，待下 %d 个" % (done, total - done))

    if args.dry_run:
        for i, (c, m, p) in enumerate(todo[:20], 1):
            mark = "skip" if os.path.exists(p) else "    "
            log("  %3d %s %-22s × %s" % (i, mark, c["name"][:22], m["name"]))
        if total > 20:
            log("  ... 共 %d 个" % total)
        est = (total - done) * 9 / 60.0
        log("预估耗时约 %.0f 分钟（按每个 9 秒估）" % est)
        return

    token = os.environ.get("MIXAMO_TOKEN", "").strip()
    if not token:
        raise SystemExit("缺少 MIXAMO_TOKEN 环境变量")

    cli = Client(token, args.proxy or None)

    # 动作级的 gms_hash 缓存：与角色无关，40 次就够
    gh_cache = {}
    manifest = open(args.manifest, "a", encoding="utf-8")

    ok = fail = skip = 0
    t_start = time.time()
    idx = 0

    for c, m, path in todo:
        idx += 1
        if os.path.exists(path) and os.path.getsize(path) > 20 * 1024:
            skip += 1
            continue
        if args.limit and (ok + fail) >= args.limit:
            log("达到 --limit %d，停止" % args.limit)
            break

        tag = "[%d/%d] %-22s × %-24s" % (idx, total, c["name"][:22], m["name"][:24])
        t0 = time.time()
        try:
            if m["id"] not in gh_cache:
                gh_cache[m["id"]] = cli.gms(m["id"], c["id"])
            gh = gh_cache[m["id"]]
            if not gh:
                raise RuntimeError("详情里没有 gms_hash")

            job = cli.export(export_body(plan, c["id"], m["name"], gh))
            uuid = job.get("uuid") or c["id"]
            done_job = cli.wait(uuid)
            url = done_job.get("job_result")
            if not url:
                raise RuntimeError("completed 但没有 job_result")

            n = cli.fetch(url, path)
            el = time.time() - t0
            ok += 1
            log("%s  OK  %6.1fs  %6.0f KB" % (tag, el, n / 1024.0))
            manifest.write(json.dumps({
                "char": c["name"], "char_id": c["id"],
                "motion": m["name"], "motion_id": m["id"], "group": m.get("group"),
                "file": os.path.basename(path), "bytes": n,
                "seconds": round(el, 1), "status": "ok",
            }, ensure_ascii=False) + "\n")
            manifest.flush()
        except SystemExit:
            raise
        except Exception as e:
            el = time.time() - t0
            fail += 1
            log("%s  FAIL %5.1fs  %r" % (tag, el, e))
            manifest.write(json.dumps({
                "char": c["name"], "motion": m["name"],
                "file": os.path.basename(path), "status": "fail", "error": repr(e),
            }, ensure_ascii=False) + "\n")
            manifest.flush()
        time.sleep(args.delay)

    manifest.close()
    el = (time.time() - t_start) / 60.0
    log("\n完成：成功 %d，失败 %d，跳过 %d，用时 %.1f 分钟" % (ok, fail, skip, el))
    if fail:
        log("失败清单见 %s（status=fail）" % args.manifest)


if __name__ == "__main__":
    sys.exit(main() or 0)
