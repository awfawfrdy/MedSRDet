#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_vindr.py — PhysioNet 凭据下载脚本 (MedSRDet 复现工具)

用途:
    在 (1) 你的 PhysioNet 凭据申请已批准 且 (2) 你已签署 VinDr-CXR 1.0.0 的 DUA 之后,
    用个人访问令牌 (Personal Access Token) 把 VinDr-CXR 数据下到本地, 供
    scripts/gen_vindr_manifest.py 生成真实的 7:1:2 划分 manifest 使用。

凭据获取路径 (只能本人操作):
    PhysioNet 网站 -> 账号 Settings -> Manage Personal Access Tokens -> 新建 token

为什么默认只下标注 CSV:
    VinDr-CXR 公开层只暴露匿名 image_id, 不暴露 patient_id, 因此 V 的划分只能是 image-level。
    gen_vindr_manifest.py 只需要 annotations_train.csv / annotations_test.csv 里的 image_id 列表来
    做固定 7:1:2 划分, 不需要 ~18GB 的影像。故本脚本默认 --annotations-only, 几分钟即可。

用法:
    # 仅标注 (生成 manifest 够用, 推荐)
    python download_vindr.py --token $PHYSIO_TOKEN --out ./vindr-cxr-1.0.0

    # 完整数据集 (影像 + 标注, ~18GB, 较慢)
    python download_vindr.py --token $PHYSIO_TOKEN --out ./vindr-cxr-1.0.0 --all

依赖:
    pip install requests tqdm

退出码:
    0  成功
    2  认证/授权失败 (凭据未批准 或 DUA 未签 或 token 无效)
"""
import argparse
import os
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("[DEP] 缺少 requests, 请先: pip install requests tqdm")

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

BASE = "https://physionet.org"
PROJECT = "vindr-cxr"
VERSION = "1.0.0"
ANNOTATION_FILES = ["annotations_train.csv", "annotations_test.csv"]
IMAGE_DIRS = ("train", "test", "image")  # 这些目录下是影像, 非 --all 时跳过


def _list_dir(token: str, rel: str) -> dict:
    url = f"{BASE}/api/files/{PROJECT}/{VERSION}/{rel}".rstrip("/") + "/"
    r = requests.get(url, params={"auth_token": token}, timeout=30)
    if r.status_code in (401, 403):
        sys.exit(
            "[AUTH] 凭据/DUA 未就绪或 token 无效 (HTTP %d)。\n"
            "       请确认: (a) 凭据申请已批准  (b) 已签 VinDr-CXR 1.0.0 的 DUA  (c) token 正确。"
            % r.status_code
        )
    if r.status_code == 404:
        sys.exit("[ERR] 路径不存在 (HTTP 404): %s" % url)
    r.raise_for_status()
    return r.json()


def _walk(token: str, rel: str, out: Path, want_all: bool, pbar):
    data = _list_dir(token, rel)
    for d in data.get("directories", []):
        if not want_all and d["name"] in IMAGE_DIRS:
            continue  # 跳过影像目录, 只拿标注
        _walk(token, f"{rel}/{d['name']}".strip("/"), out, want_all, pbar)
    for f in data.get("files", []):
        name = f["name"]
        if not want_all and name not in ANNOTATION_FILES:
            continue
        _download(token, f["url"], out)
        if pbar is not None:
            pbar.update(1)


def _download(token: str, url: str, out: Path):
    # url 形如 /files/vindr-cxr/1.0.0/annotations_train.csv
    rel = url.replace(f"/files/{PROJECT}/{VERSION}/", "").lstrip("/")
    dst = out / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    full = f"{BASE}{url}?auth_token={token}"
    try:
        with requests.get(full, stream=True, timeout=120) as r:
            if r.status_code in (401, 403):
                sys.exit("[AUTH] 下载被拒 (HTTP %d), 凭据/DUA/token 未通过。" % r.status_code)
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            mode = "wb"
            kwargs = {}
            if tqdm is not None and total:
                pbar_f = tqdm(total=total, unit="B", unit_scale=True, desc=rel, leave=False)
            else:
                pbar_f = None
            with open(dst, mode) as fh:
                for chunk in r.iter_content(1 << 16):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    if pbar_f is not None:
                        pbar_f.update(len(chunk))
            if pbar_f is not None:
                pbar_f.close()
    except requests.RequestException as e:
        sys.exit("[ERR] 下载失败 %s: %s" % (rel, e))


def main():
    ap = argparse.ArgumentParser(description="Download VinDr-CXR 1.0.0 from PhysioNet (credentialed).")
    ap.add_argument("--token", required=True, help="PhysioNet Personal Access Token")
    ap.add_argument("--out", default="./vindr-cxr-1.0.0", help="输出目录 (默认 ./vindr-cxr-1.0.0)")
    ap.add_argument("--all", action="store_true", help="下载完整数据集 (含影像, ~18GB); 默认仅标注 CSV")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    want_all = args.all

    print("[INFO] 项目 %s@%s  target=%s  mode=%s" % (
        PROJECT, VERSION, out, "FULL" if want_all else "ANNOTATIONS-ONLY"))
    if not want_all:
        print("[INFO] 仅下载 %s (生成 manifest 够用)" % ", ".join(ANNOTATION_FILES))

    start = time.time()
    _walk(args.token, "", out, want_all, None)
    print("[OK] 完成, 用时 %.1fs -> %s" % (time.time() - start, out))

    if not want_all:
        print("[NEXT] 拿到标注后运行:")
        print("      python scripts/gen_vindr_manifest.py "
              "--annotations %s\\annotations_train.csv %s\\annotations_test.csv" % (out, out))


if __name__ == "__main__":
    main()
