#!/usr/bin/env python3
"""Run the strict top-k/throughput gate with a managed llama-server."""

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--llama-server", default="llama-server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8027)
    p.add_argument("--ctx", type=int, default=512)
    p.add_argument("--ngl", type=int, default=99)
    p.add_argument("--np", type=int, default=1)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--log", default="/tmp/bitnet-llama-topk-server.log")
    args, compare_args = p.parse_known_args()
    if compare_args and compare_args[0] == "--":
        compare_args = compare_args[1:]
    args.compare_args = compare_args
    return args


def choose_free_port(host):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def wait_for_server(url, timeout, proc=None):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=2) as resp:
                if 200 <= resp.status < 500:
                    return
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
        time.sleep(0.5)
    raise TimeoutError(f"llama-server did not become ready: {last_error}")


def print_log_tail(path, lines=80):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            tail = f.readlines()[-lines:]
    except OSError:
        return
    if tail:
        print("--- llama-server log tail ---", file=sys.stderr)
        for line in tail:
            print(line.rstrip(), file=sys.stderr)


def main():
    args = parse_args()
    compare_args = list(args.compare_args)
    if args.port == 0:
        args.port = choose_free_port(args.host)

    server_url = f"http://{args.host}:{args.port}"
    cmd = [
        args.llama_server,
        "-m", args.model,
        "-ngl", str(args.ngl),
        "-fa", "on",
        "-c", str(args.ctx),
        "-np", str(args.np),
        "--host", args.host,
        "--port", str(args.port),
    ]

    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    with open(args.log, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            wait_for_server(server_url, args.timeout, proc)
            compare_cmd = [
                sys.executable,
                "test/compare_llama_topk.py",
                args.model,
                "--llama-server-url",
                server_url,
            ] + compare_args
            rc = subprocess.call(compare_cmd)
            if rc != 0:
                print_log_tail(args.log)
            return rc
        except Exception:
            print_log_tail(args.log)
            raise
        finally:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except OSError:
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    pass
                proc.wait()


if __name__ == "__main__":
    raise SystemExit(main())
