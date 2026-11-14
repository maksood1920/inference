"""Serve the lab's continuous batching engine on localhost."""
import argparse
from concurrent.futures import Future
from errno import EADDRINUSE
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from queue import Empty, Full, Queue
from threading import Thread
from time import perf_counter

import torch
from transformers import AutoTokenizer

from engine import QwenEngine, _Request
from run import MODEL, ROOT


class InferenceWorker:
    """One worker owns model execution. HTTP threads submit jobs and wait."""

    def __init__(self, engine, tokenizer, eos_ids):
        self.engine = engine
        self.tokenizer = tokenizer
        self.eos_ids = eos_ids
        self.queue = Queue(maxsize=16)
        Thread(target=self.run, daemon=True).start()

    def submit(self, prompt, limit, received_at, updates=None):
        reply = Future()
        self.queue.put_nowait((prompt, limit, received_at, reply, updates))
        return reply

    def run(self):
        active = []
        pending_replies = {}
        round_number = 0
        while True:
            admitted = []
            for _ in range(2 - len(active)):
                try:
                    prompt, limit, received_at, reply, updates = self.queue.get(
                        block=not active and not admitted)
                except Empty:
                    break
                try:
                    rendered = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False, add_generation_prompt=True)
                    ids = self.tokenizer(rendered, return_tensors="pt",
                                         add_special_tokens=False)["input_ids"][0]
                    ids = ids.to(self.engine.device)
                    self.engine._validate_batch_prompts([ids], limit)
                except Exception as error:
                    reply.set_exception(error)
                    continue
                request = _Request(ids, limit)
                pending_replies[id(request)] = (received_at, reply, updates)
                admitted.append(request)
            if not active and not admitted:
                continue
            current = active + admitted
            try:
                active = self.engine._generation_round(
                    active, admitted, self.eos_ids, self.tokenizer.pad_token_id, round_number)
            except Exception as error:
                for request in current:
                    _, reply, _ = pending_replies.pop(id(request))
                    reply.set_exception(error)
                active = []
            else:
                for request in current:
                    received_at, reply, updates = pending_replies[id(request)]
                    if request.finished_at < 0:
                        if updates is not None:
                            updates.put_nowait(tuple(request.output))
                        continue
                    del pending_replies[id(request)]
                    try:
                        reply.set_result({
                            "text": self.tokenizer.decode(request.output, skip_special_tokens=True),
                            "output_ids": request.output,
                            "stop": "eos" if request.pending in self.eos_ids else "token_limit",
                            "admitted_round": request.admitted_at,
                            "finished_round": request.finished_at,
                            "time_to_first_token_ms": 1000 * (request.first_token_at - received_at),
                            "total_ms": 1000 * (perf_counter() - received_at),
                        })
                    except Exception as error:
                        reply.set_exception(error)
            round_number += 1


def make_handler(worker):
    page = (ROOT / "generate.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, payload):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path not in ("/", "/generate"):
                self.send_json(404, {"error": "Page not found"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)

        def send_stream(self, reply, updates):
            # HTTP/1.0 uses connection close to delimit a response of unknown size.
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

            def send_event(name, payload):
                self.wfile.write(f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode())
                self.wfile.flush()

            reply.add_done_callback(lambda _: updates.put_nowait(None))
            previous = ""
            try:
                while (tokens := updates.get()) is not None:
                    # Decode the prefix together so split UTF-8 characters can resolve.
                    text = worker.tokenizer.decode(list(tokens), skip_special_tokens=True)
                    text = text.rstrip("\ufffd")
                    if text != previous:
                        send_event("text", {"text": text})
                        previous = text
                send_event("done", reply.result())
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as error:
                message = str(error) if isinstance(error, ValueError) else "Generation failed"
                if not isinstance(error, ValueError):
                    logging.exception("Streaming generation failed")
                try:
                    send_event("error", {"error": message})
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def do_POST(self):
            received_at = perf_counter()
            if self.path != "/generate":
                self.send_json(404, {"error": "Use POST /generate"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 1 <= length <= 16384:
                    raise ValueError("Request body must be 1 to 16384 bytes")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("Expected a JSON object")
                prompt = body.get("prompt")
                limit = body.get("max_new_tokens", 32)
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError("prompt must be a nonempty string")
                if type(limit) is not int or not 1 <= limit <= 256:
                    raise ValueError("max_new_tokens must be an integer from 1 to 256")
                stream = body.get("stream", False)
                if not isinstance(stream, bool):
                    raise ValueError("stream must be a boolean")
                updates = Queue() if stream else None
                reply = worker.submit(prompt, limit, received_at, updates)
                if not stream:
                    result = reply.result()
            except ValueError as error:
                self.send_json(400, {"error": str(error)})
            except Full:
                self.send_json(503, {"error": "Inference queue is full"})
            except Exception:
                logging.exception("Generation failed")
                self.send_json(500, {"error": "Generation failed"})
            else:
                if stream:
                    self.send_stream(reply, updates)
                else:
                    self.send_json(200, result)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS unavailable")

    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), BaseHTTPRequestHandler)
    except OSError as error:
        if error.errno != EADDRINUSE:
            raise
        parser.error(f"Port {args.port} is already in use. Stop the existing server "
                     "or use --port 0 to select a free port.")

    with server:
        lock = json.loads((ROOT / "model-lock.json").read_text())
        if lock["model_id"] != MODEL:
            raise ValueError("Unexpected model lock")
        snapshot = ROOT / ".cache/huggingface/hub" / ("models--" + MODEL.replace("/", "--")) / "snapshots" / lock["revision"]
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        dtype = torch.float16 if args.device == "mps" else torch.float32
        engine = QwenEngine.from_snapshot(snapshot, args.device, dtype)
        eos = json.loads((snapshot / "generation_config.json").read_text())["eos_token_id"]
        worker = InferenceWorker(engine, tokenizer, [eos] if isinstance(eos, int) else eos)
        server.RequestHandlerClass = make_handler(worker)
        print(f"Listening on http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
