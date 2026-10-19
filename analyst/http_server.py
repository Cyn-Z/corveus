import os
import json
import re
import urllib.request
import analyst
import orchestrator
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from analyst import start_prompt, resolve_selection, get_llama_stats

# HTTP Request Handler for both UI and API
WEB_DIR = Path(__file__).parent / "web"


class OrchestratorHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.serve_file(WEB_DIR / "index.html", "text/html; charset=utf-8")
        elif self.path == "/style.css":
            self.serve_file(WEB_DIR / "style.css", "text/css; charset=utf-8")
        elif self.path == "/app.js":
            self.serve_file(WEB_DIR / "app.js", "application/javascript; charset=utf-8")
        else:
            self.send_error(404, "Not Found")

    def serve_file(self, file_path, content_type):
        try:
            content = file_path.read_bytes()

            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()

            self.wfile.write(content)

        except FileNotFoundError:
            self.send_error(404, "File Not Found")

        except Exception as e:
            print(f"[Static File Error] {e}")
            self.send_error(500, "Internal Server Error")

    '''def do_POST(self):

        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return

        try:
            # Read and parse the incoming JSON request.
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode("utf-8"))

            messages = data.get("messages", [])

            # Find the most recent user message.
            user_prompt = next(
                (
                    message.get("content", "")
                    for message in reversed(messages)
                    if message.get("role") == "user"
                ),
                "",
            )

            # Reject requests without a usable text prompt.
            if not isinstance(user_prompt, str) or not user_prompt.strip():
                error_data = json.dumps(
                    {"error": "A non-empty user message is required."}
                ).encode("utf-8")

                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(error_data)))
                self.end_headers()
                self.wfile.write(error_data)
                return

            print(
                f"\n[UI Prompt Received] -> {user_prompt}",
                flush=True,
            )

            # Delegate skill selection and LLM processing to analyst.py.
            result = analyst.process_prompt(user_prompt)
            stats = analyst.get_llama_stats()

            # Preserve the response structure expected by the UI.
            response_payload = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": result,
                        }
                    }
                ],
                "stats": stats,
            }

            response_data = json.dumps(response_payload).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_data)))
            self.end_headers()
            self.wfile.write(response_data)

        except Exception as e:
            # Log the detailed error on the server.
            print(f"[Error] {e}", flush=True)

            error_data = json.dumps(
                {"error": "Failed to process chat request."}
            ).encode("utf-8")

            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(error_data)))
            self.end_headers()
            self.wfile.write(error_data)
'''

    def do_POST(self):
        if self.path not in (
            "/v1/chat/completions",
            "/v1/chat/selection",
        ):
            self.send_error(404, "Not Found")
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))

            if content_length <= 0 or content_length > 1_000_000:
                self.send_json(400, {"error": "Invalid request size."})
                return

            data = json.loads(self.rfile.read(content_length).decode("utf-8"))

            if self.path == "/v1/chat/completions":
                messages = data.get("messages", [])

                user_prompt = next(
                    (
                        message.get("content", "")
                        for message in reversed(messages)
                        if message.get("role") == "user"
                    ),
                    "",
                )

                if not isinstance(user_prompt, str) or not user_prompt.strip():
                    self.send_json(400, {
                        "error": "A non-empty user message is required."
                    })
                    return

                result = start_prompt(user_prompt.strip())

            else:
                result = resolve_selection(
                    data.get("request_id"),
                    data.get("remove_skill_names"),
                )

            if result.get("selection_required"):
                self.send_json(200, result)
                return

            self.send_json(200, {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": result["content"],
                    }
                }],
                "stats": result["stats"],
                "selection_required": False,
            })

        except Exception as e:
            print(f"[Error] {e}", flush=True)

            error_data = json.dumps({
                "error": str(e) if isinstance(e, ValueError)
                else "Failed to process chat request."
            }).encode("utf-8")

            self.send_response(
                400 if isinstance(e, ValueError) else 500
            )
            self.send_header(
                "Content-Type",
                "application/json; charset=utf-8"
            )
            self.send_header("Content-Length", str(len(error_data)))
            self.end_headers()
            self.wfile.write(error_data)

    def send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")

        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 5000), OrchestratorHandler)
    print("=" * 60)
    print(f" SECURE UI & ORCHESTRATOR RUNNING ON PORT 5000 ")

    print("=" * 60)
    server.serve_forever()
