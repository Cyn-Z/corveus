import os
import json
import re
import urllib.request
import analyst
import orchestrator
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from analyst import start_prompt, resolve_selection, get_llama_stats
from dataclasses import asdict
from service_api import LocalJobAPI
from service_contracts import ErrorCode, ServiceError, MAX_PROMPT_BYTES
from memory_store import (
    DEFAULT_CONVERSATION_TITLE,
    ConversationNotFoundError,
)

from urllib.parse import parse_qs, urlsplit

# HTTP Request Handler for both UI and API
WEB_DIR = Path(__file__).parent / "web"


class OrchestratorHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/v1/job-capabilities":
            api = getattr(getattr(self, "server", None), "job_api", None)
            self.send_json(200, {"enabled": isinstance(api, LocalJobAPI)})
            return
        if self._handle_job_request("GET"):
            return
        if self.path == "/v1/status":
            self.component_status()
            return
        if self._handle_conversation_request("GET"):
            return

        if self.path in ("/", "/index.html"):
            self.serve_file(WEB_DIR / "index.html", "text/html; charset=utf-8")
        elif self.path == "/style.css":
            self.serve_file(WEB_DIR / "style.css", "text/css; charset=utf-8")
        elif self.path == "/jobs-client.js":
            self.serve_file(
                WEB_DIR / "jobs-client.js",
                "application/javascript; charset=utf-8",
            )
        elif self.path == "/app.js":
            self.serve_file(WEB_DIR / "app.js", "application/javascript; charset=utf-8")
        elif self.path == "/shell.js":
            self.serve_file(
                WEB_DIR / "shell.js", "application/javascript; charset=utf-8"
            )
        elif re.fullmatch(r"/corveus-assets/[a-z][a-z-]*\.svg", self.path):
            self.serve_file(WEB_DIR / self.path.lstrip("/"), "image/svg+xml")
        else:
            self.send_error(404, "Not Found")

    def component_status(self):
        components = {
            "analyst": "online",
            "router": "offline",
            "model": "offline",
            "persistence": "offline",
        }
        api = getattr(getattr(self, "server", None), "job_api", None)
        if api is not None:
            try:
                if api.router_available():
                    components["router"] = "online"
            except Exception:
                pass
        elif getattr(analyst, "SKILL_ROUTER", None) is not None:
            components["router"] = "online"
        manager = analyst.MEMORY_MANAGER
        if manager.enabled:
            try:
                manager.list_conversations(limit=1)
                components["persistence"] = "online"
            except Exception:
                pass
        else:
            components["persistence"] = "disabled"
        try:
            url = urlsplit(analyst.LLM_URL)
            health_url = f"{url.scheme}://{url.netloc}/health"
            with urllib.request.urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    components["model"] = "online"
        except Exception:
            pass
        self.send_json(
            200,
            {
                "components": components,
                "model": {
                    "name": getattr(
                        getattr(analyst, "settings", None), "model_name", None
                    ),
                    "runtime": "llama.cpp",
                },
            },
        )

    def do_DELETE(self):
        if self._handle_job_request("DELETE"):
            return
        if not self._handle_conversation_request("DELETE"):
            self.send_json(404, {"error": "Not Found"})

    def _handle_job_request(self, method):
        """Serve the explicitly attached local job API, without running inference."""
        try:
            parsed = urlsplit(self.path)
        except ValueError:
            self.send_json(400, ServiceError(ErrorCode.INVALID_REQUEST).as_payload())
            return True
        if parsed.path != "/v1/jobs" and not parsed.path.startswith("/v1/jobs/"):
            return False
        try:
            if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
                raise ServiceError(ErrorCode.INVALID_REQUEST)
            match = re.fullmatch(
                r"/v1/jobs/([0-9a-f]{32})(?:/(result|cancel|review))?",
                parsed.path,
            )
            if parsed.path == "/v1/jobs":
                route, job_id, allowed = "submit", None, ("POST",)
            elif match is not None:
                job_id = match.group(1)
                route = match.group(2) or "status"
                if route == "review":
                    allowed = ("GET", "POST")
                elif route == "cancel":
                    allowed = ("POST",)
                else:
                    allowed = ("GET",)
            else:
                raise ServiceError(ErrorCode.NOT_FOUND)
            if method not in allowed:
                self.send_json(405, {"error": {"code": "method_not_allowed"}})
                return True
            api = getattr(getattr(self, "server", None), "job_api", None)
            if not isinstance(api, LocalJobAPI):
                raise ServiceError(ErrorCode.SERVICE_UNAVAILABLE)
            # Reject ambiguous framing and duplicate headers before reading.
            lengths = self.headers.get_all("Content-Length", [])
            if self.headers.get_all("Transfer-Encoding", []) or len(lengths) > 1:
                raise ServiceError(ErrorCode.INVALID_REQUEST)
            raw_length = lengths[0] if lengths else "0"
            if re.fullmatch(r"[0-9]{1,10}", raw_length) is None:
                raise ServiceError(ErrorCode.INVALID_REQUEST)
            length = int(raw_length)
            if method == "GET":
                if length != 0:
                    raise ServiceError(ErrorCode.INVALID_REQUEST)
                if route == "result":
                    payload = api.result(job_id)
                elif route == "review":
                    payload = api.inspect_review(job_id)
                else:
                    payload = api.status(job_id)
                self.send_json(200, payload)
                return True
            if route == "submit":
                limit = MAX_PROMPT_BYTES + 4096
            elif route == "review":
                limit = 65536
            else:
                limit = 1024
            if not 0 < length <= limit:
                raise ServiceError(ErrorCode.INVALID_REQUEST)
            if self.headers.get_content_type() != "application/json":
                raise ServiceError(ErrorCode.INVALID_REQUEST)
            body = self.rfile.read(length)
            if len(body) != length:
                raise ServiceError(ErrorCode.INVALID_REQUEST)
            data = json.loads(body.decode("utf-8"))
            if route == "submit":
                keys = self.headers.get_all("Idempotency-Key", [])
                if len(keys) != 1:
                    raise ServiceError(ErrorCode.INVALID_REQUEST)
                payload = api.submit(data, idempotency_key=keys[0])
                self.send_json(202, payload)
            elif route == "review":
                self.send_json(200, api.apply_review(job_id, data))
            else:
                self.send_json(200, api.cancel(job_id, data))
        except (ValueError, UnicodeError):
            self.close_connection = True
            self.send_json(400, ServiceError(ErrorCode.INVALID_REQUEST).as_payload())
        except ServiceError as exc:
            self.close_connection = True
            self.send_json(exc.status_code, exc.as_payload())
        except Exception as exc:
            self.close_connection = True
            print(f"[Job API Error] {type(exc).__name__}", flush=True)
            self.send_json(500, ServiceError(ErrorCode.INTERNAL_ERROR).as_payload())
        return True

    @staticmethod
    def _conversation_query(query, allowed_names):
        values = parse_qs(
            query,
            keep_blank_values=True,
            strict_parsing=True,
            max_num_fields=8,
        )

        if set(values) - set(allowed_names):
            raise ValueError("Unsupported query parameter.")

        if any(len(items) != 1 for items in values.values()):
            raise ValueError("Duplicate query parameters are not allowed.")

        return {name: items[0] for name, items in values.items()}

    @staticmethod
    def _conversation_integer(values, name, default, minimum, maximum):
        text = values.get(name)

        if text is None:
            return default

        if len(text) > 10 or re.fullmatch(r"[0-9]+", text) is None:
            raise ValueError(
                f"{name} must be an integer between " f"{minimum} and {maximum}."
            )

        value = int(text)

        if not minimum <= value <= maximum:
            raise ValueError(
                f"{name} must be an integer between " f"{minimum} and {maximum}."
            )

        return value

    def _handle_conversation_request(self, method):
        """Return True when this method has handled a memory API route."""
        try:
            parsed = urlsplit(self.path)
        except ValueError:
            self.send_json(400, {"error": "Invalid request URL."})
            return True

        path = parsed.path

        if not (path == "/v1/conversations" or path.startswith("/v1/conversations/")):
            return False

        try:
            if parsed.scheme or parsed.netloc or parsed.fragment:
                raise ValueError("Invalid request URL.")

            suffix = path[len("/v1/conversations") :]
            conversation_id = None

            if suffix == "":
                route = "list"
            else:
                match = re.fullmatch(
                    r"/([0-9a-f]{32})(?:/(messages|usage))?",
                    suffix,
                )

                if match is None:
                    self.send_json(404, {"error": "Not Found"})
                    return True

                conversation_id = match.group(1)
                route = match.group(2) or "conversation"

            if method == "DELETE" and route != "conversation":
                self.send_json(405, {"error": "Method not allowed."})
                return True

            manager = analyst.MEMORY_MANAGER

            if not manager.enabled:
                self.send_json(
                    409,
                    {"error": "Conversation persistence is disabled."},
                )
                return True

            if method == "DELETE":
                self._conversation_query(parsed.query, ())

                if self.headers.get("Transfer-Encoding") is not None:
                    raise ValueError("Conversation deletion does not accept a body.")

                if int(self.headers.get("Content-Length", "0")) != 0:
                    raise ValueError("Conversation deletion does not accept a body.")

                deleted = manager.delete_conversation(conversation_id)

                if not deleted:
                    raise RuntimeError("Conversation deletion was unavailable.")

                self.send_json(
                    200,
                    {
                        "conversation_id": conversation_id,
                        "deleted": True,
                    },
                )
                return True

            if route == "list":
                values = self._conversation_query(parsed.query, ("limit", "offset"))
                limit = self._conversation_integer(values, "limit", 100, 1, 1000)
                offset = self._conversation_integer(values, "offset", 0, 0, 2147483647)

                conversations = manager.list_conversations(
                    limit=limit,
                    offset=offset,
                )

                self.send_json(
                    200,
                    {
                        "conversations": [asdict(item) for item in conversations],
                        "limit": limit,
                        "offset": offset,
                    },
                )
                return True

            if route == "usage":
                self._conversation_query(parsed.query, ())
                usage = manager.get_conversation_usage(conversation_id)
                if usage is None:
                    raise RuntimeError("Conversation usage was unavailable.")
                self.send_json(200, {"usage": usage})
                return True

            if route == "messages":
                values = self._conversation_query(parsed.query, ("recent_exchanges",))
                recent_exchanges = self._conversation_integer(
                    values, "recent_exchanges", 50, 1, 1000
                )

                messages = manager.get_messages(
                    conversation_id,
                    recent_exchanges=recent_exchanges,
                )

                self.send_json(
                    200,
                    {
                        "conversation_id": conversation_id,
                        "messages": [asdict(item) for item in messages],
                        "recent_exchanges": recent_exchanges,
                    },
                )
                return True

            self._conversation_query(parsed.query, ())
            conversation = manager.get_conversation(conversation_id)

            if conversation is None:
                raise RuntimeError("Conversation inspection was unavailable.")

            self.send_json(200, {"conversation": asdict(conversation)})
            return True

        except ConversationNotFoundError:
            self.send_json(
                404,
                {"error": ("Conversation was not found in the configured scope.")},
            )

        except ValueError as exc:
            self.send_json(400, {"error": str(exc)})

        except Exception as exc:
            print(
                f"[Conversation API Error] {type(exc).__name__}",
                flush=True,
            )
            self.send_json(
                500,
                {"error": "Failed to process conversation request."},
            )

        return True

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

    def do_POST(self):
        if self._handle_job_request("POST"):
            return
        if getattr(getattr(self, "server", None), "job_api", None) is not None and self.path in (
            "/v1/chat/completions", "/v1/chat/selection",
        ):
            self.close_connection = True
            self.send_json(409, ServiceError(ErrorCode.CONFLICT).as_payload())
            return
        title_match = re.fullmatch(
            r"/v1/conversations/([0-9a-f]{32})/title",
            self.path,
        )

        if title_match is None and self.path not in (
            "/v1/chat/completions",
            "/v1/chat/selection",
            "/v1/conversations",
        ):
            self.send_error(404, "Not Found")
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))

            if content_length <= 0 or content_length > 1_000_000:
                self.send_json(400, {"error": "Invalid request size."})
                return

            data = json.loads(self.rfile.read(content_length).decode("utf-8"))

            if not isinstance(data, dict):
                raise ValueError("Request body must be a JSON object.")

            if title_match is not None:
                if set(data) != {"title"}:
                    raise ValueError("Renaming requires only a title field.")

                if not analyst.MEMORY_MANAGER.enabled:
                    self.send_json(
                        409,
                        {"error": "Conversation persistence is disabled."},
                    )
                    return

                conversation = analyst.MEMORY_MANAGER.rename_conversation(
                    title_match.group(1),
                    title=data["title"],
                )

                if conversation is None:
                    raise RuntimeError("Conversation renaming was unavailable.")

                self.send_json(200, {"conversation": asdict(conversation)})
                return

            if self.path == "/v1/conversations":
                if set(data) - {"title"}:
                    raise ValueError(
                        "Conversation creation accepts only a title field. "
                        "Scope is supplied by server configuration."
                    )
                if not analyst.MEMORY_MANAGER.enabled:
                    self.send_json(
                        409,
                        {"error": "Conversation persistence is disabled."},
                    )
                    return

                conversation = analyst.MEMORY_MANAGER.create_conversation(
                    title=data.get("title", DEFAULT_CONVERSATION_TITLE)
                )

                if conversation is None:
                    raise RuntimeError("Conversation creation was unavailable.")

                self.send_json(201, {"conversation": asdict(conversation)})
                return

            if self.path == "/v1/chat/completions":
                messages = data.get("messages", [])

                if not isinstance(messages, list) or not all(
                    isinstance(message, dict) for message in messages
                ):
                    raise ValueError("messages must be an array of message objects.")

                user_prompt = next(
                    (
                        message.get("content", "")
                        for message in reversed(messages)
                        if message.get("role") == "user"
                    ),
                    "",
                )

                if not isinstance(user_prompt, str) or not user_prompt.strip():
                    raise ValueError("A non-empty user message is required.")

                result = start_prompt(
                    user_prompt.strip(),
                    conversation_id=data.get("conversation_id"),
                )

            else:
                if "conversation_id" in data:
                    raise ValueError(
                        "Skill selections use the conversation associated "
                        "with their original request. Do not supply "
                        "conversation_id."
                    )

                result = resolve_selection(
                    data.get("request_id"),
                    data.get("remove_skill_names"),
                )

            if result.get("selection_required"):
                self.send_json(200, result)
                return

            payload = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": result["content"],
                        }
                    }
                ],
                "stats": result["stats"],
                "selection_required": False,
            }

            if "conversation_id" in result:
                payload["conversation_id"] = result["conversation_id"]

            if "memory_persistence" in result:
                payload["memory_persistence"] = result["memory_persistence"]
            if "conversation_usage" in result:
                payload["conversation_usage"] = result["conversation_usage"]

            self.send_json(200, payload)

        except ConversationNotFoundError:
            self.send_json(
                404,
                {"error": ("Conversation was not found in the configured scope.")},
            )

        except KeyError:
            self.send_json(
                404,
                {"error": ("The requested resource was not found or expired.")},
            )

        except ValueError as exc:
            self.send_json(400, {"error": str(exc)})

        except Exception as exc:
            # Log the exception type, without request or database contents.
            print(
                f"[Chat Error] {type(exc).__name__}",
                flush=True,
            )
            self.send_json(500, {"error": "Failed to process chat request."})

    def send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")

        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def attach_local_job_api(server, api):
    """Attach before serving; the worker is started separately by the caller.

    Existing conversation endpoints and job finalization must use the same
    configured memory scope. Attaching disables direct legacy chat inference.
    This function does not change the default application entry point.
    """
    if not isinstance(api, LocalJobAPI):
        raise ValueError("api must be LocalJobAPI.")
    if api.memory is not analyst.MEMORY_MANAGER:
        raise ValueError("Job and conversation APIs must share the memory manager.")
    server.job_api = api


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 5000), OrchestratorHandler)
    print("=" * 60)
    print(f" SECURE UI & ORCHESTRATOR RUNNING ON PORT 5000 ")

    print("=" * 60)
    server.serve_forever()
