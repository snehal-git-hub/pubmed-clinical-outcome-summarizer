
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import time


class MockAPI(BaseHTTPRequestHandler):

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        request_data = json.loads(self.rfile.read(length))

        messages = request_data.get("messages", [])
        prompt = next(
            (m["content"] for m in messages if m["role"] == "user"),
            ""
        )

        if "PMID:349281" in prompt:
            summary = (
                "Drug X was effective in 40% of the 500 patients, "
                "although nausea was reported as an adverse effect."
            )
        else:
            summary = (
                "The study reported no significant improvement "
                "in cohort B compared with placebo."
            )

        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": summary
                    }
                }
            ]
        }

        body = json.dumps(response).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

        print("Received request successfully!")


server = HTTPServer(("127.0.0.1", 8000), MockAPI)

print("Mock API server running at http://127.0.0.1:8000")
server.serve_forever()
