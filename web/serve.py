"""Static dev server for the web demo that disables browser caching.

`python3 -m http.server` sends no cache headers, so browsers reuse an old
demo.js after it changes. This sends `Cache-Control: no-store` instead, so a
normal reload always picks up the latest files.

Run from the repository root:  python3 web/serve.py [port]   (default 8090)
"""

import functools
import http.server
import os
import sys


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8090
    web_dir = os.path.dirname(os.path.abspath(__file__))
    handler = functools.partial(NoCacheHandler, directory=web_dir)
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        print(f"serving {web_dir} on http://localhost:{port} (no-cache)")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
