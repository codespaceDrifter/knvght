import chess.svg
import webbrowser
import http.server
import socketserver
import threading
import subprocess
import time

#http://localhost:8000
PORT = 8000
current_svg = ""

class Handler(http.server.SimpleHTTPRequestHandler):

    def log_message(self, format, *args):
        pass  # <--- ADD THIS to disable all server logging

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = """
            <html>
            <head><meta http-equiv="refresh" content="0.5"></head>
            <body>
            <img src="/board.svg" width="720" height="720">
            </body>
            </html>
            """
            self.wfile.write(html.encode('utf-8'))
        elif self.path == '/board.svg':
            self.send_response(200)
            self.send_header('Content-type', 'image/svg+xml')
            self.end_headers()
            self.wfile.write(current_svg.encode('utf-8'))
        else:
            self.send_error(404)

def display_board(board):
    global current_svg
    current_svg = chess.svg.board(board)

httpd = socketserver.TCPServer(("", PORT), Handler)
thread = threading.Thread(target=httpd.serve_forever, daemon=True)
thread.start()



