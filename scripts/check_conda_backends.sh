#!/bin/bash
# Check and optionally clean up conda backend server processes

echo "=== Checking for conda backend servers on ports 8000-8099 ==="
echo ""

# Find all listening ports in range 8000-8099
if command -v ss &> /dev/null; then
    PORTS=$(ss -tulpn 2>/dev/null | grep LISTEN | grep -E ':(800[0-9]|80[1-9][0-9])' | awk '{print $5}' | cut -d':' -f2)
elif command -v netstat &> /dev/null; then
    PORTS=$(netstat -tulpn 2>/dev/null | grep LISTEN | grep -E ':(800[0-9]|80[1-9][0-9])' | awk '{print $4}' | cut -d':' -f2)
elif command -v lsof &> /dev/null; then
    PORTS=$(lsof -i -P -n 2>/dev/null | grep LISTEN | grep -E ':(800[0-9]|80[1-9][0-9])' | awk '{print $9}' | cut -d':' -f2)
else
    echo "Error: Need ss, netstat, or lsof to check ports"
    exit 1
fi

if [ -z "$PORTS" ]; then
    echo "No conda backend servers found on ports 8000-8099"
    exit 0
fi

echo "Found servers on the following ports:"
echo "$PORTS" | while read port; do
    echo "  Port $port:"
    if command -v lsof &> /dev/null; then
        PID=$(lsof -ti:$port 2>/dev/null)
        if [ ! -z "$PID" ]; then
            ps -p $PID -o pid=,user=,cmd= --no-headers 2>/dev/null | sed 's/^/    /'
        fi
    elif command -v fuser &> /dev/null; then
        PID=$(fuser $port/tcp 2>&1 | grep -oE '[0-9]+' | head -1)
        if [ ! -z "$PID" ]; then
            ps -p $PID -o pid=,user=,cmd= --no-headers 2>/dev/null | sed 's/^/    /'
        fi
    fi
    echo ""
done

echo "=== Options ==="
echo "1. To kill all conda backend servers, run:"
echo "   kill \$(ps aux | grep 'backend/server.py' | grep -v grep | awk '{print \$2}')"
echo ""
echo "2. To kill servers on specific ports, run:"
echo "   kill \$(lsof -ti:8000,8001,8002,8003,8004,8005,8006)"
echo ""
echo "3. To check processes in detail:"
echo "   ps aux | grep 'backend/server.py' | grep -v grep"

