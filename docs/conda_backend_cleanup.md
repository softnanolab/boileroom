# Conda Backend Process Management

The conda backend runs HTTP servers in separate conda environments. This document explains how to check for running servers and clean them up when needed.

## Checking for Running Servers

### Quick Check

Use the provided utility script:

```bash
# Check for running servers
scripts/check_conda_backends.sh

# Or use the Python utility
python3 scripts/cleanup_conda_backends.py
```

### Manual Commands

**Check ports 8000-8099:**

```bash
# Using ss (modern)
ss -tulpn | grep LISTEN | grep -E ':(800[0-9]|80[1-9][0-9])'

# Using netstat (legacy)
netstat -tulpn | grep LISTEN | grep -E ':(800[0-9]|80[1-9][0-9])'

# Using lsof
lsof -i -P -n | grep LISTEN | grep -E ':(800[0-9]|80[1-9][0-9])'
```

**Find server.py processes:**

```bash
# Find all backend server processes
ps aux | grep 'backend/server.py' | grep -v grep

# Or using pgrep
pgrep -af 'backend/server.py'
```

## Cleaning Up Servers

### Using the Utility Script

```bash
# Show running servers and instructions
python3 scripts/cleanup_conda_backends.py

# Kill all conda backend servers gracefully (SIGTERM)
python3 scripts/cleanup_conda_backends.py --kill

# Force kill all servers (SIGKILL)
python3 scripts/cleanup_conda_backends.py --kill --force
```

### Manual Cleanup

**Kill by port:**

```bash
# Kill specific ports
kill $(lsof -ti:8000,8001,8002,8003,8004,8005,8006)

# Or using fuser
fuser -k 8000/tcp 8001/tcp 8002/tcp
```

**Kill by process name:**

```bash
# Find and kill all server.py processes
kill $(ps aux | grep 'backend/server.py' | grep -v grep | awk '{print $2}')

# Force kill if needed
kill -9 $(ps aux | grep 'backend/server.py' | grep -v grep | awk '{print $2}')
```

**Kill by PID:**

```bash
# Kill specific processes
kill 2863211 2864576 2868304

# Force kill
kill -9 2863211 2864576 2868304
```

## Automatic Cleanup

`ModelWrapper` now automatically cleans up backend resources when model objects are destroyed. This means:

- Backend servers are shut down when model objects are garbage collected
- Context manager support (`with` statements) ensures cleanup even if exceptions occur
- An atexit hook provides an additional safety net when Python exits normally

### Automatic Cleanup (Default Behavior)

You can simply create and use models - cleanup happens automatically:

```python
from boileroom import ESM2

model = ESM2(backend="conda", device="cuda:0")
result = model.embed(sequences)
# Backend is automatically shut down when model goes out of scope or is garbage collected
```

### Explicit Cleanup with Context Managers (Recommended)

For explicit control over resource lifecycle, use context managers:

```python
from boileroom import ESM2

# Using the built-in context manager support
with ESM2(backend="conda", device="cuda:0") as model:
    result = model.embed(sequences)
# Backend is automatically shut down when exiting the with block
```

This pattern is recommended when:
- You want explicit control over when cleanup happens
- Working with multiple models in sequence
- Managing resources in long-running scripts

### Manual Cleanup

If you need to manually shut down a model before it goes out of scope:

```python
from boileroom import ESM2

model = ESM2(backend="conda", device="cuda:0")
try:
    result = model.embed(sequences)
finally:
    # Manually shut down the backend
    model._backend.stop()
```

**Note**: Manual cleanup is usually not necessary as automatic cleanup handles this. Use context managers for explicit control.

## Port Allocation

The conda backend automatically finds available ports starting from 8000. Each new model instance gets the next available port:

- Port 8000: First conda backend server
- Port 8001: Second conda backend server
- Port 8002: Third conda backend server
- ... and so on

If ports are not cleaned up, you may eventually run out of available ports. Regular cleanup prevents this issue.

## Monitoring Resource Usage

**Check CPU and memory usage:**

```bash
# Watch all server processes
watch -n 2 "ps aux | grep 'backend/server.py' | grep -v grep"

# Check specific PID
top -p 2863211
htop -p 2863211
```

**Check network connections:**

```bash
# See active connections to conda backend servers
netstat -an | grep -E ':(800[0-9]|80[1-9][0-9])' | grep ESTABLISHED
```

## Prevention Tips

1. **Use context managers** (`with` statements) for explicit resource management when needed
2. **Run cleanup utility regularly** to check for orphaned processes (useful for debugging)
3. **Automatic cleanup handles most cases** - models clean up when garbage collected or when Python exits
4. **Monitor port usage** if you're creating many model instances in quick succession
5. **Handle exceptions properly** - context managers ensure cleanup even if exceptions occur

**Note**: Automatic cleanup means orphaned processes should be rare. If you encounter them, it may indicate:
- A hard crash (SIGKILL) that prevented Python cleanup code from running
- A Python bug or interpreter issue
- Manual process killing that bypassed the shutdown mechanism

