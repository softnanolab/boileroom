#!/usr/bin/env python3
"""Cleanup utility for conda backend server processes.

This script helps identify and clean up orphaned conda backend server processes
that weren't properly shut down.
"""

import subprocess
import sys
from pathlib import Path


def find_conda_backend_processes():
    """Find all conda backend server processes.

    Returns
    -------
    list[dict]
        List of process information dictionaries with keys: pid, port, cmd
    """
    processes = []

    # Try different methods to find processes
    try:
        # Method 1: Check for server.py processes
        result = subprocess.run(
            ["pgrep", "-af", "backend/server.py"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if line and "server.py" in line:
                    parts = line.split(None, 1)
                    if len(parts) >= 2:
                        pid = parts[0]
                        cmd = parts[1]
                        # Extract port from command
                        port = None
                        if "--port" in cmd:
                            try:
                                port_idx = cmd.split().index("--port") + 1
                                port = cmd.split()[port_idx]
                            except (IndexError, ValueError):
                                pass
                        processes.append({"pid": int(pid), "port": port, "cmd": cmd})
    except FileNotFoundError:
        pass

    # Method 2: Use ps if pgrep not available
    if not processes:
        try:
            result = subprocess.run(
                ["ps", "aux"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if "backend/server.py" in line and "grep" not in line:
                        parts = line.split(None, 10)
                        if len(parts) >= 11:
                            pid = parts[1]
                            cmd = " ".join(parts[10:])
                            port = None
                            if "--port" in cmd:
                                try:
                                    port_idx = cmd.split().index("--port") + 1
                                    port = cmd.split()[port_idx]
                                except (IndexError, ValueError):
                                    pass
                            processes.append({"pid": int(pid), "port": port, "cmd": cmd})
        except FileNotFoundError:
            pass

    return processes


def kill_processes(pids, force=False):
    """Kill processes by PID.

    Parameters
    ----------
    pids : list[int]
        List of process IDs to kill.
    force : bool
        If True, use SIGKILL instead of SIGTERM.

    Returns
    -------
    bool
        True if all processes were killed successfully.
    """
    if not pids:
        return True

    signal = "SIGKILL" if force else "SIGTERM"
    try:
        subprocess.run(
            ["kill", "-9" if force else "-15"] + [str(pid) for pid in pids],
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error killing processes: {e}", file=sys.stderr)
        return False


def main():
    """Main entry point."""
    processes = find_conda_backend_processes()

    if not processes:
        print("No conda backend server processes found.")
        return 0

    print(f"Found {len(processes)} conda backend server process(es):")
    print()
    for proc in processes:
        port_info = f" (port {proc['port']})" if proc["port"] else ""
        print(f"  PID {proc['pid']}{port_info}: {proc['cmd'][:80]}...")

    if "--kill" in sys.argv or "-k" in sys.argv:
        force = "--force" in sys.argv or "-f" in sys.argv
        pids = [proc["pid"] for proc in processes]
        print()
        if force:
            print(f"Force killing {len(pids)} process(es)...")
        else:
            print(f"Terminating {len(pids)} process(es)...")
        if kill_processes(pids, force=force):
            print("Successfully terminated all processes.")
            return 0
        else:
            print("Failed to terminate some processes.", file=sys.stderr)
            return 1
    else:
        print()
        print("To kill these processes, run:")
        print(f"  {sys.argv[0]} --kill")
        print()
        print("To force kill (SIGKILL), run:")
        print(f"  {sys.argv[0]} --kill --force")
        return 0


if __name__ == "__main__":
    sys.exit(main())

