"""The sandbox image: Ubuntu 24.04, the pinned Actions runner, and the root-owned guard files.

The files in `sandbox/` are embedded into the image text rather than mounted, so the image can be
built from inside the controller container (which has the package on disk) as well as from a
laptop, and its content hash depends only on what it contains.
"""

from __future__ import annotations

import base64
from pathlib import Path

import modal

RUNNER_VERSION = "2.338.0"
RUNNER_SHA256 = "af4b794c1bc41d73d40535e3fe092a39f9679cd8d965954c2aca25a05ca41d32"
RUNNER_URL = (
    f"https://github.com/actions/runner/releases/download/v{RUNNER_VERSION}/"
    f"actions-runner-linux-x64-{RUNNER_VERSION}.tar.gz"
)
# Hosted runners ship Node.js, and Bakeoff's tests shell out to `node` (several are skipped when it is missing, so its
# absence would quietly shrink the suite). The same release the hosted image offers, verified before it is unpacked.
NODE_VERSION = "24.21.0"
NODE_SHA256 = "fd8e59d5a511510f6a298afb548f18c7d2b1be404d8b4a27d94fbe49f56cb2d6"
NODE_URL = f"https://nodejs.org/dist/v{NODE_VERSION}/node-v{NODE_VERSION}-linux-x64.tar.xz"
# Which Playwright release computes the OS package list for Chromium; browser jobs install their own (newer) Playwright.
PLAYWRIGHT_DEPS_VERSION = "1.63.0"
SANDBOX_DIR = Path(__file__).parent / "sandbox"
# What the controller ships of this package. Modal's default ships only `*.py`, which would drop
# `sandbox/job-started.sh` and make `runner_image()` fail on import inside the controller.
SOURCE_IGNORE = ["**/*.pyc", "**/__pycache__/**"]
APT_PACKAGES = (
    "ca-certificates curl git git-lfs jq unzip zip xz-utils build-essential pkg-config python3 python3-venv "
    # actions/cache includes its compression method in the archive version. Match hosted Linux
    # runners so existing zstd caches remain visible after moving a workflow to Modal.
    "python3-pip gh procps zstd"
)


def _install_file(name: str, mode: str) -> str:
    payload = base64.b64encode((SANDBOX_DIR / name).read_bytes()).decode()
    return (
        f"echo {payload} | base64 -d > /opt/ci/{name} && chown root:root /opt/ci/{name} && chmod {mode} /opt/ci/{name}"
    )


def runner_image() -> modal.Image:
    return (
        modal.Image.from_registry("ubuntu:24.04")
        .env({"DEBIAN_FRONTEND": "noninteractive"})
        .apt_install(*APT_PACKAGES.split())
        .run_commands(
            "useradd --create-home --uid 1001 --shell /bin/bash runner",
            "mkdir -p /home/runner/actions-runner /opt/hostedtoolcache /opt/ci",
            f"curl -fsSL -o /tmp/runner.tgz {RUNNER_URL}",
            f"echo '{RUNNER_SHA256}  /tmp/runner.tgz' | sha256sum -c -",
            "tar xzf /tmp/runner.tgz -C /home/runner/actions-runner && rm /tmp/runner.tgz",
            "/home/runner/actions-runner/bin/installdependencies.sh",
            "chown -R runner:runner /home/runner /opt/hostedtoolcache",
            "chown root:root /opt/ci && chmod 755 /opt/ci",
        )
        .run_commands(
            f"curl -fsSL -o /tmp/node.txz {NODE_URL}",
            f"echo '{NODE_SHA256}  /tmp/node.txz' | sha256sum -c -",
            "tar xJf /tmp/node.txz -C /usr/local --strip-components=1 --no-same-owner && rm /tmp/node.txz",
            "node --version && npm --version",
        )
        .run_commands(
            # Only the shared libraries and fonts Chromium needs, so a browser job can run unprivileged: its own
            # `playwright install chromium` then downloads the browser into the runner's home with no sudo. The
            # throwaway venv exists to let Playwright compute the package list for this exact distribution.
            "python3 -m venv /tmp/pw",
            f"/tmp/pw/bin/pip install --quiet playwright=={PLAYWRIGHT_DEPS_VERSION}",
            "/tmp/pw/bin/playwright install-deps chromium",
            "rm -rf /tmp/pw /var/lib/apt/lists/*",
        )
        .run_commands(
            _install_file("guard.py", "644"),
            _install_file("supervisor.py", "644"),
            _install_file("job-started.sh", "755"),
        )
    )
