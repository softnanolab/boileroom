#!/usr/bin/env bash
# Task-local, non-setuid source build for the explicit Ubuntu 22.04 CPU SIF test.
# Official Apptainer v1.5.4 INSTALL.md and dist/rpm/apptainer.spec.in define
# the source/helper versions. Nothing installs or reloads a host security profile.
set -euo pipefail
: "${RUNNER_TEMP:?}" "${GITHUB_RUN_ID:?}" "${GITHUB_RUN_ATTEMPT:?}"
: "${ARTIFACT_DIR:?}" "${APPTAINER_SOURCE:?}" "${GITHUB_ENV:?}" "${GITHUB_PATH:?}"
[[ "${GITHUB_ACTIONS:-}" == true && "${RUNNER_ENVIRONMENT:-}" == github-hosted ]]
[[ "${GITHUB_REPOSITORY_VISIBILITY:-}" == public && "$(id -u)" != 0 ]]
root="$RUNNER_TEMP/boileroom-apptainer-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT"
# Never replace a prior installation or cache.
[[ ! -e "$APPTAINER_SOURCE" ]]
mkdir "$root"
printf '%s:%s\n' "$GITHUB_RUN_ID" "$GITHUB_RUN_ATTEMPT" > "$root/.owner"
mkdir -p "$ARTIFACT_DIR"
exec > >(tee "$ARTIFACT_DIR/apptainer-source-build.log") 2>&1
prefix="$root/install"
export GOPATH="$root/gopath" GOCACHE="$root/go-cache" GOMODCACHE="$root/go-modules"
export GOTOOLCHAIN=local GOPROXY=off GOMAXPROCS=2 MAKEFLAGS=-j2
export GOFLAGS="-p=2 -mod=vendor"
fetch() {
  curl --fail --location --proto '=https' --tlsv1.2 --max-time 180 "$1" --output "$2"
  echo "$3  $2" | sha256sum --check -
}
fetch https://github.com/apptainer/apptainer/releases/download/v1.5.4/apptainer-1.5.4.tar.gz \
  "$APPTAINER_SOURCE" ec1f51e696ba384f90bc7ab2c7a438fee70da98763d001b8f3175d14a91cfcf6
fetch https://go.dev/dl/go1.26.9.linux-amd64.tar.gz "$root/go.tar.gz" \
  42d158b4d8f7b61ac0a830567c940a86098fb7aac52e467a5ebec03ef5cc2f8d
fetch https://github.com/vasi/squashfuse/archive/0.6.2/squashfuse-0.6.2.tar.gz "$root/squashfuse.tar.gz" \
  ae194df463ac5a9cbb25f94021d44bfb14dc21a0fc577e16348eca2fdc77c83b
fetch https://github.com/plougher/squashfs-tools/archive/4.7.5/squashfs-tools-4.7.5.tar.gz "$root/squashfs-tools.tar.gz" \
  547b7b7f4d2e44bf91b6fc554664850c69563701deab9fd9cd7e21f694c88ea6
fetch https://github.com/containers/fuse-overlayfs/archive/v1.16/fuse-overlayfs-1.16.tar.gz "$root/fuse-overlayfs.tar.gz" \
  45968517603389ead067222d234bc8d8ed33e4b4f8ba16216bdd3e6aedcccea9
# Build libraries only: no Apptainer package maintainer hooks, AppArmor tooling,
# uidmap changes, setuid install, namespace restrictions, or new apt repositories.
sudo apt-get update
sudo apt-get install -y --no-install-recommends build-essential libseccomp-dev libtalloc-dev \
  libattr1-dev libprotobuf-c-dev uuid-dev autoconf automake libtool pkg-config libfuse3-dev \
  zlib1g-dev liblzo2-dev liblz4-dev liblzma-dev libzstd-dev
for archive in "$APPTAINER_SOURCE" "$root/go.tar.gz" "$root/squashfuse.tar.gz" \
  "$root/squashfs-tools.tar.gz" "$root/fuse-overlayfs.tar.gz"; do
  tar -xzf "$archive" -C "$root"
done
export PATH="$root/go/bin:$PATH"
go version
cd "$root/apptainer-1.5.4"
./mconfig --without-suid --without-libsubid --prefix="$prefix" --sysconfdir="$prefix/etc" --localstatedir="$prefix/var"
make -C builddir -j2
make -C builddir install
# Compile only the three mount/build helpers needed for ordinary read-only SIFs.
# Do not download/run PRoot or change container execution semantics.
cd "$root/squashfuse-0.6.2"
./autogen.sh
CFLAGS=-std=c99 ./configure --enable-multithreading
make -j2 squashfuse_ll
install -m 0755 squashfuse_ll "$prefix/libexec/apptainer/bin/"
cd "$root/squashfs-tools-4.7.5/squashfs-tools"
make -j2 mksquashfs unsquashfs GZIP_SUPPORT=1 LZO_SUPPORT=1 LZ4_SUPPORT=1 XZ_SUPPORT=1 ZSTD_SUPPORT=1
install -m 0755 mksquashfs unsquashfs "$prefix/libexec/apptainer/bin/"
cd "$root/fuse-overlayfs-1.16"
./autogen.sh
./configure
make -j2
install -m 0755 fuse-overlayfs "$prefix/libexec/apptainer/bin/"
"$prefix/bin/apptainer" --version
"$prefix/bin/apptainer" buildcfg | tee "$ARTIFACT_DIR/apptainer-source-buildcfg.log"
grep -Eq '^APPTAINER_SUID_INSTALL[[:space:]]*=[[:space:]]*0$' "$ARTIFACT_DIR/apptainer-source-buildcfg.log"
for field in "PREFIX=$prefix" "SYSCONFDIR=$prefix/etc" "LIBEXECDIR=$prefix/libexec"; do
  grep -Fx "$field" "$ARTIFACT_DIR/apptainer-source-buildcfg.log"
done
[[ -z "$(find "$prefix" -type f -perm /6000 -print)" ]]
sha256sum "$APPTAINER_SOURCE" "$root/"*.tar.gz "$prefix/bin/apptainer" \
  "$prefix/libexec/apptainer/bin/"* > "$ARTIFACT_DIR/apptainer-toolchain-sha256.txt"
echo "APPTAINER_BUILD_PREFIX=$prefix" >> "$GITHUB_ENV"
echo "$prefix/bin" >> "$GITHUB_PATH"
