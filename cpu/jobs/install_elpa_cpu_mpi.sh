#!/usr/bin/env bash
set -eo pipefail

export LC_ALL=C
export LANG=C

source /etc/profile >/dev/null 2>&1 || true
source "$HOME/.bashrc" >/dev/null 2>&1 || true
if command -v module >/dev/null 2>&1; then module purge >/dev/null 2>&1 || true || true; fi
if command -v module >/dev/null 2>&1; then module load gcc/11.2.0 intel/2021.3.0 python/3.8.13 tbb/2022.3 compiler-rt/2025.3.0 mkl/2025.3 mpi/2021.17 || true; fi

unset CPATH
unset C_INCLUDE_PATH
unset CPLUS_INCLUDE_PATH

export I_MPI_F90=ifort
export I_MPI_F77=ifort
export I_MPI_CC=icc
export I_MPI_CXX=g++
export MKL_INTERFACE_LAYER=LP64
export MKL_THREADING_LAYER=SEQUENTIAL
export MKL_DYNAMIC=FALSE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS=1

AE="${AE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ROOT="$AE/cpu/bench"
DEPS="$AE/results/deps"
TARBALL="$DEPS/elpa-src-548c64b.tar.gz"
SRC="$DEPS/elpa-src-548c64b"
BUILD="$DEPS/elpa-build-cpu-mpi"
PREFIX="$DEPS/elpa-install-cpu-mpi"
AUTOTOOLS_PREFIX="$DEPS/autotools"

install_autotools() {
  if [ -x "$AUTOTOOLS_PREFIX/bin/autoconf" ] && \
     [ -x "$AUTOTOOLS_PREFIX/bin/automake" ] && \
     [ -x "$AUTOTOOLS_PREFIX/bin/libtoolize" ] && \
     "$AUTOTOOLS_PREFIX/bin/autoconf" --version | head -1 | grep -q '2\.72' && \
     "$AUTOTOOLS_PREFIX/bin/libtoolize" --version | head -1 | grep -q '2\.5\.4'; then
    return
  fi

  echo "=== install autotools ==="
  mkdir -p "$DEPS/src" "$DEPS/build-autotools" "$AUTOTOOLS_PREFIX"
  cd "$DEPS/src"
  [ -f autoconf-2.72.tar.gz ] || wget -q https://ftp.gnu.org/gnu/autoconf/autoconf-2.72.tar.gz
  [ -f automake-1.17.tar.gz ] || wget -q https://ftp.gnu.org/gnu/automake/automake-1.17.tar.gz
  [ -f libtool-2.5.4.tar.gz ] || wget -q https://ftp.gnu.org/gnu/libtool/libtool-2.5.4.tar.gz

  rm -rf "$DEPS/build-autotools/autoconf-2.72"
  tar -xzf autoconf-2.72.tar.gz -C "$DEPS/build-autotools"
  cd "$DEPS/build-autotools/autoconf-2.72"
  ./configure --prefix="$AUTOTOOLS_PREFIX"
  make -j "${BUILD_JOBS:-8}"
  make install

  export PATH="$AUTOTOOLS_PREFIX/bin:$PATH"

  rm -rf "$DEPS/build-autotools/libtool-2.5.4"
  tar -xzf "$DEPS/src/libtool-2.5.4.tar.gz" -C "$DEPS/build-autotools"
  cd "$DEPS/build-autotools/libtool-2.5.4"
  ./configure --prefix="$AUTOTOOLS_PREFIX"
  make -j "${BUILD_JOBS:-8}"
  make install

  rm -rf "$DEPS/build-autotools/automake-1.17"
  tar -xzf "$DEPS/src/automake-1.17.tar.gz" -C "$DEPS/build-autotools"
  cd "$DEPS/build-autotools/automake-1.17"
  ./configure --prefix="$AUTOTOOLS_PREFIX"
  make -j "${BUILD_JOBS:-8}"
  make install
}

echo "=== job info ==="
date
hostname
echo "MKLROOT=$MKLROOT"
echo "I_MPI_ROOT=$I_MPI_ROOT"
mpiifort -show
mpiicc -show
mpiicpc -show
g++ --version | head -1
ifort --version | head -1
python3 --version

install_autotools
export PATH="$AUTOTOOLS_PREFIX/bin:$PATH"
export ACLOCAL_PATH="$AUTOTOOLS_PREFIX/share/aclocal:${ACLOCAL_PATH:-}"
autoconf --version | head -1
automake --version | head -1
libtoolize --version | head -1

echo "=== unpack ==="
rm -rf "$SRC" "$BUILD" "$PREFIX"
mkdir -p "$SRC" "$BUILD" "$PREFIX"
tar -xzf "$TARBALL" -C "$SRC" --strip-components=1

echo "=== autogen ==="
cd "$SRC"
./autogen.sh

echo "=== configure ==="
cd "$BUILD"
MKL_LDFLAGS="-L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib"
MKL_FCFLAGS="-I${MKLROOT}/include -I${MKLROOT}/include/mkl/intel64/lp64"
MKL_BLAS_LIBS="-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl"
MKL_SCALAPACK_LIBS="-lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl"
GCC_CXX_TARGET_INCLUDE="/public/software/tools/gcc_11.2/include/c++/11.2.0/x86_64-linux-gnu"
"$SRC/configure" \
  --prefix="$PREFIX" \
  --with-mpi=yes \
  --disable-mpi-module \
  --disable-openmp \
  --disable-c-tests \
  --disable-cpp-tests \
  --disable-Fortran-tests \
  --with-test-programs=no \
  --disable-python \
  CC=mpiicc \
  CXX=g++ \
  FC=mpiifort \
  F77=mpiifort \
  CFLAGS="-O2" \
  CXXFLAGS="-O2 -march=native -isystem $GCC_CXX_TARGET_INCLUDE" \
  FCFLAGS="-O2 $MKL_FCFLAGS" \
  SCALAPACK_FCFLAGS="$MKL_FCFLAGS" \
  SCALAPACK_LDFLAGS="$MKL_LDFLAGS $MKL_SCALAPACK_LIBS" \
  LDFLAGS="$MKL_LDFLAGS" \
  LIBS="$MKL_BLAS_LIBS"

echo "=== build ==="
make -j "${BUILD_JOBS:-8}" libelpa.la

echo "=== install ==="
make install-libLTLIBRARIES \
  install-pkgconfigDATA \
  install-nobase_elpa_includeHEADERS \
  install-nobase_nodist_elpa_includeHEADERS

echo "=== installed files ==="
find "$PREFIX" -maxdepth 3 -type f | sort | sed -n '1,120p'
echo "=== pkg-config ==="
PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:$PREFIX/lib64/pkgconfig:${PKG_CONFIG_PATH:-}" pkg-config --cflags --libs elpa || true
echo "=== done ==="
date
