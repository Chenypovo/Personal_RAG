import os

# faiss-cpu and torch each bundle their own OpenMP runtime (libomp). On macOS
# loading both aborts with "OMP: Error #15: ... libomp.dylib already initialized".
# Set this before torch/faiss are imported (this package init runs first) so the
# duplicate runtime is tolerated. Can be overridden by setting the env var yourself.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# faiss-cpu and torch fighting over OpenMP threads can segfault (SIGSEGV) on
# macOS when both run in one process (e.g. building a FAISS index with the local
# torch embedder). Pinning OpenMP to a single thread avoids the crash; override
# by exporting OMP_NUM_THREADS yourself if you need multi-threaded BLAS.
os.environ.setdefault("OMP_NUM_THREADS", "1")
