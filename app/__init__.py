import os

# faiss-cpu and torch each bundle their own OpenMP runtime (libomp). On macOS
# loading both aborts with "OMP: Error #15: ... libomp.dylib already initialized".
# Set this before torch/faiss are imported (this package init runs first) so the
# duplicate runtime is tolerated. Can be overridden by setting the env var yourself.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
