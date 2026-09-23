# Upstream sources

The dependency source files are not redistributed in this repository.

- MPSCore: https://github.com/chiamin/JuliaMPS
  - Revision: `8bdfeb4db57d568e0ff3b7b3730d4a5b66b55f80`
  - Supplied archive SHA-256: `4190eda7abd7c0bf0d312f50a854545baf14fca632f2893c80dc9108fee7e281`
- QTTCore: https://github.com/chiamin/JuliaQTT
  - Revision: `3b3dc804c5051a88e2dc61b0fc29ed87abf586af`
  - Supplied archive SHA-256: `01608402ba84a0302ab1e3b3448783dd1b67d15d2d2bb7115e775f504c2abf33`

The application core imports MPSCore and QTTCore; it does not separately include
the standalone LinearSolver.jl attachment. Any locally modified upstream source
would need to be reconciled against the pinned revision before a reproduction
claim. No new license is assigned to upstream code by this repository.
