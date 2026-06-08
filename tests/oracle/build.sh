#!/usr/bin/env bash
# Configure + build the NAMCore oracle (Release). Produces tests/oracle/build/oracle.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cmake -S "$here" -B "$here/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$here/build" -j
echo "built: $here/build/oracle"
