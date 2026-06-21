#!/bin/bash
# Build script: injects BACKEND_URL env variable into config.js at deploy time
echo "window.__ENV__ = { BACKEND_URL: \"${BACKEND_URL:-}\" };" > config.js
echo "Generated config.js with BACKEND_URL=${BACKEND_URL:-'(empty/local mode)'}"
