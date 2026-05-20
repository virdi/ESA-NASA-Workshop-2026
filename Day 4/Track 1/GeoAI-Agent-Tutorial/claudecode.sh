#!/usr/bin/env bash
set -e

if ! command -v claude >/dev/null 2>&1; then
    echo "claude not found..."
    echo "installing claude code..."
    curl -fsSL https://claude.ai/install.sh | bash
fi

exec claude "/prithvi-workshop-care"
