#!/usr/bin/env bash
set -e

if ! command -v opencode >/dev/null 2>&1; then
    echo "opencode not found..."
    echo "installing opencode..."
    curl -fsSL https://opencode.ai/install | bash
fi

exec opencode --agent prithvi-workshop-care
