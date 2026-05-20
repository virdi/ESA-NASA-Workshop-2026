#!/usr/bin/env bash
set -e

if ! command -v opencode >/dev/null 2>&1; then
    echo "opencode not found..."
    echo "installing opencode..."
    curl -fsSL https://opencode.ai/install | bash
    # installer adds opencode to PATH via ~/.bashrc; reload so this shell sees it
    [ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc"
fi

exec opencode --agent prithvi-workshop-care --model amazon-bedrock/mistral.mistral-large-3-675b-instruct
