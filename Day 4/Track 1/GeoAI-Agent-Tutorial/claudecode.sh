#!/usr/bin/env bash
set -e

if ! command -v claude >/dev/null 2>&1; then
    echo "claude not found..."
    echo "installing claude code..."
    curl -fsSL https://claude.ai/install.sh | bash
    # installer puts claude in ~/.local/bin; ensure it's on PATH for this shell
    export PATH="$HOME/.local/bin:$PATH"
    [ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc"
fi

exec claude "/prithvi-workshop-care"
