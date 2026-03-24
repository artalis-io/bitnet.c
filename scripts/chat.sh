#!/bin/bash
DEFAULT_MODEL="models/bitnet-b1.58-2B-4T.gguf"
DEFAULT_URL="https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/bitnet-b1.58-2B-4T.gguf"

MODEL="${1:-$DEFAULT_MODEL}"
shift 2>/dev/null  # consume model arg, remaining args pass through

if [ ! -f "$MODEL" ]; then
    echo "Model not found: $MODEL"
    if [ "$MODEL" = "$DEFAULT_MODEL" ]; then
        echo "Download it with:"
        echo "  mkdir -p models && wget -O $MODEL $DEFAULT_URL"
    fi
    echo ""
    echo "Available models:"
    ls models/*.gguf 2>/dev/null | sed 's/^/  /'
    exit 1
fi

./bitnet "$MODEL" --chat "$@"
