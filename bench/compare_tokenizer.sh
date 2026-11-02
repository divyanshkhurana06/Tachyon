#!/usr/bin/env bash
# Compares Tachyon's tokenizer with llama.cpp's on any files: every file's
# token ids must be identical.
#
#   bench/compare_tokenizer.sh path/to/llama-tokenize model.gguf models/qwen2.5-0.5b-instruct FILE...
set -euo pipefail
lt=$1 gguf=$2 model=$3
shift 3
same=0 diff=0 tokens=0
for f in "$@"; do
  a=$("$lt" -m "$gguf" -f "$f" --ids --no-bos --no-parse-special --no-escape --log-disable 2>/dev/null | tail -1)
  b=$(./target/release/tachyon tokenize --model "$model" --file "$f")
  tokens=$((tokens + $(tr ',' '\n' <<<"$b" | wc -l)))
  if [ "$a" = "$b" ]; then same=$((same + 1)); else diff=$((diff + 1)); echo "differs: $f"; fi
done
echo "$same files identical, $diff different, $tokens tokens"
[ "$diff" -eq 0 ]
