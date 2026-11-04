#!/usr/bin/env bash
# Run pdfcancel against a self-hosted OpenAI-compatible / LiteLLM endpoint.
#
# Site settings come from a profile (default ~/.config/pdfcancel/skill.env; see
# profile.env.example), exported explicitly so a stray .env can't redirect a run.
#
# Usage:
#   pdfcancel-skill.sh check                      # verify endpoint, list models
#   pdfcancel-skill.sh convert <pdf|dir> [opts]   # OCR + figures + --full + chunks
#   pdfcancel-skill.sh ocr     <pdf|dir> [opts]   # text-only pass (no figures)
#   pdfcancel-skill.sh ask <pdf> <page> "prompt"  # one page -> model, answer to stdout
#   pdfcancel-skill.sh <pdfcancel args...>        # passthrough (search, status, ...)
set -euo pipefail

PROFILE="${PDFCANCEL_SKILL_PROFILE:-$HOME/.config/pdfcancel/skill.env}"
[ -f "$PROFILE" ] || {
  echo "No skill profile at $PROFILE. Copy profile.env.example there and fill it in." >&2
  exit 78
}
set -a
# shellcheck disable=SC1090
. "$PROFILE"
set +a

: "${PDFCANCEL_VENV:?set PDFCANCEL_VENV in $PROFILE}"
: "${PDFCANCEL_API_BASE:?set PDFCANCEL_API_BASE in $PROFILE}"
: "${PDFCANCEL_VLM_MODEL:?set PDFCANCEL_VLM_MODEL in $PROFILE}"
BIN="$PDFCANCEL_VENV/bin/pdfcancel"
PY="$PDFCANCEL_VENV/bin/python"
HELPER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/pdfcancel_skill.py"
[ -x "$BIN" ] || { echo "pdfcancel not installed in $PDFCANCEL_VENV (run install.sh)" >&2; exit 127; }

export PDFCANCEL_OCR_ENGINE=vlm
export PDFCANCEL_API_KEY="${PDFCANCEL_API_KEY:-EMPTY}"
export MISTRAL_API_KEY=""  # never send a hosted Mistral key anywhere from this skill
if [ -n "${PDFCANCEL_CA_BUNDLE:-}" ]; then
  [ -f "$PDFCANCEL_CA_BUNDLE" ] || { echo "CA bundle missing: $PDFCANCEL_CA_BUNDLE" >&2; exit 1; }
  export MISTRAL_CA_BUNDLE="$PDFCANCEL_CA_BUNDLE"
else
  export MISTRAL_CA_BUNDLE=""
fi

_resolve_models() {
  local m
  m="$("$PY" -I "$HELPER" resolve "$PDFCANCEL_VLM_MODEL")"
  export PDFCANCEL_VLM_MODEL="$m"
  m="$("$PY" -I "$HELPER" resolve "${PDFCANCEL_VLM_FULL_MODEL:-$PDFCANCEL_VLM_MODEL}")"
  export PDFCANCEL_VLM_FULL_MODEL="$m"
}

cmd="${1:-}"
case "$cmd" in
  check)
    exec "$PY" -I "$HELPER" check
    ;;
  convert)
    shift
    [ $# -ge 1 ] || { echo "usage: $0 convert <pdf|dir> [pdfcancel opts]" >&2; exit 64; }
    _resolve_models
    # Path first (CLI default-command routing), then defaults, then the
    # caller's options so theirs win.
    target="$1"; shift
    exec "$BIN" "$target" --full --preserve-pages --chunks \
      --chunker "${PDFCANCEL_CHUNKER:-hierarchical}" "$@"
    ;;
  ocr)
    shift
    [ $# -ge 1 ] || { echo "usage: $0 ocr <pdf|dir> [pdfcancel opts]" >&2; exit 64; }
    _resolve_models
    target="$1"; shift
    exec "$BIN" "$target" --preserve-pages --chunks \
      --chunker "${PDFCANCEL_CHUNKER:-hierarchical}" "$@"
    ;;
  ask)
    shift
    [ $# -ge 3 ] || { echo "usage: $0 ask <pdf> <page> \"prompt\" [--dpi N] [--model M]" >&2; exit 64; }
    _resolve_models
    exec "$PY" -I "$HELPER" ask "$@"
    ;;
  "")
    echo "usage: $0 check | convert <pdf|dir> | ocr <pdf|dir> | ask <pdf> <page> \"prompt\" | <pdfcancel args>" >&2
    exit 64
    ;;
  *)
    exec "$BIN" "$@"
    ;;
esac
