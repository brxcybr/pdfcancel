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
#   pdfcancel-skill.sh update                     # latest pdfcancel + skill from the repo
#   pdfcancel-skill.sh <pdfcancel args...>        # passthrough (search, status, ...)
set -euo pipefail

PROFILE="${PDFCANCEL_SKILL_PROFILE:-$HOME/.config/pdfcancel/skill.env}"
[ -f "$PROFILE" ] || {
  echo "No skill profile at $PROFILE. Copy profile.env.example there and fill it in," >&2
  echo "or run install.sh (see SKILL.md, \"Setup on a new workstation\")." >&2
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

# update: fetch the latest repository and re-run its installer, which upgrades
# pdfcancel in the venv and refreshes these skill files. The profile and CA
# bundle are kept (no --profile/--ca is passed).
if [ "${1:-}" = "update" ]; then
  SKILL_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  if [ -f "$SKILL_DIR/../../pyproject.toml" ]; then
    echo "This skill is linked to a checkout ($(cd "$SKILL_DIR/../.." && pwd))." >&2
    echo "Update it there: git pull && make install" >&2
    exit 0
  fi
  REPO="${PDFCANCEL_REPO:-https://github.com/brxcybr/pdfcancel}"
  REF="${PDFCANCEL_REF:-main}"
  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  echo "Fetching $REPO ($REF) ..."
  if command -v git >/dev/null; then
    git clone --quiet --depth 1 --branch "$REF" "$REPO" "$TMP/src"
  else
    case "$REPO" in
      https://github.com/*)
        mkdir -p "$TMP/src"
        curl -fsSL "https://codeload.github.com/${REPO#https://github.com/}/tar.gz/$REF" |
          tar -xz -C "$TMP/src" --strip-components 1 ;;
      *) echo "update needs git to fetch $REPO" >&2; exit 69 ;;
    esac
  fi
  PDFCANCEL_REPO="$REPO" PDFCANCEL_REF="$REF" \
    bash "$TMP/src/skills/pdfcancel/install.sh" --venv "$PDFCANCEL_VENV"
  exit $?
fi
[ -x "$BIN" ] || {
  echo "pdfcancel not installed in $PDFCANCEL_VENV. Run the skill's install.sh, or:" >&2
  echo "  git clone https://github.com/brxcybr/pdfcancel && pdfcancel/skills/pdfcancel/install.sh" >&2
  exit 127
}

export PDFCANCEL_OCR_ENGINE=vlm
export PDFCANCEL_API_KEY="${PDFCANCEL_API_KEY:-EMPTY}"
export MISTRAL_API_KEY=""  # never send a hosted Mistral key anywhere from this skill
# Fall back to the CA bundle install.sh writes, if the profile names none.
if [ -z "${PDFCANCEL_CA_BUNDLE:-}" ] && [ -f "$HOME/.config/pdfcancel/skill-ca.pem" ]; then
  PDFCANCEL_CA_BUNDLE="$HOME/.config/pdfcancel/skill-ca.pem"
fi
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
    echo "usage: $0 check | convert <pdf|dir> | ocr <pdf|dir> | ask <pdf> <page> \"prompt\" | update | <pdfcancel args>" >&2
    exit 64
    ;;
  *)
    exec "$BIN" "$@"
    ;;
esac
