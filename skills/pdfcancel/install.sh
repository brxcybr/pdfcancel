#!/usr/bin/env bash
# Install the pdfcancel agent skill for every local agent that reads
# ~/.agents/skills (and Claude Code via ~/.claude/skills).
#
#   ./install.sh [--profile FILE] [--ca CERT]... [--venv DIR] [--no-venv] [--link]
#
#   --profile FILE  site profile to install as ~/.config/pdfcancel/skill.env
#                   (an existing one is backed up). Default: keep the current
#                   profile, or start from profile.env.example.
#   --ca CERT       CA certificate(s) concatenated into
#                   ~/.config/pdfcancel/skill-ca.pem (repeatable). A bundle's
#                   ca/ directory is used automatically.
#   --venv DIR      where to create the pdfcancel virtualenv
#                   (default: the profile's PDFCANCEL_VENV, else
#                   ~/.local/share/pdfcancel/venv).
#   --no-venv       don't create or update a virtualenv (use an existing one).
#   --link          symlink the skill instead of copying it (for development
#                   from a checkout, so edits apply immediately).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF="$HOME/.config/pdfcancel"
SKILLS="$HOME/.agents/skills"
PROFILE_SRC="" VENV="" MAKE_VENV=1 LINK=0
CAS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --profile) PROFILE_SRC="$2"; shift 2 ;;
    --ca) CAS+=("$2"); shift 2 ;;
    --venv) VENV="$2"; shift 2 ;;
    --no-venv) MAKE_VENV=0; shift ;;
    --link) LINK=1; shift ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 64 ;;
  esac
done
[ -d "$HERE/ca" ] && for c in "$HERE"/ca/*; do [ -f "$c" ] && CAS+=("$c"); done

mkdir -p "$CONF" "$SKILLS"

# 1. Profile.
if [ -n "$PROFILE_SRC" ]; then
  [ -f "$CONF/skill.env" ] && cp "$CONF/skill.env" "$CONF/skill.env.bak.$(date +%Y%m%d%H%M%S)"
  cp "$PROFILE_SRC" "$CONF/skill.env"
elif [ ! -f "$CONF/skill.env" ]; then
  cp "$HERE/profile.env.example" "$CONF/skill.env"
  echo "Created $CONF/skill.env from the example: edit it before use."
fi
chmod 600 "$CONF/skill.env"

# 2. CA bundle (verification stays on; this only adds trust for your CA).
if [ ${#CAS[@]} -gt 0 ]; then
  cat ${CAS[@]+"${CAS[@]}"} > "$CONF/skill-ca.pem"
  echo "Wrote $CONF/skill-ca.pem from ${#CAS[@]} certificate file(s)."
fi

# 3. Virtualenv with pdfcancel[vlm,chunking].
if [ -z "$VENV" ]; then
  VENV="$( set -a; . "$CONF/skill.env"; echo "${PDFCANCEL_VENV:-}" )"
  VENV="${VENV:-$HOME/.local/share/pdfcancel/venv}"
fi
if [ "$MAKE_VENV" = 1 ]; then
  PYBIN=""
  for p in python3.12 python3.13 python3.11 python3.10 python3; do
    if command -v "$p" >/dev/null && "$p" -c 'import sys; sys.exit(not (3, 10) <= sys.version_info < (3, 14))'; then
      PYBIN="$p"; break
    fi
  done
  [ -n "$PYBIN" ] || { echo "Need Python 3.10–3.13 on PATH." >&2; exit 69; }
  [ -x "$VENV/bin/python" ] || "$PYBIN" -m venv "$VENV"
  if [ -f "$HERE/pdfcancel-src.tar.gz" ]; then
    SRC="$HERE/pdfcancel-src.tar.gz"            # pinned source shipped in a bundle
  elif [ -f "$HERE/../../pyproject.toml" ]; then
    SRC="$(cd "$HERE/../.." && pwd)"             # running from a checkout
  else
    SRC="git+https://github.com/brxcybr/pdfcancel"
  fi
  echo "Installing pdfcancel from $SRC into $VENV ..."
  "$VENV/bin/python" -m pip install --quiet --upgrade pip
  case "$SRC" in git+*) URL="$SRC" ;; *) URL="file://$SRC" ;; esac
  "$VENV/bin/python" -m pip install --quiet "pdfcancel[vlm,chunking] @ $URL"
fi

# 4. Skill files, shared by every agent; Claude Code reads ~/.claude/skills.
DEST="$SKILLS/pdfcancel"
rm -rf "$DEST"
if [ "$LINK" = 1 ]; then
  ln -s "$HERE" "$DEST"
else
  mkdir -p "$DEST"
  cp -R "$HERE/SKILL.md" "$HERE/scripts" "$HERE/profile.env.example" "$DEST/"
fi
if [ -d "$HOME/.claude" ]; then
  mkdir -p "$HOME/.claude/skills"
  ln -sfn "../../.agents/skills/pdfcancel" "$HOME/.claude/skills/pdfcancel"
fi
echo "Installed skill → $DEST"

# 5. Smoke test the endpoint.
if "$DEST/scripts/pdfcancel-skill.sh" check; then
  echo "Endpoint OK."
else
  echo "Endpoint check failed: fix $CONF/skill.env, then run $DEST/scripts/pdfcancel-skill.sh check" >&2
  exit 1
fi
