#!/usr/bin/env bash
# Build a shareable skill bundle for a team that uses the same endpoint.
#
#   ./make-bundle.sh --profile site.env [--ca CERT]... [--pin] [-o bundle.tar.gz]
#
# The bundle holds the skill, the site profile, and any CA certificates. The
# installer pulls the latest pdfcancel from the public repository, and
# `pdfcancel-skill.sh update` refreshes pdfcancel and the skill later, so a
# bundle only needs resharing when the site settings change. --pin also ships
# a pdfcancel source archive (git HEAD) for machines that can't reach the
# repository; pinned installs don't follow updates. Recipients only run:
#   tar xzf bundle.tar.gz && ./pdfcancel-skill/install.sh
# (install.sh picks up the bundled profile.env and ca/ on its own.)
# Recipients get the profile as-is, so the builder refuses one carrying secrets.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PROFILE="" OUT="" CAS=() PIN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --ca) CAS+=("$2"); shift 2 ;;
    -o) OUT="$2"; shift 2 ;;
    --pin) PIN=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 64 ;;
  esac
done
[ -f "$PROFILE" ] || { echo "usage: $0 --profile site.env [--ca CERT]... [--pin] [-o out.tar.gz]" >&2; exit 64; }
# Refuse anything secret-shaped: a non-placeholder value for any *KEY*,
# *TOKEN*, *SECRET*, *PASS* variable, or any URL with embedded credentials.
VARS="$(sed -nE 's/^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=.*/\2/p' "$PROFILE" | sort -u)"
BAD=""
for v in $VARS; do
  val="$( set -a; . "$PROFILE"; eval "printf '%s' \"\${$v:-}\"" )"
  case "$(printf '%s' "$v" | tr '[:lower:]' '[:upper:]')" in
    *KEY*|*TOKEN*|*SECRET*|*PASS*)
      case "$val" in ""|EMPTY|none|NONE) ;; *) BAD="$BAD $v" ;; esac ;;
  esac
  printf '%s' "$val" | grep -Eq '://[^/@[:space:]]+@' && BAD="$BAD $v(url-credentials)"
done
if [ -n "$BAD" ]; then
  echo "Refusing: $PROFILE carries secrets:$BAD. Recipients should add their own." >&2
  exit 65
fi
OUT="${OUT:-$REPO/dist/pdfcancel-skill.tar.gz}"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
B="$STAGE/pdfcancel-skill"
mkdir -p "$B/ca"
cp -R "$HERE/SKILL.md" "$HERE/scripts" "$HERE/profile.env.example" "$HERE/install.sh" "$B/"
cp "$PROFILE" "$B/profile.env"
for c in ${CAS[@]+"${CAS[@]}"}; do cp "$c" "$B/ca/"; done
if [ "$PIN" = 1 ]; then
  git -C "$REPO" archive --format=tar.gz --prefix=pdfcancel/ HEAD > "$B/pdfcancel-src.tar.gz"
  SOURCE_NOTE="pdfcancel is pinned to $(git -C "$REPO" rev-parse --short HEAD) (bundled source; \`update\` will move it to the latest)."
else
  SOURCE_NOTE="pdfcancel is installed from the latest https://github.com/brxcybr/pdfcancel (main)."
fi
cat > "$B/INSTALL.md" <<EOF
# pdfcancel agent skill

Built $(date +%Y-%m-%d). $SOURCE_NOTE

    ./pdfcancel-skill/install.sh

(from the directory you extracted into). This installs pdfcancel into a virtualenv, the skill into ~/.agents/skills/pdfcancel
(linked into ~/.claude/skills), the site profile into ~/.config/pdfcancel/skill.env,
and any CA certificates in ca/ into ~/.config/pdfcancel/skill-ca.pem. It finishes
with an endpoint check. Agents then follow SKILL.md.

To get the latest pdfcancel and skill later (keeps your profile and CA):

    ~/.agents/skills/pdfcancel/scripts/pdfcancel-skill.sh update

Source and updates: https://github.com/brxcybr/pdfcancel (skills/pdfcancel).
EOF
find "$B" -name __pycache__ -prune -exec rm -rf {} +
mkdir -p "$(dirname "$OUT")"
tar -C "$STAGE" -czf "$OUT" pdfcancel-skill
echo "$OUT"
