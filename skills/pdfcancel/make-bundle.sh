#!/usr/bin/env bash
# Build a shareable skill bundle for a team that uses the same endpoint.
#
#   ./make-bundle.sh --profile site.env [--ca CERT]... [-o bundle.tar.gz]
#
# The bundle holds the skill, the site profile, any CA certificates, and a
# pinned pdfcancel source archive (git HEAD), so recipients only run:
#   tar xzf bundle.tar.gz && ./pdfcancel-skill/install.sh
# The profile should not contain personal secrets: recipients get it as-is.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PROFILE="" OUT="" CAS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --ca) CAS+=("$2"); shift 2 ;;
    -o) OUT="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 64 ;;
  esac
done
[ -f "$PROFILE" ] || { echo "usage: $0 --profile site.env [--ca CERT]... [-o out.tar.gz]" >&2; exit 64; }
KEY="$( set -a; . "$PROFILE"; echo "${PDFCANCEL_API_KEY:-EMPTY}" )"
if [ "$KEY" != "EMPTY" ]; then
  echo "Refusing: $PROFILE sets a real PDFCANCEL_API_KEY. Recipients should add their own." >&2
  exit 65
fi
OUT="${OUT:-$REPO/dist/pdfcancel-skill-$(git -C "$REPO" rev-parse --short HEAD).tar.gz}"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
B="$STAGE/pdfcancel-skill"
mkdir -p "$B/ca"
cp -R "$HERE/SKILL.md" "$HERE/scripts" "$HERE/profile.env.example" "$HERE/install.sh" "$B/"
cp "$PROFILE" "$B/profile.env"
for c in ${CAS[@]+"${CAS[@]}"}; do cp "$c" "$B/ca/"; done
git -C "$REPO" archive --format=tar.gz --prefix=pdfcancel/ HEAD > "$B/pdfcancel-src.tar.gz"
cat > "$B/INSTALL.md" <<EOF
# pdfcancel agent skill

Built $(date +%Y-%m-%d) from pdfcancel $(git -C "$REPO" rev-parse --short HEAD).

    ./install.sh --profile profile.env

This installs pdfcancel into a virtualenv, the skill into ~/.agents/skills/pdfcancel
(linked into ~/.claude/skills), the site profile into ~/.config/pdfcancel/skill.env,
and any CA certificates in ca/ into ~/.config/pdfcancel/skill-ca.pem. It finishes
with an endpoint check. Agents then follow SKILL.md.
EOF
find "$B" -name __pycache__ -prune -exec rm -rf {} +
mkdir -p "$(dirname "$OUT")"
tar -C "$STAGE" -czf "$OUT" pdfcancel-skill
echo "$OUT"
