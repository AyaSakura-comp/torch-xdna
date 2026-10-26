#!/bin/bash
# Build tools/npu_trace/peano_wrap: a Peano (llvm-aie) root whose opt and llc are wrappers, so a
# Triton-XDNA build can log the core compiler calls and change them without rebuilding aiecc.
#   PEANO_WRAP_LOG       log every opt / llc call
#   PEANO_OPT_PRE_SED    sed -E expression applied in place to opt's input .ll first
#                        (e.g. the AIE2P crSat index fix, see results/npu_profile.md)
#   PEANO_OPT_EXTRA      appended to opt's arguments
#   PEANO_OPT_POST       a second opt pass pipeline over opt's output (PEANO_OPT_POST_FLAGS options)
#   PEANO_LLC_REAL       run this llc instead of the wheel's (e.g. a patched build)
#   PEANO_LLC_EXTRA      appended to llc's arguments
# usage: tools/npu_trace/make_peano_wrap.sh [peano root, default: the sandbox llvm-aie wheel]
set -e
here=$(cd "$(dirname "$0")" && pwd)
real=${1:-$HOME/src/Triton-XDNA/sandbox/lib/python3.13/site-packages/llvm-aie}
W=$here/peano_wrap
rm -rf "$W"; mkdir -p "$W/bin"
for d in include lib share; do ln -s "$real/$d" "$W/$d"; done
for f in "$real"/bin/*; do b=$(basename "$f"); [ "$b" = opt ] || [ "$b" = llc ] || ln -s "$f" "$W/bin/$b"; done
cat > "$W/bin/opt" <<OPT
#!/bin/bash
[ -n "\$PEANO_WRAP_LOG" ] && echo "opt \$*" >> "\$PEANO_WRAP_LOG"
REAL=$real/bin/opt
out=""; in=""; prev=""
for a in "\$@"; do
  [ "\$prev" = "-o" ] && out="\$a"
  case "\$a" in *.ll) [ "\$prev" != "-o" ] && in="\$a";; esac
  prev="\$a"
done
if [ -n "\$PEANO_OPT_PRE_SED" ] && [ -n "\$in" ]; then sed -E -i "\$PEANO_OPT_PRE_SED" "\$in"; fi
\$REAL "\$@" \$PEANO_OPT_EXTRA || exit \$?
if [ -n "\$PEANO_OPT_POST" ] && [ -n "\$out" ]; then
  \$REAL -S --passes="\$PEANO_OPT_POST" \$PEANO_OPT_POST_FLAGS "\$out" -o "\$out.post.ll" && mv "\$out.post.ll" "\$out"
fi
OPT
cat > "$W/bin/llc" <<LLC
#!/bin/bash
[ -n "\$PEANO_WRAP_LOG" ] && echo "llc \$*" >> "\$PEANO_WRAP_LOG"
exec \${PEANO_LLC_REAL:-$real/bin/llc} "\$@" \$PEANO_LLC_EXTRA
LLC
chmod +x "$W/bin/opt" "$W/bin/llc"
echo "$W"
