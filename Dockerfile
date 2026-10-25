# ── Stage 1: build the React UI ───────────────────────────────────
FROM node:20-slim AS ui-builder
ARG CODEX_VERSION=0.154.0
WORKDIR /ui
COPY ui/mira/package.json ui/mira/package-lock.json ./
RUN npm ci --no-audit --no-fund
RUN npm install --global --no-audit --no-fund "@openai/codex@${CODEX_VERSION}"
COPY ui/mira/ ./
RUN npm run build

# Claude Code CLI: the newest release on every build. ADD re-downloads the
# npm registry's manifest for the tag on every build and invalidates the cache
# only when its content changes, so a new Claude Code release busts the cache
# for this layer automatically and an unchanged one reuses it. The install
# then uses the exact version from that manifest. Override with
# --build-arg CLAUDE_CODE_VERSION=<version> to pin, or build with --no-cache to
# force everything fresh.
ARG CLAUDE_CODE_VERSION=latest
ADD https://registry.npmjs.org/@anthropic-ai/claude-code/${CLAUDE_CODE_VERSION} /tmp/claude-code-release.json
RUN npm install --global --no-audit --no-fund \
      "@anthropic-ai/claude-code@$(node -p "require('/tmp/claude-code-release.json').version")"

# ── Stage 2: backend + bundled UI ─────────────────────────────────
FROM python:3.12-slim
LABEL org.opencontainers.image.source="https://github.com/miracodeai/mira"
LABEL org.opencontainers.image.description="Self-hostable AI code reviewer"
LABEL org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir "/app[serve,bedrock]"

# Codex CLI and Claude Code CLI are bundled for the optional codex-cli and
# claude-cli backends. Codex is pinned above; Claude Code tracks the newest
# release (see stage 1) and Mira logs its version at startup.
COPY --from=ui-builder /usr/local/bin/node /usr/local/bin/node
COPY --from=ui-builder /usr/local/lib/node_modules/@openai/codex /usr/local/lib/node_modules/@openai/codex
RUN ln -s /usr/local/lib/node_modules/@openai/codex/bin/codex.js /usr/local/bin/codex
# The Claude Code package installs a self-contained native binary; only it is needed at runtime.
COPY --from=ui-builder /usr/local/lib/node_modules/@anthropic-ai/claude-code/bin/claude.exe /usr/local/bin/claude
RUN claude --version

# Pull the built UI in from stage 1. webhooks.create_app() picks this up
# automatically and serves it at / with SPA fallback.
COPY --from=ui-builder /ui/dist /app/ui_dist

EXPOSE 8000
# ENTRYPOINT (not CMD) so `docker run … image --config /app/mira.yaml`
# appends the args to `mira serve` instead of replacing the command.
ENTRYPOINT ["mira", "serve"]
