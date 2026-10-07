# ── Stage 1: build the React UI ───────────────────────────────────
FROM node:20-slim AS ui-builder
ARG CODEX_VERSION=0.154.0
ARG CLAUDE_CODE_VERSION=2.1.270
WORKDIR /ui
COPY ui/mira/package.json ui/mira/package-lock.json ./
RUN npm ci --no-audit --no-fund
RUN npm install --global --no-audit --no-fund "@openai/codex@${CODEX_VERSION}"
RUN npm install --global --no-audit --no-fund "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}"
COPY ui/mira/ ./
RUN npm run build

# ── Stage 2: backend + bundled UI ─────────────────────────────────
FROM python:3.12-slim
LABEL org.opencontainers.image.source="https://github.com/miracodeai/mira"
LABEL org.opencontainers.image.description="Self-hostable AI code reviewer"
LABEL org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir "/app[serve,bedrock]"

# Codex CLI and Claude Code CLI are bundled for the optional codex-cli and
# claude-cli backends. Keep the versions pinned above so image rebuilds do not
# silently change the review runtime.
COPY --from=ui-builder /usr/local/bin/node /usr/local/bin/node
COPY --from=ui-builder /usr/local/lib/node_modules/@openai/codex /usr/local/lib/node_modules/@openai/codex
RUN ln -s /usr/local/lib/node_modules/@openai/codex/bin/codex.js /usr/local/bin/codex
# The Claude Code package installs a self-contained native binary; only it is needed at runtime.
COPY --from=ui-builder /usr/local/lib/node_modules/@anthropic-ai/claude-code/bin/claude.exe /usr/local/bin/claude

# Pull the built UI in from stage 1. webhooks.create_app() picks this up
# automatically and serves it at / with SPA fallback.
COPY --from=ui-builder /ui/dist /app/ui_dist

EXPOSE 8000
# ENTRYPOINT (not CMD) so `docker run … image --config /app/mira.yaml`
# appends the args to `mira serve` instead of replacing the command.
ENTRYPOINT ["mira", "serve"]
