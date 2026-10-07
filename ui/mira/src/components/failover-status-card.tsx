import { Loader2, RefreshCw } from "lucide-react"
import { useCallback, useEffect, useState } from "react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { api } from "@/lib/api"
import type { FailoverStatus, FailoverTier } from "@/lib/api/settings"

const BACKEND_LABELS: Record<string, string> = {
  zai: "Z.AI",
  openrouter: "OpenRouter",
  bedrock: "AWS Bedrock",
  "codex-cli": "Codex CLI",
  "claude-cli": "Claude Code CLI",
  "openai-compatible": "OpenAI-compatible endpoint",
}

function formatDuration(totalSeconds: number): string {
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  if (minutes === 0) return `${seconds}s`
  return seconds === 0 ? `${minutes}m` : `${minutes}m ${seconds}s`
}

function TierModels({ tier }: { tier: FailoverTier }) {
  if (tier.review_model === tier.indexing_model) {
    return <>Model: {tier.review_model}</>
  }
  return (
    <>
      Review: {tier.review_model} · Indexing: {tier.indexing_model}
    </>
  )
}

export function FailoverStatusCard() {
  const [status, setStatus] = useState<FailoverStatus | null>(null)
  const [fetchedAt, setFetchedAt] = useState(0)
  const [now, setNow] = useState(() => Date.now())
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchStatus = useCallback(
    () =>
      api
        .getFailoverStatus()
        .then((s) => {
          const fetched = Date.now()
          setStatus(s)
          setFetchedAt(fetched)
          setNow(fetched)
          setError(null)
        })
        .catch(() => setError("Couldn't load failover status.")),
    []
  )

  useEffect(() => {
    fetchStatus()
  }, [fetchStatus])

  const refresh = () => {
    setRefreshing(true)
    fetchStatus().finally(() => setRefreshing(false))
  }

  const elapsed = Math.floor((now - fetchedAt) / 1000)
  const remaining = (tier: FailoverTier) =>
    Math.max(0, tier.cooldown_remaining_seconds - elapsed)
  const tiers = status?.tiers ?? []
  const anyCooling = tiers.some((tier) => remaining(tier) > 0)
  // The next call goes to the first tier that isn't cooling down; when every
  // tier is cooling, they're tried in order.
  const servingTier =
    tiers.find((tier) => remaining(tier) === 0)?.tier ?? tiers[0]?.tier

  useEffect(() => {
    if (!anyCooling) return
    const id = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(id)
  }, [anyCooling])

  return (
    <Card>
      <CardHeader>
        <CardTitle>Failover</CardTitle>
        <CardDescription>
          {status && !status.enabled ? (
            <>
              Not configured. Add an{" "}
              <code className="text-xs">llm.failover</code> block to{" "}
              <code className="text-xs">mira.yaml</code> to fall back to a
              second provider.
            </>
          ) : (
            <>
              When a provider is rate limited, down, or rejecting its
              credentials, calls move to the next tier. Configured in{" "}
              <code className="text-xs">mira.yaml</code>.
            </>
          )}
        </CardDescription>
        <CardAction>
          <Button
            variant="outline"
            size="sm"
            onClick={refresh}
            disabled={refreshing}
            aria-label="Refresh failover status"
          >
            {refreshing ? (
              <Loader2 className="h-3 w-3 animate-spin" />
            ) : (
              <RefreshCw className="h-3 w-3" />
            )}
            Refresh
          </Button>
        </CardAction>
      </CardHeader>
      <CardContent className="space-y-3">
        {error && (
          <p className="text-xs text-destructive">
            {status
              ? "Couldn't refresh failover status; showing the last loaded status."
              : error}
          </p>
        )}
        {!status && !error && (
          <p className="text-xs text-muted-foreground">Loading…</p>
        )}
        {tiers.length > 0 && (
          <ol className="divide-y rounded-md border">
            {tiers.map((tier) => {
              const cooling = remaining(tier)
              return (
                <li
                  key={tier.tier}
                  className="flex flex-wrap items-center justify-between gap-2 px-3 py-2.5"
                >
                  <div className="min-w-0">
                    <p className="text-sm font-medium">
                      <span className="mr-2 text-xs font-normal text-muted-foreground">
                        Tier {tier.tier}
                      </span>
                      {BACKEND_LABELS[tier.backend] ?? tier.backend}
                    </p>
                    <p className="truncate text-xs text-muted-foreground">
                      <TierModels tier={tier} />
                    </p>
                  </div>
                  {cooling > 0 ? (
                    <Badge variant="destructive">
                      Cooling down · {formatDuration(cooling)}
                    </Badge>
                  ) : tier.tier === servingTier ? (
                    <Badge>Serving</Badge>
                  ) : (
                    <Badge variant="outline">Standby</Badge>
                  )}
                </li>
              )
            })}
          </ol>
        )}
        {status?.enabled && (
          <p className="text-xs text-muted-foreground">
            After a rate limit, outage, auth error, or timeout, a tier is
            skipped for {formatDuration(status.cooldown_seconds)}. The primary
            retries {status.primary_max_retries}{" "}
            {status.primary_max_retries === 1 ? "time" : "times"} before failing
            over. Cooldowns reset when Mira restarts.
          </p>
        )}
      </CardContent>
    </Card>
  )
}
