import { useMemo } from "react";
import {
  ArrowUpRight,
  Boxes,
  CircleAlert,
  Clock3,
  Cpu,
  Gauge,
  HardDrive,
  MemoryStick,
  ServerCog,
  Waypoints,
} from "lucide-react";
import { usePoll } from "../lib/usePoll";
import type { Overview } from "../lib/types";
import { bytes, clockTime, decimal, duration, ms, number, relativeTime, shortId, titleCase } from "../lib/format";
import { Badge, Button, Empty, LoadingBlock, Panel, Progress, Sparkline, Stat } from "../components/ui";
import type { PageId } from "../components/AppShell";

function statusTone(value: string): "good" | "warn" | "bad" | "info" {
  if (["healthy", "ready", "completed", "loaded", "success"].includes(value)) return "good";
  if (["degraded", "loading", "running", "queued"].includes(value)) return "warn";
  if (["failed", "crashed", "error", "unhealthy"].includes(value)) return "bad";
  return "info";
}

function TrafficBars({ series }: { series: Overview["series"] }) {
  const values = series.length ? series.map((point) => point.count) : [];
  if (!values.length) {
    return (
      <div className="traffic-empty">
        <div className="traffic-empty-rule" />
        <span>No request traffic recorded yet.</span>
      </div>
    );
  }
  const max = Math.max(1, ...values);
  return (
    <div className="traffic" aria-label="Requests per minute over the last 30 minutes">
      <div className="traffic-bars">
        {values.map((value, index) => (
          <div
            key={`${series[index].bucket}-${index}`}
            className="traffic-bar"
            style={{ height: `${Math.max(3, (value / max) * 100)}%` }}
            title={`${clockTime(series[index].bucket)} · ${value} request${value === 1 ? "" : "s"}`}
          />
        ))}
      </div>
      <div className="traffic-axis">
        <span>{clockTime(series[0]?.bucket)}</span>
        <span>{max} peak / min</span>
        <span>{clockTime(series[series.length - 1]?.bucket)}</span>
      </div>
    </div>
  );
}

export function OverviewPage({ onNavigate }: { onNavigate: (page: PageId) => void }) {
  const overview = usePoll<Overview>("/admin/api/overview", 3500);
  const data = overview.data;

  const sparkValues = useMemo(() => {
    if (!data?.series.length) return [];
    return data.series.map((point) => point.count);
  }, [data?.series]);

  if (!data) {
    return (
      <>
        {overview.error && <div className="error-banner">{overview.error}</div>}
        <div className="stats-grid">
          {Array.from({ length: 5 }).map((_, index) => (
            <div className="stat" key={index}>
              <div className="skeleton skeleton-label" />
              <div className="skeleton skeleton-value" />
            </div>
          ))}
        </div>
        <div className="overview-grid">
          <Panel title="Loading control room"><LoadingBlock rows={6} /></Panel>
          <Panel title="Resources"><LoadingBlock rows={5} /></Panel>
        </div>
      </>
    );
  }

  const { stats, resources, hardware, readiness } = data;
  const gpuTotal = resources.gpu_total_bytes;
  const gpuPercent = gpuTotal ? (resources.gpu_used_bytes / gpuTotal) * 100 : 0;
  const loaded = data.models.filter((model) => model.runtime.loaded);
  const maxResident = stats.max_resident_models || 1;

  return (
    <>
      {overview.error && (
        <div className="error-banner">
          <CircleAlert size={15} /> Live refresh failed: {overview.error}
        </div>
      )}

      <section className="status-ribbon">
        <div className="status-ribbon-main">
          <span className={`pulse ${readiness.ready ? "is-live" : "is-warn"}`} />
          <div>
            <strong>{readiness.ready ? "Inference gateway is ready" : "Gateway is not ready"}</strong>
            <span>
              {readiness.llama_binary ? "llama.cpp online" : "llama.cpp binary missing"} ·{" "}
              {hardware.backend.toUpperCase()} path · {readiness.available_models} models available
            </span>
          </div>
        </div>
        <div className="status-ribbon-meta">
          <div>
            <span>Uptime</span>
            <strong>{duration(data.app.uptime_s)}</strong>
          </div>
          <div>
            <span>Started</span>
            <strong>{clockTime(data.app.started_at * 1000)}</strong>
          </div>
          <div>
            <span>Build</span>
            <strong className="mono">{data.health.llama_version?.replace("version: ", "") || "not detected"}</strong>
          </div>
          <Badge tone={statusTone(data.health.status)} dot>{titleCase(data.health.status)}</Badge>
        </div>
      </section>

      <div className="stats-grid">
        <Stat
          label="Requests · 24h"
          value={number(stats.requests_last_24h)}
          detail={<><span>{number(stats.requests_last_minute)}</span> last minute</>}
        >
          <Sparkline values={sparkValues} label="Request trend" />
        </Stat>
        <Stat
          label="Average latency"
          value={ms(stats.avg_latency_ms)}
          detail={`${number(stats.prompt_tokens_24h + stats.completion_tokens_24h, true)} tokens processed`}
          tone={stats.avg_latency_ms && stats.avg_latency_ms > 8000 ? "warn" : undefined}
        />
        <Stat
          label="Token throughput"
          value={<>{decimal(stats.tokens_per_second_5m, 1)}<small> tok/s</small></>}
          detail={`${number(stats.tokens_last_5m, true)} tokens · 5 min`}
        />
        <Stat
          label="Resident models"
          value={<>{stats.models_loaded}<small> / {maxResident}</small></>}
          detail={`${stats.models_registered} registered · ${stats.models_enabled} enabled`}
        />
        <Stat
          label="In flight"
          value={stats.active_requests}
          detail={`${stats.queued_requests} queued · ${stats.errors_24h} errors today`}
          tone={stats.errors_24h > 0 ? "warn" : undefined}
        />
      </div>

      <div className="overview-grid">
        <div className="overview-main">
          <Panel
            eyebrow="Traffic"
            title="Request rate"
            actions={<span className="muted small mono">last 30 minutes</span>}
          >
            <TrafficBars series={data.series} />
          </Panel>

          <Panel
            title="Recent requests"
            actions={
              <Button variant="ghost" size="sm" onClick={() => onNavigate("logs")} icon={<ArrowUpRight size={14} />}>
                All logs
              </Button>
            }
            flush
          >
            {data.recent_requests.length ? (
              <div className="table-wrap">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Time</th>
                      <th>Endpoint</th>
                      <th>Model</th>
                      <th>Task</th>
                      <th>Status</th>
                      <th className="num">Latency</th>
                      <th className="num">Tokens</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.recent_requests.map((request) => (
                      <tr key={request.request_id}>
                        <td className="mono muted">{clockTime(request.created_at)}</td>
                        <td className="mono">{request.endpoint.replace("/v1/", "")}</td>
                        <td>
                          <span className="model-cell">{request.resolved_model || request.requested_model}</span>
                        </td>
                        <td><span className="muted">{request.task_category ? titleCase(request.task_category) : "—"}</span></td>
                        <td><Badge tone={statusTone(request.status)}>{titleCase(request.status)}</Badge></td>
                        <td className="num mono">{ms(request.total_ms)}</td>
                        <td className="num mono">{number(request.prompt_tokens + request.completion_tokens)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <Empty
                title="No requests yet"
                body={'Point an OpenAI-compatible client at the /v1 endpoint with model "auto". The first request will load a feasible model.'}
                action={<Button size="sm" onClick={() => onNavigate("api")}>Connection details</Button>}
              />
            )}
          </Panel>
        </div>

        <div className="overview-side">
          <Panel eyebrow="Pressure" title="System resources" actions={<Gauge size={16} className="muted" />}>
            <div className="resource-list">
              <div className="resource">
                <div className="resource-head">
                  <span><Cpu size={14} /> CPU</span>
                  <strong>{decimal(resources.cpu_percent, 0, "%")}</strong>
                </div>
                <Progress value={resources.cpu_percent} tone={resources.cpu_percent > 90 ? "bad" : resources.cpu_percent > 70 ? "warn" : "accent"} />
                <div className="resource-sub">{hardware.physical_cores} cores / {hardware.logical_cores} threads · load {decimal(resources.load_1m, 2)}</div>
              </div>
              <div className="resource">
                <div className="resource-head">
                  <span><MemoryStick size={14} /> Memory</span>
                  <strong>{decimal(resources.ram_percent, 0, "%")}</strong>
                </div>
                <Progress value={resources.ram_percent} tone={resources.ram_percent > 90 ? "bad" : resources.ram_percent > 75 ? "warn" : "accent"} />
                <div className="resource-sub">{bytes(resources.used_ram_bytes)} used · {bytes(resources.available_ram_bytes)} available</div>
              </div>
              <div className="resource">
                <div className="resource-head">
                  <span><ServerCog size={14} /> GPU</span>
                  <strong>{gpuTotal ? `${decimal(resources.gpu_utilization ?? gpuPercent, 0, "%")}` : "CPU only"}</strong>
                </div>
                <Progress value={gpuTotal ? Math.max(resources.gpu_utilization ?? 0, gpuPercent) : 0} tone={gpuPercent > 90 ? "bad" : "accent"} />
                <div className="resource-sub">
                  {gpuTotal ? `${bytes(resources.gpu_used_bytes)} / ${bytes(gpuTotal)} VRAM` : "No accelerator memory detected"}
                </div>
              </div>
              <div className="resource">
                <div className="resource-head">
                  <span><HardDrive size={14} /> Disk</span>
                  <strong>{bytes(resources.disk_free_bytes)}</strong>
                </div>
                <div className="resource-sub">free on the model volume</div>
              </div>
            </div>
            <div className="resource-foot">
              <span className="muted small">
                {String((data.recommendation as { tier?: string }).tier || "adaptive")} profile ·{" "}
                {String((data.recommendation as { recommended_model_size?: string }).recommended_model_size || "")}
              </span>
            </div>
          </Panel>

          <Panel
            eyebrow="Residency"
            title="Loaded models"
            actions={<Boxes size={16} className="muted" />}
            flush
          >
            {loaded.length ? (
              <div className="resident-list">
                {loaded.map((model) => (
                  <div className="resident-item" key={model.id}>
                    <div className="resident-top">
                      <div>
                        <div className="resident-name">{model.name}</div>
                        <div className="resident-meta">
                          {model.parameter_size} · {model.quantization} · ctx {number(model.runtime.context_length)}
                        </div>
                      </div>
                      <Badge tone="good" dot>resident</Badge>
                    </div>
                    <div className="resident-stats">
                      <span><Waypoints size={12} /> {model.runtime.active_requests} active</span>
                      <span>{model.runtime.queued_requests} queued</span>
                      <span>{bytes(model.runtime.estimated_memory?.ram_bytes)} est.</span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <Empty title="No resident models" body="Models load on demand and unload when idle. Nothing is pinned right now." />
            )}
            <div className="panel-inline-foot">
              <Button variant="ghost" size="sm" onClick={() => onNavigate("models")} icon={<ArrowUpRight size={14} />}>
                Manage residency
              </Button>
            </div>
          </Panel>

          <Panel eyebrow="Host" title="Hardware">
            <dl className="kv">
              <div><dt>CPU</dt><dd>{hardware.cpu_model}</dd></div>
              <div><dt>Platform</dt><dd>{hardware.os} {hardware.arch}</dd></div>
              <div><dt>Backend</dt><dd className="mono">{hardware.backend}</dd></div>
              <div><dt>Memory</dt><dd>{bytes(hardware.total_ram_bytes)}</dd></div>
              {hardware.gpus.map((gpu) => (
                <div key={gpu.name}><dt>GPU</dt><dd>{gpu.name}{gpu.total_mb ? ` · ${gpu.total_mb} MiB` : ""}</dd></div>
              ))}
            </dl>
            {hardware.warnings.length > 0 && (
              <div className="hardware-warning">
                <CircleAlert size={14} />
                <span>{hardware.warnings[0]}</span>
              </div>
            )}
          </Panel>
        </div>
      </div>

      <div className="overview-footline">
        <span><Clock3 size={13} /> Updated {relativeTime(overview.lastUpdated)}</span>
        <span className="mono">request {shortId(data.recent_requests[0]?.request_id)}</span>
      </div>
    </>
  );
}
