import { useState } from "react";
import {
  AlertTriangle,
  ChevronRight,
  CircleDot,
  Clock3,
  FileText,
  RefreshCw,
  ScrollText,
  TerminalSquare,
} from "lucide-react";
import { api } from "../lib/api";
import { usePoll } from "../lib/usePoll";
import { qs } from "../lib/api";
import type { EventRow, RequestRow } from "../lib/types";
import { clockTime, dateTime, ms, number, relativeTime, shortId, titleCase } from "../lib/format";
import { Badge, Button, CopyButton, Empty, LoadingBlock, Modal, Panel, Select } from "../components/ui";

type Tab = "requests" | "events";

function statusTone(value: string): "good" | "warn" | "bad" | "neutral" | "info" {
  if (["completed", "success"].includes(value)) return "good";
  if (["running", "pending"].includes(value)) return "warn";
  if (["failed", "error"].includes(value)) return "bad";
  if (["cancelled"].includes(value)) return "warn";
  return "neutral";
}

function levelTone(value: string): "good" | "warn" | "bad" | "info" {
  if (value === "error") return "bad";
  if (value === "warning" || value === "warn") return "warn";
  if (value === "info") return "info";
  return "good";
}

export function LogsPage() {
  const [tab, setTab] = useState<Tab>("requests");
  const [status, setStatus] = useState("");
  const [level, setLevel] = useState("");
  const [detail, setDetail] = useState<RequestRow | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [expandedEvent, setExpandedEvent] = useState<number | null>(null);

  const requests = usePoll<{ requests: RequestRow[] }>(
    `/admin/api/logs/requests${qs({ limit: 150, status })}`,
    4000,
    tab === "requests",
  );
  const events = usePoll<{ events: EventRow[] }>(
    `/admin/api/logs/events${qs({ limit: 150, level })}`,
    4000,
    tab === "events",
  );

  const openDetail = async (row: RequestRow) => {
    setDetail(row);
    setDetailLoading(true);
    try {
      const response = await api<RequestRow>(`/admin/api/logs/requests/${row.request_id}`);
      setDetail(response);
    } catch {
      // Keep the summary row if the detail fetch fails.
    } finally {
      setDetailLoading(false);
    }
  };

  const refresh = () => {
    if (tab === "requests") void requests.refresh();
    else void events.refresh();
  };

  return (
    <>
      <div className="page-toolbar">
        <div className="segmented">
          <button type="button" className={tab === "requests" ? "is-active" : ""} onClick={() => setTab("requests")}>
            <ScrollText size={14} /> Requests
          </button>
          <button type="button" className={tab === "events" ? "is-active" : ""} onClick={() => setTab("events")}>
            <TerminalSquare size={14} /> Backend events
          </button>
        </div>
        <div className="toolbar-actions">
          <Select
            value={tab === "requests" ? status : level}
            onChange={(event) => tab === "requests" ? setStatus(event.target.value) : setLevel(event.target.value)}
            className="filter-select"
            aria-label="Filter logs"
          >
            <option value="">All {tab === "requests" ? "statuses" : "levels"}</option>
            {tab === "requests" ? (
              <>
                <option value="completed">Completed</option>
                <option value="failed">Failed</option>
                <option value="cancelled">Cancelled</option>
                <option value="running">Running</option>
              </>
            ) : (
              <>
                <option value="info">Info</option>
                <option value="warning">Warning</option>
                <option value="error">Error</option>
              </>
            )}
          </Select>
          <Button variant="secondary" icon={<RefreshCw size={14} className={(tab === "requests" ? requests.loading : events.loading) ? "spin" : ""} />} onClick={refresh}>
            Refresh
          </Button>
        </div>
      </div>

      {tab === "requests" ? (
        <Panel
          title="Request history"
          eyebrow="Last 150 records"
          actions={<span className="muted small"><Clock3 size={13} /> live · {relativeTime(requests.lastUpdated)}</span>}
          flush
        >
          {requests.error && <div className="error-banner inset">{requests.error}</div>}
          {!requests.data ? (
            <LoadingBlock rows={8} />
          ) : requests.data.requests.length ? (
            <div className="table-wrap">
              <table className="data-table logs-table">
                <thead>
                  <tr>
                    <th>Time</th>
                    <th>Request</th>
                    <th>Endpoint</th>
                    <th>Requested → resolved</th>
                    <th>Task</th>
                    <th>Status</th>
                    <th className="num">Queue / load</th>
                    <th className="num">TTFT</th>
                    <th className="num">Latency</th>
                    <th className="num">Tokens</th>
                    <th aria-label="Open" />
                  </tr>
                </thead>
                <tbody>
                  {requests.data.requests.map((row) => (
                    <tr key={row.request_id} onClick={() => void openDetail(row)} className="clickable-row">
                      <td className="mono muted" title={dateTime(row.created_at)}>{clockTime(row.created_at)}</td>
                      <td className="mono">{shortId(row.request_id, 12)}</td>
                      <td className="mono">{row.endpoint.replace("/v1/", "")}</td>
                      <td>
                        <span className="muted">{row.requested_model}</span>
                        <ChevronRight size={11} className="inline-chev" />
                        <span className="model-cell">{row.resolved_model || "—"}</span>
                      </td>
                      <td><span className="muted">{row.task_category ? titleCase(row.task_category) : "—"}</span></td>
                      <td><Badge tone={statusTone(row.status)}>{titleCase(row.status)}</Badge></td>
                      <td className="num">
                        <div className="mono">{ms(row.queue_ms)}</div>
                        <div className="cell-sub">{row.load_ms ? `load ${ms(row.load_ms)}` : "warm"}</div>
                      </td>
                      <td className="num mono">{ms(row.ttft_ms)}</td>
                      <td className="num mono">{ms(row.total_ms)}</td>
                      <td className="num mono">{number(row.prompt_tokens + row.completion_tokens)}</td>
                      <td><ChevronRight size={14} className="muted" /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <Empty title="No matching requests" body="Requests appear here after they reach the inference API." />
          )}
        </Panel>
      ) : (
        <Panel
          title="Backend events"
          eyebrow="Lifecycle, routing and failures"
          actions={<span className="muted small">live · {relativeTime(events.lastUpdated)}</span>}
          flush
        >
          {events.error && <div className="error-banner inset">{events.error}</div>}
          {!events.data ? (
            <LoadingBlock rows={8} />
          ) : events.data.events.length ? (
            <div className="event-list">
              {events.data.events.map((event) => (
                <div key={event.id} className={`event-row level-${event.level}`}>
                  <div className="event-icon">
                    {event.level === "error" || event.level === "warning" ? <AlertTriangle size={15} /> : <CircleDot size={15} />}
                  </div>
                  <div className="event-main">
                    <div className="event-head">
                      <Badge tone={levelTone(event.level)}>{event.level}</Badge>
                      <span className="event-name mono">{event.event}</span>
                      {event.model_id && <span className="event-model">{event.model_id}</span>}
                      <span className="event-time">{relativeTime(event.created_at)}</span>
                    </div>
                    <div className="event-message">{event.message}</div>
                    {Object.keys(event.metadata || {}).length > 0 && (
                      <button className="event-toggle" onClick={() => setExpandedEvent(expandedEvent === event.id ? null : event.id)}>
                        <FileText size={12} /> {expandedEvent === event.id ? "Hide metadata" : "View metadata"}
                      </button>
                    )}
                    {expandedEvent === event.id && <pre className="metadata-view">{JSON.stringify(event.metadata, null, 2)}</pre>}
                  </div>
                  <span className="event-time-abs mono">{clockTime(event.created_at)}</span>
                </div>
              ))}
            </div>
          ) : (
            <Empty title="No matching events" body="Model loads, evictions, crashes and tunnel changes are logged here." />
          )}
        </Panel>
      )}

      <Modal
        open={Boolean(detail)}
        title={detail ? `Request ${shortId(detail.request_id, 16)}` : "Request"}
        subtitle={detail ? `${detail.endpoint} · ${detail.requested_model} → ${detail.resolved_model || "unresolved"}` : undefined}
        onClose={() => setDetail(null)}
        wide
      >
        {!detail ? null : detailLoading ? <LoadingBlock rows={6} /> : (
          <div className="request-detail">
            <div className="detail-grid">
              <div><span>Status</span><Badge tone={statusTone(detail.status)}>{titleCase(detail.status)}</Badge></div>
              <div><span>HTTP</span><strong className="mono">{detail.http_status || "—"}</strong></div>
              <div><span>Task</span><strong>{detail.task_category ? titleCase(detail.task_category) : "—"}</strong></div>
              <div><span>Streaming</span><strong>{detail.stream ? "yes" : "no"}</strong></div>
              <div><span>Queue wait</span><strong className="mono">{ms(detail.queue_ms)}</strong></div>
              <div><span>Model load wait</span><strong className="mono">{ms(detail.load_ms)}</strong></div>
              <div><span>TTFT</span><strong className="mono">{ms(detail.ttft_ms)}</strong></div>
              <div><span>Total</span><strong className="mono">{ms(detail.total_ms)}</strong></div>
              <div><span>Tokens</span><strong className="mono">{number(detail.prompt_tokens)} in / {number(detail.completion_tokens)} out</strong></div>
              <div><span>Started</span><strong>{dateTime(detail.created_at)}</strong></div>
              <div><span>Completed</span><strong>{dateTime(detail.completed_at)}</strong></div>
            </div>
            {detail.error && <div className="form-alert is-warn"><AlertTriangle size={14} /> {detail.error}</div>}
            {detail.routing && (
              <div className="routing-trace">
                <div className="routing-trace-head">
                  <h3>Routing trace</h3>
                  <Badge tone={detail.routing.fallback_used ? "warn" : "good"}>{detail.routing.fallback_used ? "fallback used" : "direct"}</Badge>
                  <span className="mono">{ms(detail.routing.routing_ms)}</span>
                  <span className="mono">confidence {Math.round(detail.routing.confidence * 100)}%</span>
                </div>
                <p>{detail.routing.explanation}</p>
                <div className="detail-grid">
                  <div><span>Classifier</span><strong>{detail.routing.classifier_source}</strong></div>
                  <div><span>Strategy</span><strong>{detail.routing.strategy}</strong></div>
                  <div><span>Capability</span><strong>{detail.routing.capability}</strong></div>
                  <div><span>Selected</span><strong>{detail.routing.selected_model || "—"}</strong></div>
                </div>
                <div className="history-alts">
                  {detail.routing.alternatives?.map((alt) => (
                    <div key={alt.model}>
                      <span className="mono">{alt.model}</span>
                      <span>{alt.score.toFixed(1)}</span>
                      <span className="muted">{alt.reason}</span>
                    </div>
                  ))}
                </div>
                <pre className="metadata-view">{JSON.stringify(detail.routing.constraints, null, 2)}</pre>
              </div>
            )}
            {detail.routing_history && !detail.routing && (
              <pre className="metadata-view">{JSON.stringify(detail.routing_history, null, 2)}</pre>
            )}
            <div className="request-id-line">
              <span className="mono">{detail.request_id}</span>
              <CopyButton value={detail.request_id} label="Copy id" />
            </div>
          </div>
        )}
      </Modal>
    </>
  );
}
