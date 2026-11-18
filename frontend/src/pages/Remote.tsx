import { useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Cloud,
  CloudOff,
  ExternalLink,
  Globe2,
  KeyRound,
  RefreshCw,
  ServerCog,
  ShieldCheck,
  Square,
  Terminal,
} from "lucide-react";
import { api } from "../lib/api";
import { usePoll } from "../lib/usePoll";
import type { RemoteStatus } from "../lib/types";
import { duration, relativeTime } from "../lib/format";
import { Badge, Button, CopyButton, Field, Input, LoadingBlock, Modal, Panel, useToast } from "../components/ui";

export function RemotePage() {
  const remote = usePoll<RemoteStatus>("/admin/api/remote", 5000);
  const { push } = useToast();
  const [mode, setMode] = useState<"quick" | "token">("quick");
  const [token, setToken] = useState("");
  const [publicUrl, setPublicUrl] = useState("");
  const [tunnelName, setTunnelName] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [confirmStop, setConfirmStop] = useState(false);

  const data = remote.data;

  const start = async () => {
    setBusy("start");
    try {
      await api("/admin/api/remote/start", {
        method: "POST",
        body: JSON.stringify({
          mode,
          token: mode === "token" ? token : undefined,
          public_url: mode === "token" && publicUrl ? publicUrl : undefined,
          tunnel_name: tunnelName || undefined,
        }),
      });
      push(mode === "quick" ? "Quick tunnel started." : "Named tunnel started.", "good");
      setToken("");
      await remote.refresh();
    } catch (err) {
      push(err instanceof Error ? err.message : "Unable to start tunnel.", "bad");
    } finally {
      setBusy(null);
    }
  };

  const action = async (kind: "stop" | "restart" | "check") => {
    setBusy(kind);
    try {
      await api(`/admin/api/remote/${kind}`, { method: "POST" });
      push(kind === "stop" ? "Tunnel stopped." : kind === "restart" ? "Tunnel restarted." : "Health check completed.", "good");
      await remote.refresh();
    } catch (err) {
      push(err instanceof Error ? err.message : "Tunnel action failed.", "bad");
    } finally {
      setBusy(null);
      setConfirmStop(false);
    }
  };

  if (!data) {
    return (
      <>
        {remote.error && <div className="error-banner">{remote.error}</div>}
        <Panel title="Cloudflare Tunnel"><LoadingBlock rows={6} /></Panel>
      </>
    );
  }

  return (
    <>
      {remote.error && <div className="error-banner">{remote.error}</div>}

      <section className={`remote-hero ${data.running ? "is-running" : ""}`}>
        <div className="remote-hero-icon">
          {data.running ? <Cloud size={26} /> : <CloudOff size={26} />}
        </div>
        <div className="remote-hero-main">
          <div className="eyebrow">Remote connectivity</div>
          <h2>{data.running ? "Tunnel is live" : "Remote access is disabled"}</h2>
          <p>
            {data.running
              ? "Authenticated inference traffic can reach this machine without opening a router port."
              : "The gateway is listening on loopback only. Enable a tunnel when a remote client needs access."}
          </p>
          {data.api_base_url && (
            <div className="endpoint-copy">
              <code>{data.api_base_url}</code>
              <CopyButton value={data.api_base_url} label="Copy base URL" />
            </div>
          )}
        </div>
        <div className="remote-hero-meta">
          <Badge tone={data.installed ? "good" : "warn"} dot>{data.installed ? "cloudflared installed" : "cloudflared missing"}</Badge>
          {data.running && <Badge tone={data.health.healthy === false ? "bad" : "good"}>{data.health.healthy === false ? "health check failing" : "connected"}</Badge>}
          {data.running && <span className="muted small">uptime {duration(data.uptime_s)}</span>}
        </div>
      </section>

      <div className="remote-grid">
        <Panel
          eyebrow="Connection"
          title={data.running ? "Tunnel status" : "Start a tunnel"}
          actions={<Button variant="ghost" size="sm" icon={<RefreshCw size={14} className={remote.loading ? "spin" : ""} />} onClick={() => void remote.refresh()}>Refresh</Button>}
        >
          {data.running ? (
            <>
              <div className="status-grid">
                <div><span>Mode</span><strong>{data.mode || "—"}</strong></div>
                <div><span>Public URL</span><strong className="mono truncate">{data.public_url || "not configured"}</strong></div>
                <div><span>Local target</span><strong className="mono">{data.target}</strong></div>
                <div><span>Last check</span><strong>{data.health.checked_at ? relativeTime(data.health.checked_at) : "not checked"}</strong></div>
                <div><span>PID</span><strong className="mono">{data.pid || "—"}</strong></div>
                <div><span>Admin exposed</span><strong>{data.admin_exposed ? "YES — investigate" : "No"}</strong></div>
              </div>
              {data.health.error && <div className="form-alert is-warn"><AlertTriangle size={14} /> {data.health.error}</div>}
              <div className="toolbar-actions">
                <Button variant="secondary" icon={<RefreshCw size={14} />} loading={busy === "check"} onClick={() => void action("check")}>Run health check</Button>
                <Button variant="secondary" icon={<RefreshCw size={14} />} loading={busy === "restart"} onClick={() => void action("restart")}>Restart</Button>
                <Button variant="danger" icon={<Square size={14} />} onClick={() => setConfirmStop(true)}>Stop tunnel</Button>
              </div>
              <div className="log-panel">
                <div className="log-panel-head"><span><Terminal size={13} /> cloudflared log</span><Button variant="ghost" size="sm" onClick={() => void remote.refresh()}>Reload</Button></div>
                <pre className="log-view">{data.log_tail || "No log output yet."}</pre>
              </div>
            </>
          ) : (
            <>
              {!data.installed && (
                <div className="install-box">
                  <div className="install-head"><AlertTriangle size={16} /> cloudflared is not installed</div>
                  <p>Install the official Cloudflare connector, then refresh this page.</p>
                  <code>brew install cloudflared</code>
                  <code>winget install --id Cloudflare.cloudflared</code>
                  <a href="https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/" target="_blank" rel="noreferrer">
                    Official installation guide <ExternalLink size={12} />
                  </a>
                </div>
              )}
              <div className="segmented">
                <button type="button" className={mode === "quick" ? "is-active" : ""} onClick={() => setMode("quick")}>
                  <Globe2 size={14} /> Quick tunnel
                </button>
                <button type="button" className={mode === "token" ? "is-active" : ""} onClick={() => setMode("token")}>
                  <KeyRound size={14} /> Named tunnel
                </button>
              </div>
              {mode === "quick" ? (
                <div className="connection-mode">
                  <p className="muted">
                    Creates an ephemeral <code>*.trycloudflare.com</code> hostname. No Cloudflare account
                    or domain is required. The hostname changes every restart and is not intended for production.
                  </p>
                  <div className="inline-note">
                    <ShieldCheck size={14} /> Only the public inference listener is tunnelled. Admin routes remain on loopback.
                  </div>
                  <Button variant="primary" icon={<Cloud size={15} />} loading={busy === "start"} disabled={!data.installed} onClick={() => void start()}>
                    Start quick tunnel
                  </Button>
                </div>
              ) : (
                <div className="connection-mode">
                  <p className="muted">
                    Use a connector token from a named Cloudflare Tunnel. You need a Cloudflare account and a
                    domain on Cloudflare whose public hostname routes to <code>http://127.0.0.1:8000</code>. A
                    Cloudflare-managed hostname can be used if that hostname is in your account.
                  </p>
                  <Field label="Tunnel connector token" hint="Held in memory for this process. Prefer CLOUDFLARE_TUNNEL_TOKEN in the environment for restarts.">
                    <Input type="password" value={token} onChange={(event) => setToken(event.target.value)} placeholder="eyJhIjoi…" autoComplete="off" spellCheck={false} />
                  </Field>
                  <div className="form-grid compact">
                    <Field label="Public API URL" hint="Optional, e.g. https://llm.example.com">
                      <Input value={publicUrl} onChange={(event) => setPublicUrl(event.target.value)} placeholder="https://llm.example.com" />
                    </Field>
                    <Field label="Tunnel name" hint="Operator label only.">
                      <Input value={tunnelName} onChange={(event) => setTunnelName(event.target.value)} placeholder="local-llm" />
                    </Field>
                  </div>
                  <Button variant="primary" icon={<Cloud size={15} />} loading={busy === "start"} disabled={!data.installed} onClick={() => void start()}>
                    Start named tunnel
                  </Button>
                </div>
              )}
            </>
          )}
        </Panel>

        <div className="remote-side">
          <Panel eyebrow="Security" title="Exposure model" actions={<ShieldCheck size={16} className="muted" />}>
            <div className="security-list">
              <div><ShieldCheck size={16} /><div><strong>Admin is never tunnelled</strong><span>The connector targets <code>{data.target}</code>. The dashboard and admin API stay on a separate loopback listener.</span></div></div>
              <div><KeyRound size={16} /><div><strong>Every request is authenticated</strong><span>API keys remain mandatory through the public hostname. Rate limits apply.</span></div></div>
              <div><ServerCog size={16} /><div><strong>No inbound port forwarding</strong><span>cloudflared makes an outbound connection to Cloudflare's edge.</span></div></div>
            </div>
          </Panel>

          <Panel eyebrow="Prerequisites" title="Setup paths">
            <div className="setup-paths">
              <div className="setup-path">
                <Badge tone="good">1</Badge>
                <div><strong>Local only</strong><span>Default. API at 127.0.0.1:8000, dashboard at 127.0.0.1:8001. No account required.</span></div>
              </div>
              <div className="setup-path">
                <Badge tone="info">2</Badge>
                <div><strong>Quick tunnel</strong><span>cloudflared installed and outbound internet. Ephemeral URL, no domain.</span></div>
              </div>
              <div className="setup-path">
                <Badge tone="warn">3</Badge>
                <div><strong>Named tunnel</strong><span>Cloudflare account + domain on Cloudflare + named tunnel + public hostname → <code>127.0.0.1:8000</code>.</span></div>
              </div>
            </div>
            <div className="inline-note">
              <AlertTriangle size={14} />
              <span>Cloudflare is transport, not compute. All inference runs on this machine's CPU/GPU.</span>
            </div>
          </Panel>

          <Panel eyebrow="Recovery" title="If the tunnel drops">
            <ol className="ordered-list">
              <li>Run a health check to distinguish connector failure from origin failure.</li>
              <li>Stop, then restart the tunnel. The connector re-establishes outbound sessions.</li>
              <li>Check the cloudflared log for credentials, DNS or edge errors.</li>
              <li>Confirm the local service is healthy at <code>/healthz</code>.</li>
            </ol>
            <Button variant="secondary" size="sm" icon={<RefreshCw size={14} />} onClick={() => void action("check")} disabled={!data.running}>
              Test public endpoint
            </Button>
          </Panel>
        </div>
      </div>

      <Modal
        open={confirmStop}
        title="Stop remote access?"
        subtitle="The public URL will stop resolving immediately."
        onClose={() => setConfirmStop(false)}
        footer={
          <>
            <Button onClick={() => setConfirmStop(false)}>Keep running</Button>
            <Button variant="danger" loading={busy === "stop"} onClick={() => void action("stop")}>Stop tunnel</Button>
          </>
        }
      >
        <p className="muted">Inference remains available locally. Only the Cloudflare connection is closed.</p>
      </Modal>
    </>
  );
}
