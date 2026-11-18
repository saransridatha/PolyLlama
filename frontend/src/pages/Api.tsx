import { useEffect, useMemo, useState, type FormEvent } from "react";
import {
  AlertTriangle,
  Copy,
  KeyRound,
  Link2,
  LockKeyhole,
  RefreshCw,
  ShieldCheck,
  Trash2,
  UserRound,
} from "lucide-react";
import { api } from "../lib/api";
import { usePoll } from "../lib/usePoll";
import type { ApiKey, AppInfo, KeysResponse, ModelListResponse, RemoteStatus } from "../lib/types";
import { bytes, dateTime, duration, number, relativeTime, titleCase } from "../lib/format";
import {
  Badge,
  Button,
  CodeBlock,
  CopyButton,
  Empty,
  Field,
  Input,
  LoadingBlock,
  Modal,
  Panel,
  Select,
  Textarea,
  Toggle,
  useToast,
} from "../components/ui";

type Secret = { label: string; value: string; warning?: string } | null;

export function ApiPage() {
  const keys = usePoll<KeysResponse>("/admin/api/keys", 5000);
  const info = usePoll<AppInfo>("/admin/api/info", 60000);
  const models = usePoll<ModelListResponse>("/admin/api/models", 15000);
  const remote = usePoll<RemoteStatus>("/admin/api/remote", 15000);
  const { push } = useToast();
  const [createOpen, setCreateOpen] = useState(false);
  const [secret, setSecret] = useState<Secret>(null);
  const [revokeKey, setRevokeKey] = useState<ApiKey | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  const usage = useMemo(() => {
    const map = new Map<string, KeysResponse["usage"][number]>();
    for (const item of keys.data?.usage || []) map.set(item.id, item);
    return map;
  }, [keys.data?.usage]);

  const baseUrl = info.data?.public_api_base || "http://127.0.0.1:8000/v1";
  const publicBase = remote.data?.running ? remote.data.api_base_url : null;
  const effectiveBase = publicBase || baseUrl;
  const docsUrl = info.data ? `${new URL(info.data.public_api_base).origin}/docs` : "http://127.0.0.1:8000/docs";

  const revoke = async () => {
    if (!revokeKey) return;
    setBusy(revokeKey.id);
    try {
      await api(`/admin/api/keys/${revokeKey.id}`, { method: "DELETE" });
      push("API key revoked.", "good");
      setRevokeKey(null);
      await keys.refresh();
    } catch (err) {
      push(err instanceof Error ? err.message : "Could not revoke key.", "bad");
    } finally {
      setBusy(null);
    }
  };

  const rotate = async (key: ApiKey) => {
    setBusy(key.id);
    try {
      const response = await api<{ api_key: string; warning: string }>(`/admin/api/keys/${key.id}/rotate`, { method: "POST" });
      setSecret({ label: `Rotated ${key.name}`, value: response.api_key, warning: response.warning });
      await keys.refresh();
    } catch (err) {
      push(err instanceof Error ? err.message : "Rotation failed.", "bad");
    } finally {
      setBusy(null);
    }
  };

  const curl = `curl ${effectiveBase}/chat/completions \\
  -H "Authorization: Bearer $MODELSWITCH_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Explain Kubernetes networking"}],
    "stream": true
  }'`;

  const python = `from openai import OpenAI

client = OpenAI(
    base_url="${effectiveBase}",
    api_key="<your-inference-key>",
)

response = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Explain Kubernetes networking"}],
)
print(response.choices[0].message.content)`;

  const continueConfig = `models:
  - name: ModelSwitch Auto
    provider: openai
    model: auto
    apiBase: ${effectiveBase}
    apiKey: <your-inference-key>
    roles: [chat, edit, apply]`;

  return (
    <>
      <div className="connection-strip">
        <div className="connection-icon"><Link2 size={20} /></div>
        <div className="connection-main">
          <div className="eyebrow">OpenAI-compatible base URL</div>
          <div className="connection-url mono">{effectiveBase}</div>
          <div className="connection-meta">
            Model <code>auto</code> enables intelligent routing · <code>fast</code>, <code>coding</code>, <code>reasoning</code> aliases
            {publicBase ? " · exposed through Cloudflare Tunnel" : " · local loopback only"}
          </div>
        </div>
        <CopyButton value={effectiveBase} label="Copy URL" />
      </div>

      <Panel
        title="API keys"
        eyebrow="Credentials"
        actions={
          <>
            <Button variant="ghost" size="sm" icon={<RefreshCw size={14} className={keys.loading ? "spin" : ""} />} onClick={() => void keys.refresh()}>
              Refresh
            </Button>
            <Button variant="primary" size="sm" icon={<KeyRound size={14} />} onClick={() => setCreateOpen(true)}>
              Create key
            </Button>
          </>
        }
        flush
      >
        {!keys.data ? (
          <LoadingBlock rows={5} />
        ) : keys.data.keys.length ? (
          <div className="table-wrap">
            <table className="data-table keys-table">
              <thead>
                <tr>
                  <th>Key</th>
                  <th>Role</th>
                  <th>Rate limits</th>
                  <th>Usage · 24h</th>
                  <th>Last used</th>
                  <th>Status</th>
                  <th aria-label="Actions" />
                </tr>
              </thead>
              <tbody>
                {keys.data.keys.map((key) => {
                  const stat = usage.get(key.id);
                  return (
                    <tr key={key.id} className={key.revoked ? "row-disabled" : ""}>
                      <td>
                        <div className="key-name">
                          <span>{key.name}</span>
                          <code className="key-prefix">{key.prefix}…</code>
                        </div>
                        <div className="cell-sub mono">{key.id.slice(0, 12)}</div>
                      </td>
                      <td>
                        <Badge tone={key.role === "admin" ? "accent" : "info"}>{key.role}</Badge>
                        <div className="cell-sub">
                          {Array.isArray(key.permissions.models)
                            ? `${key.permissions.models.length} model scope${key.permissions.models.length === 1 ? "" : "s"}`
                            : "all models"}
                        </div>
                      </td>
                      <td>
                        <div className="mono">{number(key.rate_limit_rpm)} rpm</div>
                        <div className="cell-sub">{number(key.rate_limit_tpm, true)} tpm</div>
                      </td>
                      <td>
                        <div className="mono">{number(stat?.requests_24h)} requests</div>
                        <div className="cell-sub">{number(stat?.tokens_24h, true)} tokens</div>
                      </td>
                      <td className="muted">{relativeTime(key.last_used_at)}</td>
                      <td>
                        {key.revoked ? <Badge tone="bad">revoked</Badge> : key.expires_at && Date.parse(key.expires_at) < Date.now() ? <Badge tone="warn">expired</Badge> : <Badge tone="good" dot>active</Badge>}
                        {key.expires_at && <div className="cell-sub">expires {dateTime(key.expires_at)}</div>}
                      </td>
                      <td>
                        <div className="row-actions">
                          <button className="icon-btn" title="Rotate key" aria-label={`Rotate ${key.name}`} disabled={key.revoked || busy === key.id} onClick={() => void rotate(key)}>
                            <RefreshCw size={14} className={busy === key.id ? "spin" : ""} />
                          </button>
                          <button className="icon-btn danger" title="Revoke key" aria-label={`Revoke ${key.name}`} disabled={key.revoked} onClick={() => setRevokeKey(key)}>
                            <Trash2 size={14} />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty title="No API keys" body="Create an inference key for your applications." />
        )}
        <div className="panel-inline-foot keys-foot">
          <span><ShieldCheck size={14} /> Keys are stored only as salted scrypt hashes. Plaintext is shown once.</span>
          <span>Defaults: {keys.data?.limits.default_rpm} rpm · {number(keys.data?.limits.default_tpm, true)} tpm</span>
        </div>
      </Panel>

      <div className="split-grid api-split">
        <Panel eyebrow="Quick start" title="Connect a client">
          <div className="code-stack">
            <CodeBlock code={curl} language="curl" />
            <CodeBlock code={python} language="python · openai sdk" />
          </div>
        </Panel>
        <Panel eyebrow="Coding agents" title="Continue / custom OpenAI clients">
          <p className="muted small">
            Point any OpenAI-compatible application at the base URL, use an inference key, and
            choose <code>auto</code> to let the router select the model.
          </p>
          <CodeBlock code={continueConfig} language="yaml · continue" />
          <dl className="kv compact-kv">
            <div><dt>Base URL</dt><dd className="mono">{effectiveBase}</dd></div>
            <div><dt>Model</dt><dd className="mono">auto</dd></div>
            <div><dt>Streaming</dt><dd>Server-Sent Events</dd></div>
            <div><dt>API</dt><dd>OpenAI chat, completions, models</dd></div>
          </dl>
          {info.data && (
            <div className="inline-note">
              <Link2 size={14} />
              <span>
                Interactive OpenAPI docs: <a href={docsUrl} target="_blank" rel="noreferrer">/docs</a> on the public listener.
                Admin routes are not available there.
              </span>
            </div>
          )}
        </Panel>
      </div>

      <div className="split-grid api-split">
        <Panel eyebrow="Auth model" title="What a key can do">
          <div className="security-list">
            <div><LockKeyhole size={16} /><div><strong>Inference keys</strong><span>Chat/completions, streaming, model listing. Model scopes and max tokens are optional.</span></div></div>
            <div><ShieldCheck size={16} /><div><strong>Admin keys</strong><span>Everything inference keys can do, plus model, routing, key and tunnel management on the loopback listener.</span></div></div>
            <div><UserRound size={16} /><div><strong>Rate limits</strong><span>Sliding-window requests/minute and tokens/minute enforced per key.</span></div></div>
          </div>
        </Panel>
        <Panel eyebrow="Remote" title="Public endpoint">
          {!remote.data ? <LoadingBlock rows={3} /> : (
            <>
              <div className="endpoint-line">
                <span className={`pulse ${remote.data.running ? "is-live" : "is-idle"}`} />
                <strong>{remote.data.running ? "Tunnel connected" : "Local only"}</strong>
                {remote.data.mode && <Badge tone="info">{remote.data.mode}</Badge>}
              </div>
              <p className="muted small">
                {remote.data.running
                  ? "The public URL proxies only the inference listener. Use the URL above with an inference key."
                  : "Remote access is disabled. Enable a Cloudflare Tunnel in Remote access when you need it."}
              </p>
              {remote.data.api_base_url && (
                <div className="endpoint-copy">
                  <code>{remote.data.api_base_url}</code>
                  <CopyButton value={remote.data.api_base_url} />
                </div>
              )}
              <div className="inline-note">
                <AlertTriangle size={14} />
                <span>Inference always executes on this machine. Cloudflare only carries the encrypted connection.</span>
              </div>
            </>
          )}
        </Panel>
      </div>

      <CreateKeyModal
        open={createOpen}
        models={models.data?.models.filter((model) => model.enabled) || []}
        defaults={keys.data?.limits}
        onClose={() => setCreateOpen(false)}
        onCreated={async (secretValue) => {
          setCreateOpen(false);
          setSecret(secretValue);
          await keys.refresh();
        }}
      />

      <Modal
        open={Boolean(secret)}
        title={secret?.label || "New API key"}
        subtitle="Copy this credential now. It cannot be recovered."
        onClose={() => setSecret(null)}
        footer={<Button variant="primary" onClick={() => setSecret(null)}>I have stored it</Button>}
      >
        {secret && (
          <>
            <div className="secret-box">
              <code>{secret.value}</code>
              <CopyButton value={secret.value} label="Copy key" />
            </div>
            <div className="form-alert is-warn">
              <AlertTriangle size={15} /> {secret.warning || "Store this in a secret manager before closing."}
            </div>
          </>
        )}
      </Modal>

      <Modal
        open={Boolean(revokeKey)}
        title="Revoke API key?"
        subtitle="Existing clients using this key will receive 401 immediately."
        onClose={() => setRevokeKey(null)}
        footer={
          <>
            <Button onClick={() => setRevokeKey(null)}>Cancel</Button>
            <Button variant="danger" loading={busy === revokeKey?.id} onClick={() => void revoke()}>Revoke key</Button>
          </>
        }
      >
        <p className="muted"><strong>{revokeKey?.name}</strong> ({revokeKey?.prefix}…) will stop working immediately. This cannot be undone.</p>
      </Modal>
    </>
  );
}

function CreateKeyModal({
  open,
  models,
  defaults,
  onClose,
  onCreated,
}: {
  open: boolean;
  models: ModelListResponse["models"];
  defaults?: KeysResponse["limits"];
  onClose: () => void;
  onCreated: (secret: { label: string; value: string; warning: string }) => Promise<void>;
}) {
  const [form, setForm] = useState({
    name: "",
    role: "inference" as "inference" | "admin",
    rpm: defaults?.default_rpm || 120,
    tpm: defaults?.default_tpm || 200000,
    allowStreaming: true,
    allowEmbeddings: true,
    maxTokens: "",
    modelScope: "all" as "all" | "some",
    selectedModels: [] as string[],
    expires: "",
  });
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    setError(null);
    setForm((current) => ({
      ...current,
      rpm: defaults?.default_rpm || 120,
      tpm: defaults?.default_tpm || 200000,
    }));
  }, [open, defaults?.default_rpm, defaults?.default_tpm]);

  const save = async () => {
    if (!form.name.trim()) {
      setError("Give the key a name.");
      return;
    }
    setSaving(true);
    setError(null);
    const permissions: Record<string, unknown> = {
      allow_streaming: form.allowStreaming,
      allow_embeddings: form.allowEmbeddings,
    };
    if (form.modelScope === "some") {
      if (!form.selectedModels.length) {
        setError("Select at least one model, or grant all models.");
        setSaving(false);
        return;
      }
      permissions.models = form.selectedModels;
    } else {
      permissions.models = ["*"];
    }
    if (form.maxTokens) permissions.max_tokens = Number(form.maxTokens);
    try {
      const response = await api<{ api_key: string; warning: string; key: ApiKey }>("/admin/api/keys", {
        method: "POST",
        body: JSON.stringify({
          name: form.name.trim(),
          role: form.role,
          permissions,
          rate_limit_rpm: Number(form.rpm),
          rate_limit_tpm: Number(form.tpm),
          expires_at: form.expires ? new Date(form.expires).toISOString() : null,
        }),
      });
      await onCreated({ label: `New ${form.role} key · ${form.name}`, value: response.api_key, warning: response.warning });
      setForm({ ...form, name: "" });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not create key.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <Modal
      open={open}
      title="Create API key"
      subtitle="The plaintext key is shown exactly once."
      onClose={onClose}
      wide
      footer={
        <>
          <Button onClick={onClose}>Cancel</Button>
          <Button variant="primary" loading={saving} onClick={() => void save()}>Create key</Button>
        </>
      }
    >
      <form onSubmit={(event) => { event.preventDefault(); void save(); }} className="model-form">
        <div className="form-grid">
          <Field label="Name"><Input value={form.name} onChange={(event) => setForm({ ...form, name: event.target.value })} placeholder="Continue (desktop)" autoFocus /></Field>
          <Field label="Role" hint="Admin keys can manage the gateway. Use sparingly.">
            <Select
              value={form.role}
              onChange={(event) => {
                const role = event.target.value as "inference" | "admin";
                setForm({
                  ...form,
                  role,
                  rpm: role === "admin" ? (defaults?.admin_rpm || 600) : (defaults?.default_rpm || 120),
                  tpm: role === "admin" ? 1_000_000 : (defaults?.default_tpm || 200_000),
                });
              }}
            >
              <option value="inference">Inference only</option>
              <option value="admin">Administrator</option>
            </Select>
          </Field>
          <Field label="Requests / minute"><Input type="number" min={1} max={100000} value={form.rpm} onChange={(event) => setForm({ ...form, rpm: Number(event.target.value) })} /></Field>
          <Field label="Tokens / minute"><Input type="number" min={100} max={100000000} value={form.tpm} onChange={(event) => setForm({ ...form, tpm: Number(event.target.value) })} /></Field>
          <Field label="Max output tokens" hint="Optional hard cap per request."><Input type="number" min={1} value={form.maxTokens} onChange={(event) => setForm({ ...form, maxTokens: event.target.value })} placeholder="No cap" /></Field>
          <Field label="Expires" hint="Optional. Local time."><Input type="datetime-local" value={form.expires} onChange={(event) => setForm({ ...form, expires: event.target.value })} /></Field>
        </div>

        <div className="toggle-stack">
          <Toggle checked={form.allowStreaming} onChange={(value) => setForm({ ...form, allowStreaming: value })} label="Allow streaming" />
          <Toggle checked={form.allowEmbeddings} onChange={(value) => setForm({ ...form, allowEmbeddings: value })} label="Allow embeddings" />
        </div>

        <Field label="Model scope">
          <div className="segmented">
            <button type="button" className={form.modelScope === "all" ? "is-active" : ""} onClick={() => setForm({ ...form, modelScope: "all" })}>All registered models</button>
            <button type="button" className={form.modelScope === "some" ? "is-active" : ""} onClick={() => setForm({ ...form, modelScope: "some" })}>Selected models</button>
          </div>
        </Field>
        {form.modelScope === "some" && (
          <div className="model-scope-list">
            {models.length ? models.map((model) => (
              <label key={model.id} className={`scope-item ${form.selectedModels.includes(model.id) ? "is-selected" : ""}`}>
                <input
                  type="checkbox"
                  checked={form.selectedModels.includes(model.id)}
                  onChange={(event) =>
                    setForm({
                      ...form,
                      selectedModels: event.target.checked
                        ? [...form.selectedModels, model.id]
                        : form.selectedModels.filter((id) => id !== model.id),
                    })
                  }
                />
                <span>{model.name}</span>
                <span className="mono">{model.id}</span>
              </label>
            )) : <span className="muted small">No models are registered yet. The key will have an empty scope.</span>}
          </div>
        )}
        {error && <div className="form-alert">{error}</div>}
      </form>
    </Modal>
  );
}
