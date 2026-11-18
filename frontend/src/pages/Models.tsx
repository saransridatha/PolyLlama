import { useEffect, useMemo, useState, type FormEvent } from "react";
import {
  AlertTriangle,
  Boxes,
  FileSearch,
  FolderSearch,
  HardDrive,
  Pencil,
  Play,
  RefreshCw,
  ScrollText,
  Square,
  Trash2,
  WandSparkles,
} from "lucide-react";
import { api } from "../lib/api";
import { usePoll } from "../lib/usePoll";
import type { DiscoveredModel, Model, ModelListResponse, SystemMetrics } from "../lib/types";
import { bytes, dateTime, decimal, ms, number, titleCase } from "../lib/format";
import {
  Badge,
  Button,
  Empty,
  Field,
  Input,
  LoadingBlock,
  Modal,
  Panel,
  Progress,
  Select,
  Textarea,
  Toggle,
  useToast,
} from "../components/ui";

const STATE_TONE: Record<string, "good" | "warn" | "bad" | "neutral" | "info"> = {
  loaded: "good",
  loading: "warn",
  unloading: "warn",
  crashed: "bad",
  failed: "bad",
  unloaded: "neutral",
};

function capLabel(capability: string): string {
  const labels: Record<string, string> = {
    chat: "chat",
    general: "general",
    reasoning: "reasoning",
    coding: "coding",
    code_generation: "codegen",
    debugging: "debug",
    math: "math",
    summarization: "summarize",
    extraction: "extract",
    translation: "translate",
    creative: "creative",
    json: "JSON",
    long_context: "long ctx",
    classification: "classify",
    vision: "vision",
    embeddings: "embed",
    instruct: "instruct",
    fast: "fast",
    tool_use: "tools",
  };
  return labels[capability] || capability;
}

export function ModelsPage() {
  const models = usePoll<ModelListResponse>("/admin/api/models", 4000);
  const system = usePoll<SystemMetrics>("/admin/api/system", 6000);
  const { push } = useToast();
  const [busy, setBusy] = useState<string | null>(null);
  const [addOpen, setAddOpen] = useState(false);
  const [editModel, setEditModel] = useState<Model | null>(null);
  const [deleteModel, setDeleteModel] = useState<Model | null>(null);
  const [logModel, setLogModel] = useState<Model | null>(null);
  const [logText, setLogText] = useState("");
  const [logLoading, setLogLoading] = useState(false);

  const data = models.data;
  const residentRam = useMemo(
    () =>
      (data?.models || [])
        .filter((model) => model.runtime.loaded)
        .reduce((total, model) => total + Number(model.runtime.estimated_memory?.ram_bytes || 0), 0),
    [data?.models],
  );
  const availableRam = system.data?.resources.available_ram_bytes || 0;
  const residentCount = (data?.models || []).filter((model) => model.runtime.loaded).length;
  const residentLimit = data?.limits.max_resident_models || 1;

  const runAction = async (modelId: string, action: "load" | "unload" | "reload") => {
    setBusy(`${modelId}:${action}`);
    try {
      await api(`/admin/api/models/${modelId}/${action}`, { method: "POST" });
      push(
        action === "load" ? "Model loaded." : action === "unload" ? "Model unloaded." : "Model reloaded.",
        "good",
      );
      await Promise.all([models.refresh(), system.refresh()]);
    } catch (err) {
      push(err instanceof Error ? err.message : "Model action failed.", "bad");
    } finally {
      setBusy(null);
    }
  };

  const openLogs = async (model: Model) => {
    setLogModel(model);
    setLogText("");
    setLogLoading(true);
    try {
      const response = await api<{ log: string }>(`/admin/api/models/${model.id}/log?lines=300`);
      setLogText(response.log || "No backend log output yet. Load the model to create a log.");
    } catch (err) {
      setLogText(err instanceof Error ? err.message : "Unable to load log.");
    } finally {
      setLogLoading(false);
    }
  };

  const removeModel = async () => {
    if (!deleteModel) return;
    setBusy("delete");
    try {
      await api(`/admin/api/models/${deleteModel.id}`, { method: "DELETE" });
      push("Registration removed. The GGUF file was not deleted.", "good");
      setDeleteModel(null);
      await models.refresh();
    } catch (err) {
      push(err instanceof Error ? err.message : "Unable to remove model.", "bad");
    } finally {
      setBusy(null);
    }
  };

  return (
    <>
      <div className="page-toolbar">
        <div className="memory-summary">
          <div className="memory-summary-top">
            <span><HardDrive size={14} /> Resident memory</span>
            <strong>{bytes(residentRam)} <span className="muted">of {bytes(availableRam)} usable</span></strong>
          </div>
          <Progress
            value={availableRam ? (residentRam / availableRam) * 100 : 0}
            tone={residentRam > availableRam * 0.9 ? "bad" : residentRam > availableRam * 0.7 ? "warn" : "accent"}
          />
          <div className="memory-summary-meta">
            {residentCount} / {residentLimit} resident slots used · estimate includes KV cache, activations and 7.5% runtime overhead
          </div>
        </div>
        <div className="toolbar-actions">
          <Button
            variant="secondary"
            icon={<FolderSearch size={15} />}
            onClick={() => setAddOpen(true)}
          >
            Scan & import
          </Button>
          <Button variant="primary" icon={<Boxes size={15} />} onClick={() => setAddOpen(true)}>
            Register GGUF
          </Button>
        </div>
      </div>

      {models.error && <div className="error-banner">{models.error}</div>}

      <Panel
        title="Registered models"
        eyebrow="GGUF registry"
        actions={
          <Button variant="ghost" size="sm" icon={<RefreshCw size={14} className={models.loading ? "spin" : ""} />} onClick={() => void models.refresh()}>
            Refresh
          </Button>
        }
        flush
      >
        {!data ? (
          <LoadingBlock rows={6} />
        ) : data.models.length === 0 ? (
          <Empty
            title="No models registered"
            body="Register a local GGUF file to make it available to the router. ModelSwitch never downloads model weights automatically."
            action={
              <Button variant="primary" onClick={() => setAddOpen(true)} icon={<FolderSearch size={15} />}>
                Scan model directories
              </Button>
            }
          />
        ) : (
          <div className="table-wrap">
            <table className="data-table models-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>State</th>
                  <th>Fit</th>
                  <th>Context</th>
                  <th>Capabilities</th>
                  <th className="num">Load / TTFT</th>
                  <th aria-label="Actions" />
                </tr>
              </thead>
              <tbody>
                {data.models.map((model) => {
                  const estimate = Number(model.runtime.estimated_memory?.ram_bytes || model.estimate?.ram_bytes || 0);
                  const fit = availableRam ? (estimate / availableRam) * 100 : 0;
                  const fitLabel = fit > 100 ? "over budget" : fit > 82 ? "tight" : "fits";
                  const fitTone = fit > 100 ? "bad" : fit > 82 ? "warn" : "good";
                  const actionBusy = busy?.startsWith(`${model.id}:`);
                  return (
                    <tr key={model.id} className={!model.enabled ? "row-disabled" : ""}>
                      <td>
                        <div className="model-title">
                          <span className="model-name">{model.name}</span>
                          {!model.enabled && <Badge tone="neutral">disabled</Badge>}
                          {model.restart_required && <Badge tone="warn">reload required</Badge>}
                          {model.runtime.crash_count ? <Badge tone="bad">{model.runtime.crash_count} crash</Badge> : null}
                        </div>
                        <div className="model-sub">
                          <span className="mono">{model.id}</span>
                          <span>{model.parameter_size} · {model.quantization} · {bytes(model.size_bytes)}</span>
                        </div>
                        {model.availability && !model.availability.available && (
                          <div className="model-availability">
                            <AlertTriangle size={12} /> {model.availability.reason}
                          </div>
                        )}
                        {model.description && <div className="model-description">{model.description}</div>}
                      </td>
                      <td>
                        <Badge tone={STATE_TONE[model.runtime.state] || "neutral"} dot>
                          {titleCase(model.runtime.state)}
                        </Badge>
                        {model.runtime.queue && (
                          <div className="cell-sub mono">
                            {model.runtime.queue.in_use}/{model.runtime.queue.limit} busy
                            {model.runtime.queue.waiting ? ` · ${model.runtime.queue.waiting} queued` : ""}
                          </div>
                        )}
                        {model.runtime.error && <div className="cell-error" title={model.runtime.error}>{model.runtime.error}</div>}
                      </td>
                      <td>
                        <div className="fit-cell">
                          <div className="fit-row">
                            <Badge tone={fitTone}>{fitLabel}</Badge>
                            <span className="mono">{bytes(estimate)}</span>
                          </div>
                          <div className="fit-bar"><span style={{ width: `${Math.min(100, fit)}%` }} /></div>
                          <div className="cell-sub">{decimal(fit, 0)}% of usable RAM</div>
                        </div>
                      </td>
                      <td>
                        <div className="mono">{number(model.context_length)}</div>
                        <div className="cell-sub">slots {model.max_concurrency} · {model.runtime.gpu_layers ?? 0} GPU layers</div>
                      </td>
                      <td>
                        <div className="chips">
                          {model.capabilities.slice(0, 4).map((capability) => (
                            <span className="chip" key={capability}>{capLabel(capability)}</span>
                          ))}
                          {model.capabilities.length > 4 && <span className="chip chip-more">+{model.capabilities.length - 4}</span>}
                        </div>
                        <div className="cell-sub">
                          priority {model.priority} · {model.residency === "keep_loaded" ? "keep loaded" : "on demand"}
                        </div>
                      </td>
                      <td className="num">
                        <div className="mono">{model.performance?.avg_ttft_ms ? ms(model.performance.avg_ttft_ms) : "—"}</div>
                        <div className="cell-sub">{model.performance?.avg_tps ? `${decimal(model.performance.avg_tps, 1)} tok/s` : "no samples"}</div>
                      </td>
                      <td>
                        <div className="row-actions">
                          {model.runtime.loaded ? (
                            <>
                              <button
                                className="icon-btn"
                                title="Unload model"
                                aria-label={`Unload ${model.name}`}
                                disabled={actionBusy}
                                onClick={() => void runAction(model.id, "unload")}
                              >
                                <Square size={14} />
                              </button>
                              <button
                                className="icon-btn"
                                title="Reload model"
                                aria-label={`Reload ${model.name}`}
                                disabled={actionBusy}
                                onClick={() => void runAction(model.id, "reload")}
                              >
                                <RefreshCw size={14} className={busy === `${model.id}:reload` ? "spin" : ""} />
                              </button>
                            </>
                          ) : (
                            <button
                              className="icon-btn"
                              title="Load model"
                              aria-label={`Load ${model.name}`}
                              disabled={actionBusy || !model.enabled}
                              onClick={() => void runAction(model.id, "load")}
                            >
                              <Play size={14} className={busy === `${model.id}:load` ? "spin" : ""} />
                            </button>
                          )}
                          <button className="icon-btn" title="Edit configuration" aria-label={`Edit ${model.name}`} onClick={() => setEditModel(model)}>
                            <Pencil size={14} />
                          </button>
                          <button className="icon-btn" title="Backend log" aria-label={`Log for ${model.name}`} onClick={() => void openLogs(model)}>
                            <ScrollText size={14} />
                          </button>
                          <button className="icon-btn danger" title="Remove registration" aria-label={`Remove ${model.name}`} onClick={() => setDeleteModel(model)}>
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
        )}
      </Panel>

      <div className="split-grid">
        <Panel eyebrow="Discovery" title="Model roots" actions={<FolderSearch size={15} className="muted" />}>
          <p className="muted small">
            Files inside these roots can be registered without enabling arbitrary filesystem access.
          </p>
          <div className="path-list">
            {data?.limits.model_roots.map((root) => <code key={root}>{root}</code>)}
          </div>
          <div className="inline-note">
            <WandSparkles size={14} />
            <span>
              External paths are {data?.limits.allow_external_model_paths ? "allowed" : "blocked"} by policy.
              Configure <code>MS_MODEL_ROOTS</code> to add another trusted directory.
            </span>
          </div>
        </Panel>
        <Panel eyebrow="Backend" title="Inference runtime">
          <dl className="kv">
            <div><dt>llama.cpp</dt><dd className="mono truncate">{data?.limits.llama_binary || "not found"}</dd></div>
            <div><dt>Resident limit</dt><dd>{residentLimit} models</dd></div>
            <div><dt>Default context</dt><dd>{number(data?.limits.default_context_length)} tokens</dd></div>
            <div><dt>Platform</dt><dd>{system.data ? `${system.data.hardware.os_name} ${system.data.hardware.arch}` : "—"}</dd></div>
            <div><dt>GPU backend</dt><dd className="mono">{system.data?.hardware.llama_backend || system.data?.hardware.backend || "—"}</dd></div>
          </dl>
          {system.data?.hardware.warnings.map((warning) => (
            <div className="hardware-warning" key={warning}><AlertTriangle size={14} /> {warning}</div>
          ))}
        </Panel>
      </div>

      <AddModelModal
        open={addOpen}
        capabilities={data?.capabilities || []}
        defaultContext={data?.limits.default_context_length || 8192}
        defaultConcurrency={2}
        onClose={() => setAddOpen(false)}
        onSaved={async () => {
          setAddOpen(false);
          await models.refresh();
        }}
      />

      <EditModelModal
        model={editModel}
        capabilities={data?.capabilities || []}
        onClose={() => setEditModel(null)}
        onSaved={async () => {
          setEditModel(null);
          await models.refresh();
        }}
      />

      <Modal
        open={Boolean(deleteModel)}
        title="Remove model registration?"
        subtitle="This does not delete the GGUF file."
        onClose={() => setDeleteModel(null)}
        footer={
          <>
            <Button onClick={() => setDeleteModel(null)}>Cancel</Button>
            <Button variant="danger" loading={busy === "delete"} onClick={() => void removeModel()}>Remove registration</Button>
          </>
        }
      >
        <p className="muted">
          <strong>{deleteModel?.name}</strong> will be removed from the registry, routing metadata and
          dashboard. The file <code>{deleteModel?.path}</code> stays on disk.
        </p>
      </Modal>

      <Modal
        open={Boolean(logModel)}
        title={`${logModel?.name || "Model"} · backend log`}
        subtitle="Last 300 lines from llama.cpp, served only on the admin listener."
        onClose={() => setLogModel(null)}
        wide
      >
        {logLoading ? <LoadingBlock rows={8} /> : <pre className="log-view">{logText || "No log output."}</pre>}
      </Modal>
    </>
  );
}

function AddModelModal({
  open,
  capabilities,
  defaultContext,
  defaultConcurrency,
  onClose,
  onSaved,
}: {
  open: boolean;
  capabilities: string[];
  defaultContext: number;
  defaultConcurrency: number;
  onClose: () => void;
  onSaved: () => Promise<void>;
}) {
  const { push } = useToast();
  const [tab, setTab] = useState<"scan" | "path">("scan");
  const [discovered, setDiscovered] = useState<DiscoveredModel[]>([]);
  const [scanLoading, setScanLoading] = useState(false);
  const [selected, setSelected] = useState<DiscoveredModel | null>(null);
  const [form, setForm] = useState({
    path: "",
    name: "",
    context_length: defaultContext,
    max_concurrency: defaultConcurrency,
    priority: 50,
    residency: "on_demand" as "on_demand" | "keep_loaded",
    aliases: "",
    description: "",
    enabled: true,
    capabilities: [] as string[],
  });
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    setError(null);
    setTab("scan");
    setScanLoading(true);
    api<{ models: DiscoveredModel[] }>("/admin/api/models/discover")
      .then((response) => setDiscovered(response.models))
      .catch((err) => setError(err instanceof Error ? err.message : "Scan failed."))
      .finally(() => setScanLoading(false));
  }, [open]);

  useEffect(() => {
    if (!open) return;
    setForm((current) => ({
      ...current,
      context_length: defaultContext,
      max_concurrency: defaultConcurrency,
    }));
  }, [open, defaultContext, defaultConcurrency]);

  const choose = (model: DiscoveredModel) => {
    setSelected(model);
    setForm((current) => ({
      ...current,
      path: model.path,
      name: model.name,
      context_length: Math.min(model.context_length || defaultContext, defaultContext),
      capabilities: model.capabilities || [],
    }));
  };

  const toggleCapability = (capability: string) => {
    setForm((current) => ({
      ...current,
      capabilities: current.capabilities.includes(capability)
        ? current.capabilities.filter((item) => item !== capability)
        : [...current.capabilities, capability],
    }));
  };

  const save = async () => {
    setError(null);
    if (!form.path.trim()) {
      setError("Choose a discovered model or enter a GGUF path.");
      return;
    }
    setSaving(true);
    try {
      await api("/admin/api/models", {
        method: "POST",
        body: JSON.stringify({
          path: form.path.trim(),
          name: form.name.trim() || undefined,
          context_length: Number(form.context_length),
          max_concurrency: Number(form.max_concurrency),
          priority: Number(form.priority),
          residency: form.residency,
          aliases: form.aliases.split(",").map((value) => value.trim()).filter(Boolean),
          description: form.description,
          enabled: form.enabled,
          capabilities: form.capabilities.length ? form.capabilities : undefined,
        }),
      });
      push("Model registered. It will load on first use.", "good");
      setSelected(null);
      setForm((current) => ({ ...current, path: "", name: "", aliases: "", description: "" }));
      await onSaved();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Registration failed.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <Modal
      open={open}
      title="Register a GGUF model"
      subtitle="Model metadata is read from the file header. Nothing is copied or moved."
      onClose={onClose}
      wide
      footer={
        <>
          <Button onClick={onClose}>Cancel</Button>
          <Button variant="primary" loading={saving} onClick={() => void save()}>
            Register model
          </Button>
        </>
      }
    >
      <form onSubmit={(event) => { event.preventDefault(); void save(); }} className="model-form">
        <div className="segmented" role="tablist">
          <button type="button" className={tab === "scan" ? "is-active" : ""} onClick={() => setTab("scan")} role="tab">
            <FileSearch size={14} /> Discovered files
          </button>
          <button type="button" className={tab === "path" ? "is-active" : ""} onClick={() => setTab("path")} role="tab">
            <HardDrive size={14} /> Local path
          </button>
        </div>

        {tab === "scan" ? (
          <div className="discover-list">
            {scanLoading ? (
              <LoadingBlock rows={4} />
            ) : discovered.length === 0 ? (
              <Empty title="No unregistered GGUF files found" body="Place a .gguf file in a configured model root, then rescan." />
            ) : (
              discovered.map((item) => (
                <button
                  type="button"
                  key={item.path}
                  className={`discover-item ${selected?.path === item.path ? "is-selected" : ""}`}
                  onClick={() => item.valid && choose(item)}
                  disabled={!item.valid}
                >
                  <div className="discover-main">
                    <div className="discover-name">{item.name}</div>
                    <div className="discover-path mono">{item.path}</div>
                    {!item.valid && <div className="cell-error">{item.error}</div>}
                  </div>
                  <div className="discover-meta">
                    {item.valid ? (
                      <>
                        <span>{item.parameter_size}</span>
                        <span>{item.quantization}</span>
                        <span>{bytes(item.size_bytes)}</span>
                        <span>ctx {number(item.context_length)}</span>
                      </>
                    ) : <Badge tone="bad">invalid</Badge>}
                  </div>
                </button>
              ))
            )}
          </div>
        ) : (
          <Field label="Absolute GGUF path" hint="Must resolve inside a configured model root unless external paths are allowed.">
            <Input
              value={form.path}
              onChange={(event) => setForm({ ...form, path: event.target.value })}
              placeholder="/home/you/models/model-Q4_K_M.gguf"
              className="mono"
            />
          </Field>
        )}

        <div className="form-grid">
          <Field label="Display name">
            <Input value={form.name} onChange={(event) => setForm({ ...form, name: event.target.value })} placeholder="Derived from GGUF metadata" />
          </Field>
          <Field label="Aliases" hint="Comma-separated. `auto`, `fast`, `coding` and `reasoning` are reserved.">
            <Input value={form.aliases} onChange={(event) => setForm({ ...form, aliases: event.target.value })} placeholder="my-coder, workhorse" />
          </Field>
          <Field label="Context length" hint={`Per-request context. Up to ${number(defaultContext)} by default.`}>
            <Input type="number" min={256} max={1048576} step={256} value={form.context_length} onChange={(event) => setForm({ ...form, context_length: Number(event.target.value) })} />
          </Field>
          <Field label="Concurrent slots" hint="llama.cpp parallel slots. Each slot consumes KV cache.">
            <Input type="number" min={1} max={16} value={form.max_concurrency} onChange={(event) => setForm({ ...form, max_concurrency: Number(event.target.value) })} />
          </Field>
          <Field label="Routing priority" hint="0–100 operator tie-breaker.">
            <Input type="number" min={0} max={100} value={form.priority} onChange={(event) => setForm({ ...form, priority: Number(event.target.value) })} />
          </Field>
          <Field label="Residency">
            <Select value={form.residency} onChange={(event) => setForm({ ...form, residency: event.target.value as "on_demand" | "keep_loaded" })}>
              <option value="on_demand">Load on demand · unload when idle</option>
              <option value="keep_loaded">Prefer kept loaded</option>
            </Select>
          </Field>
        </div>

        <Field label="Capability overrides" hint="Pre-filled from discovery. Leave none selected to let the backend infer from GGUF metadata.">
          <div className="cap-grid">
            {capabilities.map((capability) => (
              <button
                type="button"
                key={capability}
                className={`cap-toggle ${form.capabilities.includes(capability) ? "is-on" : ""}`}
                onClick={() => toggleCapability(capability)}
                aria-pressed={form.capabilities.includes(capability)}
              >
                {capLabel(capability)}
              </button>
            ))}
          </div>
        </Field>
        <Field label="Description">
          <Textarea rows={2} value={form.description} onChange={(event) => setForm({ ...form, description: event.target.value })} placeholder="Optional notes for operators." />
        </Field>
        <Toggle checked={form.enabled} onChange={(enabled) => setForm({ ...form, enabled })} label="Enable for routing" hint="Disabled models remain registered but are never selected." />
        {error && <div className="form-alert">{error}</div>}
      </form>
    </Modal>
  );
}

function EditModelModal({
  model,
  capabilities,
  onClose,
  onSaved,
}: {
  model: Model | null;
  capabilities: string[];
  onClose: () => void;
  onSaved: () => Promise<void>;
}) {
  const { push } = useToast();
  const [form, setForm] = useState({
    name: "",
    context_length: 8192,
    max_concurrency: 2,
    priority: 50,
    residency: "on_demand" as "on_demand" | "keep_loaded",
    aliases: "",
    description: "",
    enabled: true,
    capabilities: [] as string[],
    n_gpu_layers: "",
    threads: "",
  });
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!model) return;
    setForm({
      name: model.name,
      context_length: model.context_length,
      max_concurrency: model.max_concurrency,
      priority: model.priority,
      residency: model.residency,
      aliases: model.aliases.join(", "),
      description: model.description || "",
      enabled: model.enabled,
      capabilities: model.capabilities,
      n_gpu_layers: model.n_gpu_layers === null || model.n_gpu_layers === undefined ? "" : String(model.n_gpu_layers),
      threads: model.threads === null || model.threads === undefined ? "" : String(model.threads),
    });
    setError(null);
  }, [model]);

  if (!model) return null;

  const save = async () => {
    setSaving(true);
    setError(null);
    try {
      await api(`/admin/api/models/${model.id}`, {
        method: "PATCH",
        body: JSON.stringify({
          name: form.name,
          context_length: Number(form.context_length),
          max_concurrency: Number(form.max_concurrency),
          priority: Number(form.priority),
          residency: form.residency,
          aliases: form.aliases.split(",").map((value) => value.trim()).filter(Boolean),
          description: form.description,
          enabled: form.enabled,
          capabilities: form.capabilities,
          n_gpu_layers: form.n_gpu_layers === "" ? null : Number(form.n_gpu_layers),
          threads: form.threads === "" ? null : Number(form.threads),
        }),
      });
      push("Model configuration updated.", "good");
      await onSaved();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Update failed.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <Modal
      open={Boolean(model)}
      title={`Edit ${model.name}`}
      subtitle={model.path}
      onClose={onClose}
      wide
      footer={
        <>
          <Button onClick={onClose}>Cancel</Button>
          <Button variant="primary" loading={saving} onClick={() => void save()}>Save configuration</Button>
        </>
      }
    >
      {model.restart_required && (
        <div className="form-alert is-warn">
          This model is loaded with an older configuration. Reload it after saving to apply context, slots or GPU changes.
        </div>
      )}
      <div className="form-grid">
        <Field label="Display name"><Input value={form.name} onChange={(event) => setForm({ ...form, name: event.target.value })} /></Field>
        <Field label="Aliases" hint="Comma-separated."><Input value={form.aliases} onChange={(event) => setForm({ ...form, aliases: event.target.value })} /></Field>
        <Field label="Context length"><Input type="number" min={256} value={form.context_length} onChange={(event) => setForm({ ...form, context_length: Number(event.target.value) })} /></Field>
        <Field label="Concurrent slots"><Input type="number" min={1} max={16} value={form.max_concurrency} onChange={(event) => setForm({ ...form, max_concurrency: Number(event.target.value) })} /></Field>
        <Field label="Priority"><Input type="number" min={0} max={100} value={form.priority} onChange={(event) => setForm({ ...form, priority: Number(event.target.value) })} /></Field>
        <Field label="Residency">
          <Select value={form.residency} onChange={(event) => setForm({ ...form, residency: event.target.value as "on_demand" | "keep_loaded" })}>
            <option value="on_demand">Load on demand</option>
            <option value="keep_loaded">Prefer kept loaded</option>
          </Select>
        </Field>
        <Field label="GPU layers" hint="Empty = automatic hardware plan.">
          <Input type="number" min={-1} max={999} value={form.n_gpu_layers} onChange={(event) => setForm({ ...form, n_gpu_layers: event.target.value })} placeholder="auto" />
        </Field>
        <Field label="CPU threads" hint="Empty = physical core count.">
          <Input type="number" min={1} max={256} value={form.threads} onChange={(event) => setForm({ ...form, threads: event.target.value })} placeholder="auto" />
        </Field>
      </div>
      <Field label="Capabilities">
        <div className="cap-grid">
          {capabilities.map((capability) => (
            <button
              type="button"
              key={capability}
              className={`cap-toggle ${form.capabilities.includes(capability) ? "is-on" : ""}`}
              onClick={() =>
                setForm({
                  ...form,
                  capabilities: form.capabilities.includes(capability)
                    ? form.capabilities.filter((item) => item !== capability)
                    : [...form.capabilities, capability],
                })
              }
            >
              {capLabel(capability)}
            </button>
          ))}
        </div>
      </Field>
      <Field label="Description"><Textarea rows={2} value={form.description} onChange={(event) => setForm({ ...form, description: event.target.value })} /></Field>
      <Toggle checked={form.enabled} onChange={(enabled) => setForm({ ...form, enabled })} label="Enabled" hint="Disabled models stay registered but cannot be routed." />
      <div className="modal-meta-line">Registered {dateTime(model.created_at)} · max context {number(model.max_context_length)}</div>
      {error && <div className="form-alert">{error}</div>}
    </Modal>
  );
}
