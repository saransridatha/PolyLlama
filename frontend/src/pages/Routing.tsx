import { useEffect, useMemo, useState } from "react";
import { CheckCircle2, ChevronDown, GitBranch, RefreshCw, Save, SlidersHorizontal, Sparkles, Waypoints } from "lucide-react";
import { api } from "../lib/api";
import { usePoll } from "../lib/usePoll";
import type { RoutingResponse, RouterConfig } from "../lib/types";
import { decimal, ms, relativeTime, titleCase } from "../lib/format";
import { Badge, Button, Empty, Field, Input, LoadingBlock, Panel, Select, Toggle, useToast } from "../components/ui";

const TASKS: { key: string; capability: string; label: string }[] = [
  { key: "general", capability: "chat", label: "Conversation" },
  { key: "reasoning", capability: "reasoning", label: "Reasoning" },
  { key: "coding", capability: "coding", label: "Coding" },
  { key: "code_generation", capability: "code_generation", label: "Codegen" },
  { key: "debugging", capability: "debugging", label: "Debug" },
  { key: "math", capability: "math", label: "Math" },
  { key: "summarization", capability: "summarization", label: "Summary" },
  { key: "extraction", capability: "extraction", label: "Extract" },
  { key: "translation", capability: "translation", label: "Translate" },
  { key: "creative_writing", capability: "creative", label: "Creative" },
  { key: "structured_json", capability: "json", label: "JSON" },
  { key: "long_context", capability: "long_context", label: "Long ctx" },
  { key: "classification", capability: "classification", label: "Classify" },
];

const STRATEGY_COPY: Record<string, string> = {
  balanced: "Equal weight on task fit, quality, latency and memory. Best default.",
  quality: "Favours larger and more capable models when they fit.",
  latency: "Favours warm models and measured time-to-first-token.",
  local: "Favours small resident models and avoids cold loads.",
};

export function RoutingPage() {
  const routing = usePoll<RoutingResponse>("/admin/api/routing", 5000);
  const { push } = useToast();
  const [draft, setDraft] = useState<RouterConfig | null>(null);
  const [saving, setSaving] = useState(false);
  const [expanded, setExpanded] = useState<number | null>(null);

  useEffect(() => {
    if (routing.data) setDraft(routing.data.config);
  }, [routing.data?.config]);

  const dirty = useMemo(() => {
    if (!draft || !routing.data) return false;
    return JSON.stringify(draft) !== JSON.stringify(routing.data.config);
  }, [draft, routing.data]);

  const save = async () => {
    if (!draft) return;
    setSaving(true);
    try {
      await api("/admin/api/routing/config", { method: "PATCH", body: JSON.stringify(draft) });
      push("Routing policy updated.", "good");
      await routing.refresh();
    } catch (err) {
      push(err instanceof Error ? err.message : "Unable to update routing policy.", "bad");
    } finally {
      setSaving(false);
    }
  };

  if (!routing.data || !draft) {
    return (
      <>
        {routing.error && <div className="error-banner">{routing.error}</div>}
        <Panel title="Routing policy"><LoadingBlock rows={7} /></Panel>
      </>
    );
  }

  const data = routing.data;

  return (
    <>
      {routing.error && <div className="error-banner">{routing.error}</div>}

      <div className="page-toolbar">
        <div className="inline-note">
          <Waypoints size={15} />
          <span>
            Every decision is explainable. The router scores only models that fit the request
            context and the live memory budget.
          </span>
        </div>
        <div className="toolbar-actions">
          <Button
            variant="secondary"
            icon={<RefreshCw size={15} className={routing.loading ? "spin" : ""} />}
            onClick={() => void routing.refresh()}
          >
            Refresh
          </Button>
          <Button variant="primary" icon={<Save size={15} />} loading={saving} disabled={!dirty} onClick={() => void save()}>
            {dirty ? "Save policy" : "Saved"}
          </Button>
        </div>
      </div>

      <div className="routing-grid">
        <Panel eyebrow="Policy" title="Selection strategy" actions={<SlidersHorizontal size={16} className="muted" />}>
          <div className="strategy-grid">
            {Object.entries(STRATEGY_COPY).map(([strategy, copy]) => (
              <button
                key={strategy}
                className={`strategy-card ${draft.strategy === strategy ? "is-active" : ""}`}
                onClick={() => setDraft({ ...draft, strategy: strategy as RouterConfig["strategy"] })}
                type="button"
              >
                <div className="strategy-head">
                  <span>{titleCase(strategy)}</span>
                  {draft.strategy === strategy && <CheckCircle2 size={16} />}
                </div>
                <p>{copy}</p>
              </button>
            ))}
          </div>
          <div className="form-grid compact">
            <Field label="Ambiguity threshold" hint="Below this confidence, the optional classifier is consulted.">
              <Input
                type="number"
                min={0}
                max={1}
                step={0.05}
                value={draft.ambiguity_threshold}
                onChange={(event) => setDraft({ ...draft, ambiguity_threshold: Number(event.target.value) })}
              />
            </Field>
            <Field label="Fallback candidates" hint="Try up to this many models when loading fails.">
              <Input
                type="number"
                min={1}
                max={10}
                value={draft.max_alternatives}
                onChange={(event) => setDraft({ ...draft, max_alternatives: Number(event.target.value) })}
              />
            </Field>
          </div>
          <div className="toggle-stack">
            <Toggle
              checked={draft.prioritize_loaded}
              onChange={(value) => setDraft({ ...draft, prioritize_loaded: value })}
              label="Prefer resident models"
              hint="Avoid cold-load latency when quality is comparable."
            />
            <Toggle
              checked={draft.fallback_enabled}
              onChange={(value) => setDraft({ ...draft, fallback_enabled: value })}
              label="Automatic fallback"
              hint="Retry a compatible next-best model if loading or inference fails."
            />
            <Toggle
              checked={draft.allow_general_fallback}
              onChange={(value) => setDraft({ ...draft, allow_general_fallback: value })}
              label="Allow general fallback for specialist aliases"
              hint="If no coding/reasoning specialist exists, use the best general model."
            />
          </div>
        </Panel>

        <Panel eyebrow="Classifier" title="Lightweight decision layer" actions={<Sparkles size={16} className="muted" />}>
          <p className="muted small">
            The heuristic classifier is always active and takes under a millisecond. An optional
            small local model resolves only ambiguous requests. It is never loaded just to route.
          </p>
          <div className="toggle-stack">
            <Toggle
              checked={draft.llm_enabled}
              onChange={(value) => setDraft({ ...draft, llm_enabled: value })}
              label="Enable LLM arbitration"
              hint="Only consulted when heuristic confidence is low and a model is already resident."
            />
          </div>
          <div className="form-grid compact">
            <Field label="Decision model" hint="Empty = smallest currently resident model.">
              <Select
                value={draft.llm_model}
                onChange={(event) => setDraft({ ...draft, llm_model: event.target.value })}
                disabled={!draft.llm_enabled}
              >
                <option value="">Any resident model</option>
                {data.models.map((model) => (
                  <option key={model.id} value={model.id}>{model.name}</option>
                ))}
              </Select>
            </Field>
            <Field label="Timeout (ms)">
              <Input
                type="number"
                min={200}
                max={10000}
                step={100}
                value={draft.llm_timeout_ms}
                disabled={!draft.llm_enabled}
                onChange={(event) => setDraft({ ...draft, llm_timeout_ms: Number(event.target.value) })}
              />
            </Field>
          </div>
          <div className="inline-note">
            <GitBranch size={14} />
            <span>
              A failed or timed-out decision model falls back to the deterministic classifier. Routing
              never blocks on loading the decision model.
            </span>
          </div>
          <div className="classifier-pill">
            <span className={`pulse ${draft.llm_enabled ? "is-live" : "is-idle"}`} />
            {draft.llm_enabled ? "Hybrid classifier active" : "Deterministic classifier only"}
          </div>
        </Panel>
      </div>

      <Panel eyebrow="Aliases" title="Model role compatibility layer" actions={<Badge tone="accent">{Object.keys(data.aliases.model_aliases).length} model aliases</Badge>}>
        <div className="alias-grid">
          {Object.entries(data.aliases.role_aliases).map(([alias, info]) => (
            <div className="alias-card" key={alias}>
              <div className="alias-name mono">{alias}</div>
              <p>{info.description}</p>
              <Badge tone={info.kind === "router" ? "accent" : "info"}>{info.kind}</Badge>
            </div>
          ))}
          {Object.entries(data.aliases.model_aliases).map(([alias, info]) => (
            <div className="alias-card" key={alias}>
              <div className="alias-name mono">{alias}</div>
              <p>{info.description}</p>
              <Badge tone="neutral">model</Badge>
            </div>
          ))}
        </div>
      </Panel>

      <Panel eyebrow="Capabilities" title="Task suitability matrix" flush>
        <div className="table-wrap">
          <table className="data-table capability-table">
            <thead>
              <tr>
                <th>Model</th>
                {TASKS.map((task) => <th key={task.key} className="rotated"><span>{task.label}</span></th>)}
              </tr>
            </thead>
            <tbody>
              {data.models.map((model) => (
                <tr key={model.id}>
                  <td>
                    <div className="model-title">
                      <span className="model-name">{model.name}</span>
                      {model.loaded && <Badge tone="good" dot>resident</Badge>}
                    </div>
                    <div className="model-sub">{model.parameter_size} · {model.quantization} · prio {model.priority}</div>
                  </td>
                  {TASKS.map((task) => {
                    const match = model.capabilities.includes(task.capability);
                    const codingAlias = ["coding", "code_generation", "debugging"].includes(task.key)
                      && model.capabilities.some((c) => ["coding", "code_generation", "debugging"].includes(c));
                    const reasoningAlias = ["reasoning", "math"].includes(task.key) && model.capabilities.includes("reasoning");
                    const active = match || codingAlias || reasoningAlias;
                    return (
                      <td key={task.key} className="cap-cell">
                        <span className={`cap-dot ${active ? "is-on" : ""}`} title={active ? "capability metadata" : "no explicit capability"} />
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      <div className="split-grid">
        <Panel eyebrow="History" title="Recent routing decisions" flush>
          {data.history.length ? (
            <div className="routing-history">
              {data.history.slice(0, 30).map((row) => (
                <div className={`history-row ${expanded === row.id ? "is-open" : ""}`} key={row.id}>
                  <button className="history-summary" onClick={() => setExpanded(expanded === row.id ? null : row.id)}>
                    <span className="history-time mono">{relativeTime(row.created_at)}</span>
                    <span className="history-task">{titleCase(row.task)}</span>
                    <span className="history-model">{row.selected_model || "no model"}</span>
                    <span className="mono">{decimal(row.confidence * 100, 0)}%</span>
                    <span className="mono">{ms(row.routing_ms)}</span>
                    {row.fallback ? <Badge tone="warn">fallback</Badge> : <Badge tone="good">direct</Badge>}
                    <ChevronDown size={14} className="history-chev" />
                  </button>
                  {expanded === row.id && (
                    <div className="history-detail">
                      <p>{row.explanation}</p>
                      <div className="history-alts">
                        {row.alternatives?.length ? row.alternatives.map((alt) => (
                          <div key={alt.model}>
                            <span className="mono">{alt.model}</span>
                            <span>{decimal(alt.score, 1)}</span>
                            <span className="muted">{alt.reason}</span>
                          </div>
                        )) : <span className="muted">No alternative candidates recorded.</span>}
                      </div>
                      <div className="history-request mono">request {row.request_id}</div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <Empty title="No routing decisions yet" body={'Send a request with model "auto" to populate routing history.'} />
          )}
        </Panel>

        <Panel eyebrow="Measured" title="Model performance" flush>
          {data.models.length ? (
            <div className="table-wrap">
              <table className="data-table">
                <thead><tr><th>Model</th><th className="num">TTFT</th><th className="num">tok/s</th><th className="num">samples</th></tr></thead>
                <tbody>
                  {data.models.map((model) => (
                    <tr key={model.id}>
                      <td>
                        <div className="model-cell">{model.name}</div>
                        <div className="cell-sub">{model.loaded ? "resident" : "cold"}</div>
                      </td>
                      <td className="num mono">{ms(model.performance?.avg_ttft_ms)}</td>
                      <td className="num mono">{decimal(model.performance?.avg_tps, 1)}</td>
                      <td className="num mono">{model.performance?.samples ?? 0}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : <Empty title="No models" body="Register a model to gather performance data." />}
          <div className="panel-inline-foot muted small">
            EWMA from completed requests; updates after every inference.
          </div>
        </Panel>
      </div>
    </>
  );
}
