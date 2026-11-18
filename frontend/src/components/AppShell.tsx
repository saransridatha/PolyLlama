import { useEffect, useState, type ReactNode } from "react";
import {
  Activity,
  Boxes,
  Cloud,
  ExternalLink,
  KeyRound,
  LayoutDashboard,
  LogOut,
  Menu,
  RefreshCw,
  ScrollText,
  ShieldCheck,
  Waypoints,
  X,
} from "lucide-react";
import { usePoll } from "../lib/usePoll";
import type { AppInfo } from "../lib/types";
import { duration, relativeTime } from "../lib/format";
import { Badge, Button } from "./ui";

export type PageId = "overview" | "models" | "routing" | "api" | "remote" | "logs";

const NAV: { id: PageId; label: string; icon: ReactNode; eyebrow: string }[] = [
  { id: "overview", label: "Overview", icon: <LayoutDashboard size={17} />, eyebrow: "Control room" },
  { id: "models", label: "Models", icon: <Boxes size={17} />, eyebrow: "Registry & residency" },
  { id: "routing", label: "Routing", icon: <Waypoints size={17} />, eyebrow: "Selection policy" },
  { id: "api", label: "API access", icon: <KeyRound size={17} />, eyebrow: "Clients & credentials" },
  { id: "remote", label: "Remote access", icon: <Cloud size={17} />, eyebrow: "Cloudflare Tunnel" },
  { id: "logs", label: "Logs", icon: <ScrollText size={17} />, eyebrow: "Traffic & diagnostics" },
];

const PAGE_TITLES: Record<PageId, { eyebrow: string; title: string; blurb: string }> = {
  overview: { eyebrow: "Control room", title: "System overview", blurb: "Live inference, residency and resource posture." },
  models: { eyebrow: "Registry & residency", title: "Model registry", blurb: "Register GGUF files, control residency, inspect fit." },
  routing: { eyebrow: "Selection policy", title: "Routing policy", blurb: "Task classification, candidate scoring and fallbacks." },
  api: { eyebrow: "Clients & credentials", title: "API access", blurb: "OpenAI-compatible endpoint, keys and rate limits." },
  remote: { eyebrow: "Cloudflare Tunnel", title: "Remote access", blurb: "Optional Cloudflare Tunnel for authenticated clients." },
  logs: { eyebrow: "Traffic & diagnostics", title: "Logs & requests", blurb: "Request history, routing traces and backend events." },
};

export function AppShell({
  page,
  onNavigate,
  onLogout,
  adminName,
  children,
}: {
  page: PageId;
  onNavigate: (page: PageId) => void;
  onLogout: () => void;
  adminName: string;
  children: ReactNode;
}) {
  const [mobileOpen, setMobileOpen] = useState(false);
  const readiness = usePoll<{
    ready: boolean;
    available_models: number;
    loaded_models: string[];
    llama_version: string | null;
    uptime_s: number;
  }>("/admin/api/readiness", 8000);
  const info = usePoll<AppInfo>("/admin/api/info", 60000);
  const meta = PAGE_TITLES[page];
  const docsUrl = info.data
    ? `${new URL(info.data.public_api_base).origin}/docs`
    : "http://127.0.0.1:8000/docs";

  useEffect(() => {
    setMobileOpen(false);
  }, [page]);

  return (
    <div className="shell">
      <a className="skip-link" href="#main-content">Skip to main content</a>
      <aside className={`sidebar ${mobileOpen ? "is-open" : ""}`}>
        <div className="sidebar-brand">
          <div className="brand-mark" aria-hidden="true">
            <span />
          </div>
          <div>
            <div className="brand-name">ModelSwitch</div>
            <div className="brand-sub">Inference control plane</div>
          </div>
          <button className="mobile-close" onClick={() => setMobileOpen(false)} aria-label="Close navigation">
            <X size={18} />
          </button>
        </div>

        <div className="sidebar-status">
          <div className="status-line">
            <span className={`pulse ${readiness.data?.ready ? "is-live" : "is-idle"}`} />
            <span>{readiness.data?.ready ? "Gateway ready" : "Gateway degraded"}</span>
          </div>
          <div className="sidebar-status-meta">
            {readiness.data
              ? `${readiness.data.available_models} available · ${readiness.data.loaded_models.length} resident`
              : "checking services…"}
          </div>
        </div>

        <nav className="nav" aria-label="Primary">
          <div className="nav-label">Operate</div>
          {NAV.map((item) => (
            <button
              key={item.id}
              className={`nav-item ${page === item.id ? "is-active" : ""}`}
              onClick={() => onNavigate(item.id)}
              aria-current={page === item.id ? "page" : undefined}
            >
              {item.icon}
              <span>{item.label}</span>
              {item.id === "models" && readiness.data?.loaded_models.length ? (
                <span className="nav-count">{readiness.data.loaded_models.length}</span>
              ) : null}
            </button>
          ))}
        </nav>

        <div className="sidebar-spacer" />

        <div className="sidebar-note">
          <ShieldCheck size={15} />
          <div>
            <strong>Admin surface is isolated</strong>
            <span>Cloudflare Tunnel targets the inference port only.</span>
          </div>
        </div>

        <div className="sidebar-footer">
          <div className="sidebar-user">
            <div className="avatar">{adminName.slice(0, 1).toUpperCase()}</div>
            <div>
              <div className="sidebar-user-name">{adminName}</div>
              <div className="sidebar-user-role">Administrator</div>
            </div>
            <button className="icon-btn" onClick={onLogout} title="Sign out" aria-label="Sign out">
              <LogOut size={15} />
            </button>
          </div>
          <div className="sidebar-version">
            <span>v1.0.0</span>
            <span>{readiness.data ? duration(readiness.data.uptime_s) : "—"} uptime</span>
          </div>
        </div>
      </aside>

      {mobileOpen && <div className="sidebar-scrim" onClick={() => setMobileOpen(false)} />}

      <div className="main">
        <header className="topbar">
          <div className="topbar-left">
            <button className="icon-btn mobile-menu" onClick={() => setMobileOpen(true)} aria-label="Open navigation">
              <Menu size={19} />
            </button>
            <div>
              <div className="eyebrow">{meta.eyebrow}</div>
              <h1>{meta.title}</h1>
              <p>{meta.blurb}</p>
            </div>
          </div>
          <div className="topbar-right">
            <div className="topbar-live">
              <Activity size={14} />
              <span>{readiness.data?.ready ? "Live" : "Degraded"}</span>
              <span className="topbar-refresh">{relativeTime(readiness.lastUpdated, Date.now())}</span>
            </div>
            <Button
              variant="ghost"
              size="sm"
              icon={<RefreshCw size={14} className={readiness.loading ? "spin" : ""} />}
              onClick={() => void readiness.refresh()}
            >
              Refresh
            </Button>
            <a className="btn btn-secondary btn-sm" href={docsUrl} target="_blank" rel="noreferrer">
              <ExternalLink size={14} /> API docs
            </a>
          </div>
        </header>
        <main className="page" id="main-content">
          {children}
        </main>
        <footer className="page-foot">
          <span>ModelSwitch 1.0 · llama.cpp {readiness.data?.llama_version?.replace("version: ", "") || "not detected"}</span>
          <span className="mono">admin listener · 127.0.0.1</span>
        </footer>
      </div>
    </div>
  );
}
