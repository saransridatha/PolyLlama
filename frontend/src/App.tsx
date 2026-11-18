import { useEffect, useState } from "react";
import { api, getStoredAdminKey, setAdminKey, setUnauthorizedHandler } from "./lib/api";
import { Login } from "./components/Login";
import { AppShell, type PageId } from "./components/AppShell";
import { ToastProvider } from "./components/ui";
import { OverviewPage } from "./pages/Overview";
import { ModelsPage } from "./pages/Models";
import { RoutingPage } from "./pages/Routing";
import { ApiPage } from "./pages/Api";
import { RemotePage } from "./pages/Remote";
import { LogsPage } from "./pages/Logs";

const PAGES: PageId[] = ["overview", "models", "routing", "api", "remote", "logs"];

function pageFromHash(): PageId {
  const value = window.location.hash.replace(/^#\/?/, "");
  return (PAGES.includes(value as PageId) ? value : "overview") as PageId;
}

export default function App() {
  const [key, setKeyState] = useState("");
  const [checking, setChecking] = useState(true);
  const [authenticated, setAuthenticated] = useState(false);
  const [page, setPage] = useState<PageId>(pageFromHash);
  const [adminName, setAdminName] = useState("Administrator");

  const logout = () => {
    setAdminKey("");
    setKeyState("");
    setAuthenticated(false);
    setChecking(false);
  };

  useEffect(() => {
    setUnauthorizedHandler(() => {
      setAdminKey("");
      setKeyState("");
      setAuthenticated(false);
    });
  }, []);

  useEffect(() => {
    const stored = getStoredAdminKey();
    if (!stored) {
      setChecking(false);
      return;
    }
    setAdminKey(stored);
    api<{ key: { name: string } }>("/admin/api/auth/me")
      .then((response) => {
        setKeyState(stored);
        setAdminName(response.key.name);
        setAuthenticated(true);
      })
      .catch(() => {
        setAdminKey("");
      })
      .finally(() => setChecking(false));
  }, []);

  useEffect(() => {
    const onHash = () => setPage(pageFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const navigate = (next: PageId) => {
    window.location.hash = `/${next}`;
    setPage(next);
  };

  if (checking) {
    return (
      <div className="boot-screen">
        <div className="boot-mark">
          <span />
        </div>
        <div className="boot-text">Establishing control session…</div>
      </div>
    );
  }

  if (!authenticated || !key) {
    return (
      <ToastProvider>
        <Login
          onAuthenticated={() => {
            const stored = getStoredAdminKey();
            setKeyState(stored);
            setAuthenticated(true);
            api<{ key: { name: string } }>("/admin/api/auth/me")
              .then((response) => setAdminName(response.key.name))
              .catch(() => undefined);
          }}
        />
      </ToastProvider>
    );
  }

  return (
    <ToastProvider>
      <AppShell page={page} onNavigate={navigate} onLogout={logout} adminName={adminName}>
        {page === "overview" && <OverviewPage onNavigate={navigate} />}
        {page === "models" && <ModelsPage />}
        {page === "routing" && <RoutingPage />}
        {page === "api" && <ApiPage />}
        {page === "remote" && <RemotePage />}
        {page === "logs" && <LogsPage />}
      </AppShell>
    </ToastProvider>
  );
}
