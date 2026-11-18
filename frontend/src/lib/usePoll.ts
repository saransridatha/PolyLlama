import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";

export interface PollState<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
  refresh: () => Promise<void>;
  lastUpdated: number | null;
}

export function usePoll<T>(path: string, intervalMs = 5000, enabled = true): PollState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState<number | null>(null);
  const mounted = useRef(true);
  const inFlight = useRef(false);

  const refresh = useCallback(async () => {
    if (!enabled || inFlight.current) return;
    inFlight.current = true;
    try {
      const result = await api<T>(path);
      if (mounted.current) {
        setData(result);
        setError(null);
        setLastUpdated(Date.now());
      }
    } catch (err) {
      if (mounted.current) setError(err instanceof Error ? err.message : String(err));
    } finally {
      inFlight.current = false;
      if (mounted.current) setLoading(false);
    }
  }, [path, enabled]);

  useEffect(() => {
    mounted.current = true;
    if (!enabled) {
      setLoading(false);
      return;
    }
    setLoading(true);
    void refresh();
    const timer = window.setInterval(() => {
      if (document.visibilityState === "visible") void refresh();
    }, intervalMs);
    return () => {
      mounted.current = false;
      window.clearInterval(timer);
    };
  }, [refresh, intervalMs, enabled]);

  return { data, error, loading, refresh, lastUpdated };
}

export function useNow(intervalMs = 1000): number {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), intervalMs);
    return () => window.clearInterval(id);
  }, [intervalMs]);
  return now;
}
