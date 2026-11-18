export type Json = Record<string, unknown>;

export class ApiError extends Error {
  status: number;
  code?: string;
  constructor(message: string, status: number, code?: string) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

let adminKey = "";
let onUnauthorized: (() => void) | null = null;

export function setAdminKey(key: string) {
  adminKey = key;
  if (key) sessionStorage.setItem("modelswitch.adminKey", key);
  else sessionStorage.removeItem("modelswitch.adminKey");
}

export function getStoredAdminKey(): string {
  return sessionStorage.getItem("modelswitch.adminKey") || "";
}

export function setUnauthorizedHandler(handler: () => void) {
  onUnauthorized = handler;
}

export async function api<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers = new Headers(options.headers);
  headers.set("Authorization", `Bearer ${adminKey}`);
  if (options.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
  const response = await fetch(path, { ...options, headers, credentials: "omit" });
  const text = await response.text();
  let payload: unknown = null;
  try {
    payload = text ? JSON.parse(text) : null;
  } catch {
    payload = null;
  }
  if (!response.ok) {
    if (response.status === 401 || response.status === 403) {
      if (response.status === 401) onUnauthorized?.();
    }
    const error = (payload as { error?: { message?: string; code?: string } } | null)?.error;
    throw new ApiError(error?.message || `Request failed with HTTP ${response.status}`, response.status, error?.code);
  }
  return payload as T;
}

export function qs(params: Record<string, string | number | boolean | undefined>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== "") search.set(key, String(value));
  }
  const encoded = search.toString();
  return encoded ? `?${encoded}` : "";
}
