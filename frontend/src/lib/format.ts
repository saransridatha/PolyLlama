export function bytes(value?: number | null, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  if (value === 0) return "0 B";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let n = Math.abs(value);
  let index = 0;
  while (n >= 1024 && index < units.length - 1) {
    n /= 1024;
    index += 1;
  }
  return `${n.toFixed(index === 0 ? 0 : digits)} ${units[index]}`;
}

export function number(value?: number | null, compact = false): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return new Intl.NumberFormat("en-US", {
    notation: compact ? "compact" : "standard",
    maximumFractionDigits: compact ? 1 : 0,
  }).format(value);
}

export function decimal(value?: number | null, digits = 1, suffix = ""): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${value.toFixed(digits)}${suffix}`;
}

export function duration(seconds?: number | null): string {
  if (seconds === null || seconds === undefined) return "—";
  if (seconds < 60) return `${seconds.toFixed(0)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`;
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
}

export function ms(value?: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  if (value < 1) return `${value.toFixed(2)} ms`;
  if (value < 1000) return `${value.toFixed(0)} ms`;
  return `${(value / 1000).toFixed(2)} s`;
}

export function relativeTime(input?: string | number | null, now = Date.now()): string {
  if (!input) return "—";
  const time = typeof input === "number" ? input : Date.parse(input);
  if (Number.isNaN(time)) return "—";
  const delta = Math.round((now - time) / 1000);
  if (delta < 5) return "just now";
  if (delta < 60) return `${delta}s ago`;
  if (delta < 3600) return `${Math.floor(delta / 60)}m ago`;
  if (delta < 86400) return `${Math.floor(delta / 3600)}h ago`;
  return `${Math.floor(delta / 86400)}d ago`;
}

export function clockTime(input?: string | number | null): string {
  if (!input) return "—";
  const date = typeof input === "number" ? new Date(input) : new Date(input);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export function dateTime(input?: string | number | null): string {
  if (!input) return "—";
  const date = typeof input === "number" ? new Date(input) : new Date(input);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export function shortId(value?: string | null, length = 8): string {
  if (!value) return "—";
  return value.length > length ? value.slice(0, length) : value;
}

export function titleCase(value?: string | null): string {
  if (!value) return "—";
  return value.replace(/[_-]/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

export function pct(value?: number | null, digits = 0): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${value.toFixed(digits)}%`;
}
