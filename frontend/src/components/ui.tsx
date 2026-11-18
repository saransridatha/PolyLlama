import {
  createContext,
  useContext,
  useEffect,
  useId,
  useRef,
  useState,
  type ButtonHTMLAttributes,
  type InputHTMLAttributes,
  type ReactNode,
  type SelectHTMLAttributes,
  type TextareaHTMLAttributes,
} from "react";
import { AlertTriangle, Check, Copy, Loader2, X } from "lucide-react";

type Tone = "neutral" | "good" | "warn" | "bad" | "info" | "accent";

export function Button({
  children,
  variant = "secondary",
  size = "md",
  loading = false,
  icon,
  className = "",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "ghost" | "danger";
  size?: "sm" | "md";
  loading?: boolean;
  icon?: ReactNode;
}) {
  return (
    <button
      className={`btn btn-${variant} btn-${size} ${className}`}
      disabled={props.disabled || loading}
      {...props}
    >
      {loading ? <Loader2 size={14} className="spin" /> : icon}
      {children}
    </button>
  );
}

export function Badge({ children, tone = "neutral", dot = false }: { children: ReactNode; tone?: Tone; dot?: boolean }) {
  return (
    <span className={`badge badge-${tone}`}>
      {dot && <span className={`badge-dot dot-${tone}`} />}
      {children}
    </span>
  );
}

export function Panel({
  title,
  eyebrow,
  actions,
  children,
  className = "",
  flush = false,
}: {
  title?: ReactNode;
  eyebrow?: string;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  flush?: boolean;
}) {
  return (
    <section className={`panel ${className}`}>
      {(title || actions) && (
        <header className="panel-head">
          <div>
            {eyebrow && <div className="eyebrow">{eyebrow}</div>}
            {title && <h2 className="panel-title">{title}</h2>}
          </div>
          {actions && <div className="panel-actions">{actions}</div>}
        </header>
      )}
      <div className={flush ? "panel-body panel-body-flush" : "panel-body"}>{children}</div>
    </section>
  );
}

export function Stat({
  label,
  value,
  detail,
  tone,
  children,
}: {
  label: string;
  value: ReactNode;
  detail?: ReactNode;
  tone?: Tone;
  children?: ReactNode;
}) {
  return (
    <div className={`stat ${tone ? `stat-${tone}` : ""}`}>
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {detail && <div className="stat-detail">{detail}</div>}
      {children}
    </div>
  );
}

export function Field({
  label,
  hint,
  error,
  children,
  htmlFor,
}: {
  label: string;
  hint?: string;
  error?: string | null;
  children: ReactNode;
  htmlFor?: string;
}) {
  return (
    <label className="field" htmlFor={htmlFor}>
      <span className="field-label">{label}</span>
      {children}
      {hint && !error && <span className="field-hint">{hint}</span>}
      {error && <span className="field-error">{error}</span>}
    </label>
  );
}

export function Input(props: InputHTMLAttributes<HTMLInputElement>) {
  return <input className={`input ${props.className || ""}`} {...props} />;
}

export function Textarea(props: TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return <textarea className={`input textarea ${props.className || ""}`} {...props} />;
}

export function Select(props: SelectHTMLAttributes<HTMLSelectElement>) {
  return <select className={`input select ${props.className || ""}`} {...props} />;
}

export function Toggle({
  checked,
  onChange,
  label,
  hint,
  disabled = false,
}: {
  checked: boolean;
  onChange: (value: boolean) => void;
  label: string;
  hint?: string;
  disabled?: boolean;
}) {
  const id = useId();
  return (
    <div className={`toggle-row ${disabled ? "is-disabled" : ""}`}>
      <div>
        <label htmlFor={id} className="toggle-label">
          {label}
        </label>
        {hint && <div className="toggle-hint">{hint}</div>}
      </div>
      <button
        id={id}
        type="button"
        role="switch"
        aria-checked={checked}
        aria-label={label}
        className={`toggle ${checked ? "is-on" : ""}`}
        onClick={() => onChange(!checked)}
        disabled={disabled}
      >
        <span className="toggle-knob" />
      </button>
    </div>
  );
}

export function Modal({
  open,
  title,
  subtitle,
  children,
  footer,
  onClose,
  wide = false,
}: {
  open: boolean;
  title: string;
  subtitle?: string;
  children: ReactNode;
  footer?: ReactNode;
  onClose: () => void;
  wide?: boolean;
}) {
  const dialogRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const previous = document.activeElement as HTMLElement | null;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
        return;
      }
      if (event.key !== "Tab" || !dialogRef.current) return;
      const focusable = Array.from(
        dialogRef.current.querySelectorAll<HTMLElement>(
          'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
        ),
      ).filter((element) => element.offsetParent !== null);
      if (!focusable.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", onKey);
    document.body.classList.add("modal-open");
    const focusTimer = window.setTimeout(() => {
      const first = dialogRef.current?.querySelector<HTMLElement>(
        'input:not([disabled]), select:not([disabled]), textarea:not([disabled]), button:not([disabled])',
      );
      (first || dialogRef.current)?.focus();
    }, 60);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.classList.remove("modal-open");
      window.clearTimeout(focusTimer);
      previous?.focus?.();
    };
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div ref={dialogRef} className={`modal ${wide ? "modal-wide" : ""}`} role="dialog" aria-modal="true" aria-label={title} tabIndex={-1}>
        <header className="modal-head">
          <div>
            <h2>{title}</h2>
            {subtitle && <p>{subtitle}</p>}
          </div>
          <button className="icon-btn" onClick={onClose} aria-label="Close dialog">
            <X size={17} />
          </button>
        </header>
        <div className="modal-body">{children}</div>
        {footer && <footer className="modal-foot">{footer}</footer>}
      </div>
    </div>
  );
}

type Toast = { id: number; message: string; tone: "good" | "bad" | "info" };
const ToastContext = createContext<{ push: (message: string, tone?: Toast["tone"]) => void } | null>(null);

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const push = (message: string, tone: Toast["tone"] = "info") => {
    const id = Date.now() + Math.random();
    setToasts((current) => [...current, { id, message, tone }]);
    window.setTimeout(() => setToasts((current) => current.filter((t) => t.id !== id)), 5200);
  };
  return (
    <ToastContext.Provider value={{ push }}>
      {children}
      <div className="toast-stack" role="status" aria-live="polite">
        {toasts.map((toast) => (
          <div key={toast.id} className={`toast toast-${toast.tone}`}>
            {toast.tone === "bad" ? <AlertTriangle size={16} /> : <Check size={16} />}
            <span>{toast.message}</span>
            <button onClick={() => setToasts((current) => current.filter((t) => t.id !== toast.id))} aria-label="Dismiss">
              <X size={14} />
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast() {
  const context = useContext(ToastContext);
  if (!context) throw new Error("useToast must be used within ToastProvider");
  return context;
}

export function CopyButton({ value, label = "Copy" }: { value: string; label?: string }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(value);
    } catch {
      const area = document.createElement("textarea");
      area.value = value;
      document.body.appendChild(area);
      area.select();
      document.execCommand("copy");
      area.remove();
    }
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  };
  return (
    <button type="button" className="copy-btn" onClick={copy} aria-label={`${label}: ${value}`}>
      {copied ? <Check size={13} /> : <Copy size={13} />}
      {copied ? "Copied" : label}
    </button>
  );
}

export function CodeBlock({ code, language }: { code: string; language?: string }) {
  return (
    <div className="code-block">
      <div className="code-block-head">
        <span>{language || "shell"}</span>
        <CopyButton value={code} />
      </div>
      <pre>
        <code>{code}</code>
      </pre>
    </div>
  );
}

export function Progress({ value, tone = "accent" }: { value: number; tone?: "accent" | "warn" | "bad" }) {
  const bounded = Math.max(0, Math.min(100, Number.isFinite(value) ? value : 0));
  return (
    <div className="progress" role="progressbar" aria-valuenow={Math.round(bounded)} aria-valuemin={0} aria-valuemax={100}>
      <span className={`progress-fill fill-${tone}`} style={{ width: `${bounded}%` }} />
    </div>
  );
}

export function Sparkline({
  values,
  height = 44,
  label,
}: {
  values: number[];
  height?: number;
  label: string;
}) {
  const width = 180;
  const max = Math.max(1, ...values);
  const step = values.length > 1 ? width / (values.length - 1) : width;
  const points = values.map((value, index) => `${index * step},${height - (value / max) * (height - 6) - 3}`);
  const area = `0,${height} ${points.join(" ")} ${width},${height}`;
  return (
    <svg className="sparkline" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={label} preserveAspectRatio="none">
      {values.length > 1 ? (
        <>
          <polygon points={area} className="spark-area" />
          <polyline points={points.join(" ")} className="spark-line" />
        </>
      ) : (
        <line x1="0" y1={height - 4} x2={width} y2={height - 4} className="spark-flat" />
      )}
    </svg>
  );
}

export function Empty({ title, body, action }: { title: string; body?: ReactNode; action?: ReactNode }) {
  return (
    <div className="empty">
      <div className="empty-mark" aria-hidden="true" />
      <h3>{title}</h3>
      {body && <p>{body}</p>}
      {action}
    </div>
  );
}

export function LoadingBlock({ rows = 3 }: { rows?: number }) {
  return (
    <div className="loading-block" aria-label="Loading">
      {Array.from({ length: rows }).map((_, index) => (
        <div key={index} className="skeleton" style={{ width: `${88 - index * 12}%` }} />
      ))}
    </div>
  );
}

export function Small({ children }: { children: ReactNode }) {
  return <span className="muted small">{children}</span>;
}
