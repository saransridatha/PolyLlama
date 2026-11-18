export interface RuntimeInfo {
  state: "unloaded" | "loading" | "loaded" | "unloading" | "failed" | "crashed" | string;
  loaded: boolean;
  active_requests: number;
  queued_requests: number;
  loaded_at?: number | null;
  last_used?: number | null;
  context_length?: number;
  context_total?: number;
  slots?: number;
  gpu_layers?: number;
  threads?: number;
  estimated_memory?: MemoryEstimate;
  load_count?: number;
  crash_count?: number;
  error?: string | null;
  queue?: { limit: number; in_use: number; waiting: number; acquired_total: number; timed_out_total: number };
  backend?: { pid?: number; rss_bytes?: number; cpu_percent?: number; threads?: number; port?: number };
  log_path?: string;
}

export interface MemoryEstimate {
  weights_bytes?: number;
  kv_cache_bytes?: number;
  runtime_overhead_bytes?: number;
  ram_bytes?: number;
  vram_bytes?: number;
  file_size?: number;
  estimated_load_ms?: number;
}

export interface Model {
  id: string;
  name: string;
  path?: string;
  aliases: string[];
  capabilities: string[];
  context_length: number;
  max_context_length?: number;
  quantization: string;
  parameter_size: string;
  parameter_count?: number;
  size_bytes?: number;
  priority: number;
  enabled: boolean;
  residency: "on_demand" | "keep_loaded";
  max_concurrency: number;
  n_gpu_layers?: number | null;
  threads?: number | null;
  description?: string;
  extra_args?: string[];
  created_at?: string;
  updated_at?: string;
  availability?: { available: boolean; exists: boolean; reason?: string; changed?: boolean };
  runtime: RuntimeInfo;
  estimate?: MemoryEstimate;
  performance?: Performance | null;
  restart_required?: boolean;
}

export interface Performance {
  model_id: string;
  samples: number;
  avg_ttft_ms: number | null;
  avg_tps: number | null;
  avg_total_ms: number | null;
  success_count: number;
  failure_count: number;
  updated_at: string;
}

export interface ModelListResponse {
  models: Model[];
  capabilities: string[];
  limits: {
    max_resident_models: number;
    max_context_length: number;
    default_context_length: number;
    model_roots: string[];
    allow_external_model_paths: boolean;
    llama_binary: string | null;
  };
}

export interface RequestRow {
  request_id: string;
  api_key_id?: string;
  endpoint: string;
  requested_model: string;
  resolved_model?: string | null;
  task_category?: string | null;
  status: string;
  http_status?: number | null;
  stream: number | boolean;
  prompt_tokens: number;
  completion_tokens: number;
  queue_ms: number;
  load_ms?: number;
  ttft_ms?: number | null;
  total_ms?: number | null;
  error?: string | null;
  routing?: RoutingInfo | null;
  created_at: string;
  completed_at?: string | null;
  routing_history?: RoutingHistory | null;
}

export interface RoutingInfo {
  task: string;
  task_label: string;
  confidence: number;
  capability: string;
  strategy: string;
  selected_model: string | null;
  explanation: string;
  alternatives: { model: string; score: number; loaded: boolean; reason: string }[];
  constraints: Record<string, unknown>;
  routing_ms: number;
  classifier_source: string;
  fallback_used: boolean;
  resolved_alias?: string | null;
  classification: Record<string, unknown>;
}

export interface RoutingHistory {
  id: number;
  request_id: string;
  task: string;
  confidence: number;
  selected_model: string | null;
  explanation: string;
  alternatives: { model: string; score: number; loaded: boolean; reason: string }[];
  constraints: Record<string, unknown>;
  fallback: boolean;
  routing_ms: number;
  created_at: string;
}

export interface Overview {
  app: { name: string; version: string; uptime_s: number; started_at: number };
  health: { status: string; llama_binary: boolean; llama_version: string | null; hardware_backend: string };
  stats: {
    total_requests: number;
    requests_last_minute: number;
    avg_latency_ms: number | null;
    tokens_last_5m: number;
    requests_last_24h: number;
    prompt_tokens_24h: number;
    completion_tokens_24h: number;
    errors_24h: number;
    tokens_per_second_5m: number;
    active_requests: number;
    queued_requests: number;
    models_registered: number;
    models_loaded: number;
    models_enabled: number;
    max_resident_models: number;
  };
  resources: {
    total_ram_bytes: number;
    available_ram_bytes: number;
    used_ram_bytes: number;
    ram_percent: number;
    swap_used_bytes: number;
    disk_free_bytes: number;
    gpu_free_bytes: number;
    gpu_total_bytes: number;
    gpu_used_bytes: number;
    gpu_utilization: number | null;
    cpu_percent: number;
    load_1m: number;
  };
  hardware: {
    cpu_model: string;
    physical_cores: number;
    logical_cores: number;
    total_ram_bytes: number;
    gpus: { name: string; vendor: string; total_mb: number; free_mb: number; utilization: number | null; backend: string; cuda: boolean }[];
    backend: string;
    warnings: string[];
    os: string;
    os_version: string;
    arch: string;
  };
  models: Model[];
  loaded_models: string[];
  series: { bucket: string; count: number; avg_ms: number | null; tokens: number }[];
  recent_requests: RequestRow[];
  routing: RouterConfig;
  remote: { running: boolean; public_url: string | null; api_base_url: string | null; installed: boolean; health?: { healthy: boolean | null; checked_at: number | null } };
  recommendation: Record<string, unknown>;
  readiness: { ready: boolean; llama_binary: boolean; available_models: number; registered_models: number; loaded_models: string[] };
}

export interface RouterConfig {
  strategy: "balanced" | "quality" | "latency" | "local";
  llm_enabled: boolean;
  llm_model: string;
  llm_timeout_ms: number;
  ambiguity_threshold: number;
  prioritize_loaded: boolean;
  fallback_enabled: boolean;
  allow_general_fallback: boolean;
  max_alternatives: number;
}

export interface RoutingResponse {
  config: RouterConfig;
  aliases: {
    role_aliases: Record<string, { description: string; kind: string }>;
    model_aliases: Record<string, { description: string; kind: string }>;
  };
  models: {
    id: string;
    name: string;
    capabilities: string[];
    parameter_size: string;
    quantization: string;
    priority: number;
    enabled: boolean;
    loaded: boolean;
    performance: Performance | null;
  }[];
  history: RoutingHistory[];
  perf: { model_id: string; requests: number; completed: number; avg_ms: number | null; prompt_tokens: number; completion_tokens: number }[];
  strategies: string[];
}

export interface ApiKey {
  id: string;
  name: string;
  role: "admin" | "inference";
  prefix: string;
  permissions: Record<string, unknown>;
  rate_limit_rpm: number;
  rate_limit_tpm: number;
  created_at: string;
  last_used_at: string | null;
  expires_at: string | null;
  revoked_at: string | null;
  revoked: boolean;
}

export interface KeyUsage {
  id: string;
  name: string;
  role: string;
  prefix: string;
  rate_limit_rpm: number;
  rate_limit_tpm: number;
  last_used_at: string | null;
  revoked_at: string | null;
  requests_24h: number;
  tokens_24h: number | null;
}

export interface KeysResponse {
  keys: ApiKey[];
  usage: KeyUsage[];
  limits: { default_rpm: number; default_tpm: number; admin_rpm: number };
}

export interface RemoteStatus {
  installed: boolean;
  binary: string | null;
  supported: boolean;
  running: boolean;
  mode: string | null;
  public_url: string | null;
  api_base_url: string | null;
  started_at: number | null;
  uptime_s: number | null;
  pid: number | null;
  health: { healthy: boolean | null; checked_at: number | null; error: string | null };
  last_error: string | null;
  log_tail: string;
  target: string;
  admin_exposed: boolean;
  note: string;
  prerequisites: Record<string, string>;
}

export interface EventRow {
  id: number;
  model_id: string | null;
  level: string;
  event: string;
  message: string;
  metadata: Record<string, unknown>;
  created_at: string;
}

export interface SystemMetrics {
  hardware: Record<string, unknown> & {
    cpu_model: string;
    physical_cores: number;
    logical_cores: number;
    total_ram_bytes: number;
    gpus: { name: string; vendor: string; total_mb: number; free_mb: number; utilization: number | null; backend: string }[];
    backend?: string;
    llama_backend?: string;
    os_name: string;
    os_version: string;
    arch: string;
    warnings: string[];
  };
  resources: Overview["resources"];
  process: { pid: number; rss_bytes: number; threads: number; uptime_s: number };
  recommendation: Record<string, unknown>;
}

export interface AppInfo {
  version: string;
  public_api_base: string;
  admin_url: string;
  dashboard_built: boolean;
  docs: Record<string, string>;
  capabilities: string[];
}

export interface DiscoveredModel {
  path: string;
  name: string;
  valid: boolean;
  error?: string;
  size_bytes?: number;
  quantization?: string;
  parameter_size?: string;
  context_length?: number;
  capabilities?: string[];
}
