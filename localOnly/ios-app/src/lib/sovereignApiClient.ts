export type NetworkingMode = "proxy" | "direct";

export interface AuthUser {
  id: string;
  email: string;
  created_at: string;
}

export interface AuthTokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
  user: AuthUser;
}

export interface AppRecord {
  id: string;
  slug: string;
  name: string;
  docker_image: string;
  fly_app_name: string;
  networking_mode: NetworkingMode;
  internal_port: number;
  requires_volume: boolean;
  created_at: string;
}

export interface RouteResolution {
  app_slug: string;
  networking_mode: NetworkingMode;
  pod_id: string;
  base_url: string;
  fly_machine_id: string;
}

interface ClientOptions {
  controlPlaneUrl: string;
  edgeRouterUrl: string;
  token?: string;
}

interface PodRequest {
  appSlug: string;
  path: string;
  method?: "GET" | "POST" | "PUT" | "PATCH" | "DELETE";
  headers?: Record<string, string>;
  body?: unknown;
}

export class SovereignApiClient {
  private token?: string;
  private routeCache: Map<string, RouteResolution> = new Map();

  constructor(private readonly options: ClientOptions) {
    this.token = options.token;
  }

  public setToken(token: string | null): void {
    this.token = token ?? undefined;
    if (!token) {
      this.routeCache.clear();
    }
  }

  public getToken(): string | undefined {
    return this.token;
  }

  private async controlFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
    const headers = new Headers(init.headers ?? {});
    if (this.token) {
      headers.set("Authorization", `Bearer ${this.token}`);
    }
    if (!headers.has("Content-Type") && init.body) {
      headers.set("Content-Type", "application/json");
    }

    const response = await fetch(`${this.options.controlPlaneUrl}${path}`, {
      ...init,
      headers,
    });

    const responseText = await response.text();
    const payload = responseText ? JSON.parse(responseText) : null;

    if (!response.ok) {
      const detail = payload?.detail ?? response.statusText;
      throw new Error(`Control Plane error (${response.status}): ${detail}`);
    }

    return payload as T;
  }

  public async register(email: string, password: string): Promise<AuthTokenResponse> {
    return this.controlFetch<AuthTokenResponse>("/auth/register", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    });
  }

  public async login(email: string, password: string): Promise<AuthTokenResponse> {
    return this.controlFetch<AuthTokenResponse>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    });
  }

  public async listApps(): Promise<AppRecord[]> {
    return this.controlFetch<AppRecord[]>("/apps", {
      method: "GET",
    });
  }

  public async resolveRoute(appSlug: string, force = false): Promise<RouteResolution> {
    if (!this.token) {
      throw new Error("Missing auth token");
    }

    if (!force && this.routeCache.has(appSlug)) {
      return this.routeCache.get(appSlug)!;
    }

    const route = await this.controlFetch<RouteResolution>(
      `/pods/routing/apps/${appSlug}`,
      { method: "GET" }
    );

    if (route.networking_mode === "proxy") {
      route.base_url = `${this.options.edgeRouterUrl}/${appSlug}`;
    }

    this.routeCache.set(appSlug, route);
    return route;
  }

  public async warmRoutes(appSlugs: string[]): Promise<void> {
    await Promise.all(appSlugs.map((slug) => this.resolveRoute(slug).catch(() => null)));
  }

  public clearRoute(appSlug?: string): void {
    if (!appSlug) {
      this.routeCache.clear();
      return;
    }
    this.routeCache.delete(appSlug);
  }

  public async callPod<T>(request: PodRequest): Promise<T> {
    if (!this.token) {
      throw new Error("Missing auth token");
    }

    const route = await this.resolveRoute(request.appSlug);
    const cleanBase = route.base_url.replace(/\/$/, "");
    const cleanPath = request.path.startsWith("/") ? request.path : `/${request.path}`;
    const url = `${cleanBase}${cleanPath}`;

    const headers = new Headers(request.headers ?? {});
    headers.set("Authorization", `Bearer ${this.token}`);
    if (request.body && !headers.has("Content-Type")) {
      headers.set("Content-Type", "application/json");
    }

    const response = await fetch(url, {
      method: request.method ?? "GET",
      headers,
      body: request.body
        ? typeof request.body === "string"
          ? request.body
          : JSON.stringify(request.body)
        : undefined,
    });

    const responseText = await response.text();
    let payload: unknown = responseText;
    try {
      payload = responseText ? JSON.parse(responseText) : null;
    } catch {
      payload = responseText;
    }

    if (!response.ok) {
      const detail =
        typeof payload === "object" && payload && "detail" in payload
          ? String((payload as { detail: unknown }).detail)
          : response.statusText;
      throw new Error(`Pod API error (${response.status}): ${detail}`);
    }

    return payload as T;
  }
}
