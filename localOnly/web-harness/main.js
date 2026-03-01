const $ = (id) => {
  const element = document.getElementById(id);
  if (!element) {
    throw new Error(`Missing element: ${id}`);
  }
  return element;
};

const state = {
  token: null,
  route: null,
};

const text = (id, value) => {
  $(id).textContent = value;
};

const valueOf = (id) => $(id).value.trim();

const authHeaders = () => {
  if (!state.token) {
    throw new Error("Authenticate first");
  }
  return { Authorization: `Bearer ${state.token}` };
};

async function controlFetch(path, init = {}, { withCredentials = false } = {}) {
  const base = valueOf("controlPlaneUrl").replace(/\/+$/, "");
  const response = await fetch(`${base}${path}`, {
    ...init,
    credentials: withCredentials ? "include" : "same-origin",
  });
  const raw = await response.text();
  const payload = raw ? JSON.parse(raw) : null;

  if (!response.ok) {
    const detail = payload?.detail ?? response.statusText;
    throw new Error(`${path} -> ${response.status}: ${detail}`);
  }
  return payload;
}

async function edgeFetch(path, init = {}, { withCredentials = true } = {}) {
  const base = valueOf("edgeRouterUrl").replace(/\/+$/, "");
  const response = await fetch(`${base}${path}`, {
    ...init,
    credentials: withCredentials ? "include" : "same-origin",
  });

  const raw = await response.text();
  let payload = raw;
  try {
    payload = raw ? JSON.parse(raw) : null;
  } catch {
    payload = raw;
  }

  if (!response.ok && response.status !== 202) {
    const detail = payload?.detail ?? response.statusText;
    throw new Error(`${path} -> ${response.status}: ${detail}`);
  }

  return {
    status: response.status,
    headers: Object.fromEntries(response.headers.entries()),
    body: payload,
  };
}

async function authenticate(path) {
  const payload = {
    email: valueOf("email"),
    password: valueOf("password"),
  };
  const auth = await controlFetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  state.token = auth.access_token;
  text("authStatus", `Authenticated as ${auth.user.email}`);
}

async function setWebSessionCookie() {
  await controlFetch(
    "/auth/web-session",
    {
      method: "POST",
      headers: authHeaders(),
    },
    { withCredentials: true }
  );
  text("authStatus", "Web session cookie set for gateway routing.");
}

async function clearWebSessionCookie() {
  await controlFetch(
    "/auth/web-session",
    {
      method: "DELETE",
    },
    { withCredentials: true }
  );
  text("authStatus", "Web session cookie cleared.");
}

async function parseChart() {
  const body = {
    repo_path: valueOf("repoPath") || null,
    repo_url: valueOf("repoUrl") || null,
    ref: valueOf("repoRef") || null,
  };
  const result = await controlFetch("/apps/charts/parse", {
    method: "POST",
    headers: {
      ...authHeaders(),
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });
  text("appOutput", JSON.stringify(result, null, 2));
}

async function registerFromChart() {
  const body = {
    slug: valueOf("appSlug"),
    name: valueOf("appName"),
    docker_image: valueOf("dockerImage"),
    repo_path: valueOf("repoPath") || null,
    repo_url: valueOf("repoUrl") || null,
    ref: valueOf("repoRef") || null,
    fly_app_name: valueOf("flyAppName") || null,
  };

  const result = await controlFetch("/apps/from-chart", {
    method: "POST",
    headers: {
      ...authHeaders(),
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });
  if (result?.app?.id) {
    $("appId").value = result.app.id;
  }
  text("appOutput", JSON.stringify(result, null, 2));
}

async function listApps() {
  const result = await controlFetch("/apps", {
    method: "GET",
    headers: authHeaders(),
  });
  text("appOutput", JSON.stringify(result, null, 2));
}

async function provisionPod() {
  const appId = valueOf("appId");
  if (!appId) {
    throw new Error("App ID is required for provisioning");
  }
  const memberUserIdsRaw = valueOf("memberUserIds") || "[]";
  const memberUserIds = JSON.parse(memberUserIdsRaw);

  const result = await controlFetch("/pods/provision", {
    method: "POST",
    headers: {
      ...authHeaders(),
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      app_id: appId,
      member_user_ids: memberUserIds,
    }),
  });
  text("appOutput", JSON.stringify(result, null, 2));
}

async function resolveRoute() {
  const slug = valueOf("routeSlug");
  const route = await controlFetch(`/pods/routing/apps/${slug}`, {
    method: "GET",
    headers: authHeaders(),
  });
  state.route = route;
  text("routeOutput", JSON.stringify(route, null, 2));
}

async function callPod(method) {
  const slug = valueOf("routeSlug");
  const path = valueOf("podPath");
  const body = valueOf("podBody");

  const headers = {};
  if (state.token) {
    headers.Authorization = `Bearer ${state.token}`;
  }
  if (method === "POST") {
    headers["Content-Type"] = "application/json";
  }

  const result = await edgeFetch(`/${slug}${path}`, {
    method,
    headers,
    body: method === "POST" ? body : undefined,
  });
  text("routeOutput", JSON.stringify(result, null, 2));
}

function openWebApp() {
  const slug = valueOf("routeSlug");
  const base = valueOf("edgeRouterUrl").replace(/\/+$/, "");
  window.open(`${base}/${slug}/`, "_blank");
}

function bind(id, fn) {
  $(id).addEventListener("click", async () => {
    try {
      await fn();
    } catch (error) {
      text("routeOutput", String(error.message || error));
    }
  });
}

bind("registerBtn", () => authenticate("/auth/register"));
bind("loginBtn", () => authenticate("/auth/login"));
bind("setWebSessionBtn", setWebSessionCookie);
bind("logoutCookieBtn", clearWebSessionCookie);
bind("whoamiBtn", async () => {
  const result = await controlFetch("/auth/me", {
    method: "GET",
    headers: authHeaders(),
  });
  text("authStatus", JSON.stringify(result));
});
bind("parseChartBtn", parseChart);
bind("registerFromChartBtn", registerFromChart);
bind("listAppsBtn", listApps);
bind("provisionPodBtn", provisionPod);
bind("resolveRouteBtn", resolveRoute);
bind("openWebAppBtn", async () => openWebApp());
bind("podGetBtn", () => callPod("GET"));
bind("podPostBtn", () => callPod("POST"));
