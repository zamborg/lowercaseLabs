const stripTrailingSlash = (value: string): string => value.replace(/\/+$/, "");

export const CONTROL_PLANE_URL = stripTrailingSlash(
  process.env.EXPO_PUBLIC_CONTROL_PLANE_URL ?? "http://localhost:8000"
);

export const EDGE_ROUTER_URL = stripTrailingSlash(
  process.env.EXPO_PUBLIC_EDGE_ROUTER_URL ?? "http://localhost:8001"
);
