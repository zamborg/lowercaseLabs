import * as SecureStore from "expo-secure-store";
import { useEffect, useMemo, useState } from "react";
import {
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";

import { CONTROL_PLANE_URL, EDGE_ROUTER_URL } from "./src/config";
import {
  AppRecord,
  AuthTokenResponse,
  RouteResolution,
  SovereignApiClient,
} from "./src/lib/sovereignApiClient";

const TOKEN_KEY = "sovereign_app_store_token";

export default function App() {
  const client = useMemo(
    () =>
      new SovereignApiClient({
        controlPlaneUrl: CONTROL_PLANE_URL,
        edgeRouterUrl: EDGE_ROUTER_URL,
      }),
    []
  );

  const [email, setEmail] = useState("demo@local.dev");
  const [password, setPassword] = useState("password123");
  const [registerMode, setRegisterMode] = useState(false);
  const [token, setToken] = useState<string | null>(null);
  const [status, setStatus] = useState("Not authenticated");
  const [apps, setApps] = useState<AppRecord[]>([]);
  const [appSlug, setAppSlug] = useState("");
  const [requestPath, setRequestPath] = useState("/healthz");
  const [requestMethod, setRequestMethod] = useState<"GET" | "POST">("GET");
  const [requestBody, setRequestBody] = useState('{"hello":"world"}');
  const [resolvedRoute, setResolvedRoute] = useState<RouteResolution | null>(null);
  const [response, setResponse] = useState("No request sent yet.");

  useEffect(() => {
    const bootstrap = async () => {
      const storedToken = await SecureStore.getItemAsync(TOKEN_KEY);
      if (!storedToken) {
        return;
      }

      client.setToken(storedToken);
      setToken(storedToken);
      setStatus("Restored saved session");
      await refreshApps();
    };

    void bootstrap();
  }, []);

  const saveSession = async (auth: AuthTokenResponse) => {
    client.setToken(auth.access_token);
    setToken(auth.access_token);
    await SecureStore.setItemAsync(TOKEN_KEY, auth.access_token);
    setStatus(`Authenticated as ${auth.user.email}`);
  };

  const handleAuth = async () => {
    try {
      const auth = registerMode
        ? await client.register(email, password)
        : await client.login(email, password);
      await saveSession(auth);
      await refreshApps();
    } catch (error) {
      setStatus((error as Error).message);
    }
  };

  const refreshApps = async () => {
    try {
      const catalog = await client.listApps();
      setApps(catalog);
      if (!appSlug && catalog[0]) {
        setAppSlug(catalog[0].slug);
      }
      setStatus(`Loaded ${catalog.length} app(s) from registry`);
    } catch (error) {
      setStatus((error as Error).message);
    }
  };

  const resolveRoute = async () => {
    if (!appSlug) {
      setStatus("Choose an app slug first");
      return;
    }

    try {
      const route = await client.resolveRoute(appSlug, true);
      setResolvedRoute(route);
      setStatus(`Resolved ${appSlug} route in ${route.networking_mode} mode`);
    } catch (error) {
      setStatus((error as Error).message);
    }
  };

  const sendRequest = async () => {
    if (!appSlug) {
      setStatus("Choose an app slug first");
      return;
    }

    try {
      const body =
        requestMethod === "POST"
          ? requestBody
            ? JSON.parse(requestBody)
            : {}
          : undefined;

      const route = await client.resolveRoute(appSlug);
      setResolvedRoute(route);

      const payload = await client.callPod<unknown>({
        appSlug,
        path: requestPath,
        method: requestMethod,
        body,
      });

      setResponse(JSON.stringify(payload, null, 2));
      setStatus("Request succeeded");
    } catch (error) {
      setResponse((error as Error).message);
      setStatus("Request failed");
    }
  };

  const logout = async () => {
    client.setToken(null);
    setToken(null);
    setResolvedRoute(null);
    setResponse("No request sent yet.");
    await SecureStore.deleteItemAsync(TOKEN_KEY);
    setStatus("Logged out");
  };

  return (
    <SafeAreaView style={styles.root}>
      <ScrollView contentContainerStyle={styles.container}>
        <Text style={styles.title}>Sovereign App Store iOS Harness</Text>
        <Text style={styles.subtitle}>Control Plane: {CONTROL_PLANE_URL}</Text>
        <Text style={styles.subtitle}>Edge Router: {EDGE_ROUTER_URL}</Text>
        <Text style={styles.status}>{status}</Text>

        {!token ? (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>{registerMode ? "Register" : "Login"}</Text>
            <TextInput
              autoCapitalize="none"
              value={email}
              onChangeText={setEmail}
              placeholder="email"
              style={styles.input}
            />
            <TextInput
              secureTextEntry
              value={password}
              onChangeText={setPassword}
              placeholder="password"
              style={styles.input}
            />
            <TouchableOpacity style={styles.button} onPress={handleAuth}>
              <Text style={styles.buttonText}>{registerMode ? "Create account" : "Sign in"}</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.secondaryButton}
              onPress={() => setRegisterMode((value) => !value)}
            >
              <Text style={styles.secondaryButtonText}>
                {registerMode ? "Use login" : "Use register"}
              </Text>
            </TouchableOpacity>
          </View>
        ) : (
          <>
            <View style={styles.row}>
              <TouchableOpacity style={styles.secondaryButton} onPress={refreshApps}>
                <Text style={styles.secondaryButtonText}>Refresh apps</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.secondaryButton} onPress={logout}>
                <Text style={styles.secondaryButtonText}>Logout</Text>
              </TouchableOpacity>
            </View>

            <View style={styles.card}>
              <Text style={styles.cardTitle}>Routing</Text>
              <TextInput
                value={appSlug}
                onChangeText={setAppSlug}
                placeholder="app slug"
                style={styles.input}
              />
              <TouchableOpacity style={styles.button} onPress={resolveRoute}>
                <Text style={styles.buttonText}>Resolve mode and base URL</Text>
              </TouchableOpacity>
              <Text style={styles.codeText}>
                {resolvedRoute
                  ? `${resolvedRoute.networking_mode.toUpperCase()} -> ${resolvedRoute.base_url}`
                  : "Route not resolved"}
              </Text>
            </View>

            <View style={styles.card}>
              <Text style={styles.cardTitle}>Pod API Call</Text>
              <View style={styles.row}>
                <TouchableOpacity
                  style={requestMethod === "GET" ? styles.button : styles.secondaryButton}
                  onPress={() => setRequestMethod("GET")}
                >
                  <Text
                    style={requestMethod === "GET" ? styles.buttonText : styles.secondaryButtonText}
                  >
                    GET
                  </Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={requestMethod === "POST" ? styles.button : styles.secondaryButton}
                  onPress={() => setRequestMethod("POST")}
                >
                  <Text
                    style={requestMethod === "POST" ? styles.buttonText : styles.secondaryButtonText}
                  >
                    POST
                  </Text>
                </TouchableOpacity>
              </View>
              <TextInput
                value={requestPath}
                onChangeText={setRequestPath}
                placeholder="/healthz"
                style={styles.input}
              />
              {requestMethod === "POST" ? (
                <TextInput
                  multiline
                  value={requestBody}
                  onChangeText={setRequestBody}
                  placeholder='{"hello":"world"}'
                  style={[styles.input, styles.multilineInput]}
                />
              ) : null}
              <TouchableOpacity style={styles.button} onPress={sendRequest}>
                <Text style={styles.buttonText}>Send Request</Text>
              </TouchableOpacity>
              <Text style={styles.codeText}>{response}</Text>
            </View>

            <View style={styles.card}>
              <Text style={styles.cardTitle}>App Catalog</Text>
              <Text style={styles.codeText}>
                {apps.length > 0
                  ? apps.map((app) => `${app.slug} (${app.networking_mode})`).join("\n")
                  : "No apps available"}
              </Text>
            </View>
          </>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: "#101010",
  },
  container: {
    padding: 20,
    gap: 12,
  },
  title: {
    color: "#f7f7f7",
    fontSize: 24,
    fontWeight: "700",
  },
  subtitle: {
    color: "#c9c9c9",
    fontSize: 12,
  },
  status: {
    color: "#84e1bc",
    fontSize: 13,
  },
  card: {
    backgroundColor: "#1f1f1f",
    borderRadius: 12,
    padding: 14,
    gap: 10,
  },
  cardTitle: {
    color: "#ffffff",
    fontSize: 16,
    fontWeight: "600",
  },
  input: {
    backgroundColor: "#0d0d0d",
    color: "#f3f3f3",
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "#363636",
    paddingHorizontal: 10,
    paddingVertical: 10,
  },
  multilineInput: {
    minHeight: 90,
    textAlignVertical: "top",
  },
  button: {
    backgroundColor: "#2e7efc",
    borderRadius: 8,
    paddingVertical: 10,
    paddingHorizontal: 12,
  },
  buttonText: {
    color: "#ffffff",
    textAlign: "center",
    fontWeight: "600",
  },
  secondaryButton: {
    backgroundColor: "#2b2b2b",
    borderRadius: 8,
    paddingVertical: 10,
    paddingHorizontal: 12,
    borderWidth: 1,
    borderColor: "#444",
  },
  secondaryButtonText: {
    color: "#efefef",
    textAlign: "center",
    fontWeight: "500",
  },
  row: {
    flexDirection: "row",
    gap: 10,
  },
  codeText: {
    fontFamily: "Courier",
    color: "#cdd3e1",
    fontSize: 12,
  },
});
