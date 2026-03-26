import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  ListToolsRequestSchema,
  CallToolRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

const agentId = process.env.ROUNDROBIN_AGENT_ID!;
const agentName = process.env.ROUNDROBIN_AGENT_NAME || agentId;
const hubWsUrl = process.env.ROUNDROBIN_HUB_URL!;
const hubHttpUrl = process.env.ROUNDROBIN_HUB_HTTP!;

if (!agentId || !hubWsUrl || !hubHttpUrl) {
  console.error("Missing required env vars: ROUNDROBIN_AGENT_ID, ROUNDROBIN_HUB_URL, ROUNDROBIN_HUB_HTTP");
  process.exit(1);
}

// --- MCP Server ---
const mcp = new Server(
  { name: "roundrobin-chat", version: "0.1.0" },
  {
    capabilities: {
      experimental: { "claude/channel": {} },
      tools: {},
    },
    instructions: `You are connected to a shared chat room with other coding agents working on the same task.
Messages from other agents arrive as <channel source="roundrobin-chat" from="agent-name"> tags.
Use the send_message tool to communicate with other agents.
Use the read_history tool to catch up on messages you may have missed.`,
  }
);

// --- Tools ---
mcp.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "send_message",
      description: "Send a message to the shared chat room. Other agents will see it immediately.",
      inputSchema: {
        type: "object" as const,
        properties: {
          message: { type: "string", description: "The message to send to other agents" },
        },
        required: ["message"],
      },
    },
    {
      name: "read_history",
      description: "Read recent chat history. Use this when you first start or want to catch up on missed messages.",
      inputSchema: {
        type: "object" as const,
        properties: {
          limit: { type: "number", description: "Number of recent messages to fetch (default: 50)" },
        },
        required: [],
      },
    },
    {
      name: "declare_consensus",
      description: "Declare that you believe the group has reached consensus on a solution. Include a summary of the agreed approach. All agents must declare consensus for the run to complete.",
      inputSchema: {
        type: "object" as const,
        properties: {
          summary: { type: "string", description: "Summary of the consensus solution" },
          chosen_workspace: { type: "string", description: 'Which agent workspace has the best implementation (e.g., "agent-1")' },
        },
        required: ["summary"],
      },
    },
  ],
}));

mcp.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  if (name === "send_message") {
    const message = (args as any).message;
    if (hubWs && hubWs.readyState === WebSocket.OPEN) {
      hubWs.send(JSON.stringify({
        type: "message",
        agent_id: agentId,
        content: message,
        timestamp: new Date().toISOString(),
      }));
    }
    return { content: [{ type: "text", text: `Message sent: "${message}"` }] };
  }

  if (name === "read_history") {
    const limit = (args as any)?.limit || 50;
    try {
      const resp = await fetch(`${hubHttpUrl}/api/messages?limit=${limit}`);
      const messages = await resp.json();
      const formatted = (messages as any[])
        .map((m: any) => `[${m.timestamp}] ${m.agent_id}: ${m.content}`)
        .join("\n");
      return { content: [{ type: "text", text: formatted || "(no messages yet)" }] };
    } catch (e: any) {
      return { content: [{ type: "text", text: `Error fetching history: ${e.message}` }] };
    }
  }

  if (name === "declare_consensus") {
    const summary = (args as any).summary;
    const chosenWorkspace = (args as any)?.chosen_workspace;
    const content = chosenWorkspace
      ? `${summary} (chosen workspace: ${chosenWorkspace})`
      : summary;

    if (hubWs && hubWs.readyState === WebSocket.OPEN) {
      hubWs.send(JSON.stringify({
        type: "consensus",
        agent_id: agentId,
        content,
        timestamp: new Date().toISOString(),
      }));
    }
    return { content: [{ type: "text", text: `Consensus declared: "${summary}"` }] };
  }

  return { content: [{ type: "text", text: `Unknown tool: ${name}` }] };
});

// --- Hub WebSocket ---
let hubWs: WebSocket;

function connectToHub() {
  const url = `${hubWsUrl}?agent_id=${encodeURIComponent(agentId)}`;
  hubWs = new WebSocket(url);

  hubWs.onopen = () => {
    console.error(`[roundrobin-chat] Connected to hub as ${agentId} (${agentName})`);
  };

  hubWs.onmessage = async (event) => {
    const data = typeof event.data === "string" ? event.data : event.data.toString();
    let msg: any;
    try {
      msg = JSON.parse(data);
    } catch {
      return;
    }

    // Push to Claude via channel notification
    try {
      await mcp.notification({
        method: "notifications/claude/channel",
        params: {
          content: msg.content,
          meta: {
            from: msg.agent_id,
            type: msg.type,
            timestamp: msg.timestamp,
          },
        },
      });
    } catch (e: any) {
      console.error(`[roundrobin-chat] Failed to push notification: ${e.message}`);
    }
  };

  hubWs.onclose = () => {
    console.error(`[roundrobin-chat] Disconnected from hub, reconnecting in 2s...`);
    setTimeout(connectToHub, 2000);
  };

  hubWs.onerror = (err) => {
    console.error(`[roundrobin-chat] WebSocket error:`, err);
  };
}

// --- Start ---
async function main() {
  connectToHub();
  const transport = new StdioServerTransport();
  await mcp.connect(transport);
  console.error(`[roundrobin-chat] MCP server running for ${agentId}`);
}

main().catch((e) => {
  console.error("Fatal:", e);
  process.exit(1);
});
