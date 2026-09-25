#!/usr/bin/env node
/**
 * antigravity-code-critic — MCP Server
 *
 * An adversarial, principal-engineer-grade code critique tool that reads
 * your live git diff and evaluates it against a stated objective using a
 * powerful LLM via OpenRouter. Runs globally via `npx` or as a binary.
 *
 * Environment variables:
 *   OPENROUTER_API_KEY  (required) — your OpenRouter API key
 *   CRITIC_MODEL        (optional) — override the LLM model used
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  type CallToolResult,
} from "@modelcontextprotocol/sdk/types.js";
import { z } from "zod";
import { execSync } from "child_process";
import https from "https";

// ---------------------------------------------------------------------------
// Constants & configuration
// ---------------------------------------------------------------------------

const SERVER_NAME = "antigravity-code-critic";
const SERVER_VERSION = "1.0.0";
const OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions";
const DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct:free";
const FALLBACK_MODEL = "qwen/qwen-2.5-coder-32b-instruct:free";
const MAX_BUFFER = 10 * 1024 * 1024; // 10 MB

// ---------------------------------------------------------------------------
// Zod schema for verify_feature tool arguments
// ---------------------------------------------------------------------------

const VerifyFeatureArgsSchema = z.object({
  objective: z
    .string()
    .min(1, "Objective must not be empty")
    .describe(
      "The intended requirement, user story, or bug fix description to verify against."
    ),
  targetBranch: z
    .string()
    .optional()
    .describe(
      "Base branch for diffing (e.g. main, master, develop). If omitted, diffs uncommitted changes against HEAD."
    ),
});

type VerifyFeatureArgs = z.infer<typeof VerifyFeatureArgsSchema>;

// ---------------------------------------------------------------------------
// System prompt — adversarial principal engineer persona
// ---------------------------------------------------------------------------

const SYSTEM_PROMPT = `You are an uncompromising adversarial principal engineer and QA reviewer.
Your sole mission is to identify every way the submitted code change can fail, regress existing behaviour,
violate the stated objective, or introduce security, performance, or reliability hazards.

You must apply the following mental models to every review:
- Boundary conditions: off-by-one, empty inputs, null/undefined, integer overflow.
- Regression risks: does this change break anything that previously worked?
- Interface breakages: changed signatures, removed exports, altered API contracts.
- Unhandled exceptions: missing try/catch, unhandled promise rejections, missing error branches.
- Security: injection, hardcoded secrets, over-broad permissions, missing input validation.
- Concurrency: race conditions, non-atomic operations, missing locks or guards.
- Deviation from objective: does the implementation actually satisfy the stated requirement?

You MUST produce your response in exactly this format — no deviations:

### Verdict: [PASS] | [ACTION REQUIRED]

### Critical Issues & Regression Risks
List each finding with:
- File path and line reference (e.g. \`src/api/handler.ts:L42\`)
- Severity: CRITICAL | HIGH | MEDIUM | LOW
- Description: precise, technical, actionable

If no issues exist, write: "None detected."

### Recommended Remediations
For each finding, provide a concrete, specific code-level remediation. Be explicit — do not give vague advice.

If verdict is PASS, confirm: "The implementation satisfies the stated objective with no detected risks."`;

// ---------------------------------------------------------------------------
// Git diff helper
// ---------------------------------------------------------------------------

function getGitDiff(targetBranch?: string): string {
  const cwd = process.cwd();

  if (targetBranch) {
    // Compare target branch to HEAD — useful for PR reviews
    return execSync(`git diff ${targetBranch}...HEAD`, {
      cwd,
      maxBuffer: MAX_BUFFER,
    }).toString("utf8");
  }

  // Try to diff staged + unstaged changes vs HEAD
  try {
    const diff = execSync("git diff HEAD", {
      cwd,
      maxBuffer: MAX_BUFFER,
    }).toString("utf8");

    if (diff.trim()) return diff;

    // HEAD might be unset (fresh repo with no commits); fall back to index diff
    const indexDiff = execSync("git diff", {
      cwd,
      maxBuffer: MAX_BUFFER,
    }).toString("utf8");

    return indexDiff;
  } catch {
    // Fallback: bare diff with no HEAD reference
    return execSync("git diff", {
      cwd,
      maxBuffer: MAX_BUFFER,
    }).toString("utf8");
  }
}

// ---------------------------------------------------------------------------
// OpenRouter HTTPS request helper (no external HTTP libraries needed)
// ---------------------------------------------------------------------------

function httpsPost(
  url: string,
  headers: Record<string, string>,
  body: string
): Promise<string> {
  return new Promise((resolve, reject) => {
    const parsed = new URL(url);
    const options: https.RequestOptions = {
      hostname: parsed.hostname,
      port: 443,
      path: parsed.pathname + parsed.search,
      method: "POST",
      headers: {
        ...headers,
        "Content-Type": "application/json",
        "Content-Length": Buffer.byteLength(body),
      },
    };

    const req = https.request(options, (res) => {
      const chunks: Buffer[] = [];
      res.on("data", (chunk: Buffer) => chunks.push(chunk));
      res.on("end", () => {
        const raw = Buffer.concat(chunks).toString("utf8");
        if ((res.statusCode ?? 0) >= 400) {
          reject(
            new Error(
              `OpenRouter HTTP ${res.statusCode}: ${raw.slice(0, 500)}`
            )
          );
        } else {
          resolve(raw);
        }
      });
    });

    req.on("error", reject);
    req.write(body);
    req.end();
  });
}

// ---------------------------------------------------------------------------
// Core critique logic
// ---------------------------------------------------------------------------

async function runCritique(args: VerifyFeatureArgs): Promise<string> {
  // 1. Validate API key
  const apiKey = process.env["OPENROUTER_API_KEY"];
  if (!apiKey || !apiKey.trim()) {
    return [
      "**Error: OPENROUTER_API_KEY is not set.**",
      "",
      "Please export it before starting the MCP server:",
      "```",
      "export OPENROUTER_API_KEY=sk-or-...",
      "```",
      "You can obtain a free key at https://openrouter.ai",
    ].join("\n");
  }

  // 2. Read git diff
  let diff: string;
  try {
    diff = getGitDiff(args.targetBranch);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return `**Error reading git diff:** ${msg}\n\nEnsure the MCP server is invoked from within a valid git repository.`;
  }

  if (!diff.trim()) {
    return "**No git diff detected in the current workspace.**\n\nEnsure there are staged, unstaged, or branch changes before invoking this tool.";
  }

  // 3. Determine model
  const model =
    process.env["CRITIC_MODEL"]?.trim() || DEFAULT_MODEL;

  // 4. Build prompt
  const userContent = [
    `## Objective`,
    args.objective,
    ``,
    `## Git Diff`,
    "```diff",
    diff,
    "```",
  ].join("\n");

  const requestBody = JSON.stringify({
    model,
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: userContent },
    ],
    temperature: 0.2,
    max_tokens: 4096,
  });

  // 5. Call OpenRouter
  let rawResponse: string;
  try {
    rawResponse = await httpsPost(
      OPENROUTER_URL,
      {
        Authorization: `Bearer ${apiKey}`,
        "HTTP-Referer": "https://github.com",
        "X-Title": "Antigravity Code Critic",
      },
      requestBody
    );
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);

    // On model error, retry with fallback
    if (model !== FALLBACK_MODEL) {
      try {
        const fallbackBody = JSON.stringify({
          model: FALLBACK_MODEL,
          messages: [
            { role: "system", content: SYSTEM_PROMPT },
            { role: "user", content: userContent },
          ],
          temperature: 0.2,
          max_tokens: 4096,
        });
        rawResponse = await httpsPost(
          OPENROUTER_URL,
          {
            Authorization: `Bearer ${apiKey}`,
            "HTTP-Referer": "https://github.com",
            "X-Title": "Antigravity Code Critic",
          },
          fallbackBody
        );
      } catch (fallbackErr) {
        const fbMsg =
          fallbackErr instanceof Error ? fallbackErr.message : String(fallbackErr);
        return `**Error calling OpenRouter (primary & fallback both failed):**\nPrimary (${model}): ${msg}\nFallback (${FALLBACK_MODEL}): ${fbMsg}`;
      }
    } else {
      return `**Error calling OpenRouter:** ${msg}`;
    }
  }

  // 6. Parse and return the critique
  interface OpenRouterChoice {
    message?: { content?: string };
  }
  interface OpenRouterResponse {
    choices?: OpenRouterChoice[];
    error?: { message?: string };
  }

  let parsed: OpenRouterResponse;
  try {
    parsed = JSON.parse(rawResponse) as OpenRouterResponse;
  } catch {
    return `**Error parsing OpenRouter response:**\n\`\`\`\n${rawResponse.slice(0, 1000)}\n\`\`\``;
  }

  if (parsed.error?.message) {
    return `**OpenRouter API Error:** ${parsed.error.message}`;
  }

  const content = parsed.choices?.[0]?.message?.content;
  if (!content) {
    return `**Unexpected OpenRouter response structure:**\n\`\`\`json\n${JSON.stringify(parsed, null, 2).slice(0, 1000)}\n\`\`\``;
  }

  return content.trim();
}

// ---------------------------------------------------------------------------
// MCP Server setup
// ---------------------------------------------------------------------------

const server = new Server(
  { name: SERVER_NAME, version: SERVER_VERSION },
  {
    capabilities: {
      tools: {},
    },
  }
);

// List tools handler
server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "verify_feature",
      description:
        "Reads the active git diff of the current workspace and applies an adversarial principal-engineer critique against the stated objective. Returns a structured verdict with critical issues, regression risks, and concrete remediations.",
      inputSchema: {
        type: "object",
        properties: {
          objective: {
            type: "string",
            description:
              "The intended requirement, user story, or bug fix description to verify against.",
          },
          targetBranch: {
            type: "string",
            description:
              "Optional base branch for diffing (e.g. main, master, develop). If omitted, diffs uncommitted changes against HEAD.",
          },
        },
        required: ["objective"],
      },
    },
  ],
}));

// Call tool handler
server.setRequestHandler(CallToolRequestSchema, async (request): Promise<{ content: CallToolResult["content"] }> => {
  if (request.params.name !== "verify_feature") {
    return {
      content: [
        {
          type: "text",
          text: `Unknown tool: ${request.params.name}`,
        },
      ],
    };
  }

  let args: VerifyFeatureArgs;
  try {
    args = VerifyFeatureArgsSchema.parse(request.params.arguments);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return {
      content: [
        {
          type: "text",
          text: `**Invalid tool arguments:** ${msg}`,
        },
      ],
    };
  }

  let result: string;
  try {
    result = await runCritique(args);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    result = `**Unexpected error in code critic:** ${msg}`;
  }

  return {
    content: [
      {
        type: "text",
        text: result,
      },
    ],
  };
});

// ---------------------------------------------------------------------------
// Start server
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  // Log to stderr so it doesn't corrupt the MCP stdio protocol
  process.stderr.write(
    `[${SERVER_NAME}] v${SERVER_VERSION} running on stdio\n`
  );
}

main().catch((err) => {
  process.stderr.write(
    `[${SERVER_NAME}] Fatal error: ${err instanceof Error ? err.message : String(err)}\n`
  );
  process.exit(1);
});
