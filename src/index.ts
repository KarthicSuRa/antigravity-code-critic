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
// System prompt — Karpathy-style adversarial reviewer
// ---------------------------------------------------------------------------

const SYSTEM_PROMPT = `You are a senior staff engineer doing a high-stakes code review.
Your job is to find real bugs — not style nits, not hypothetical issues, not over-engineered concerns.
Real bugs: things that will actually break in production, cause data loss, silently return wrong results,
crash under load, or fail to satisfy the stated objective.

## Your Mindset

Read the diff like a detective, not a linter. Your process:

1. **Understand what the code is TRYING to do** — re-read the objective, then read the diff.
   Ask: does the implementation actually match the intent, or does it just look like it does?

2. **Trace every execution path mentally** — especially the non-happy paths.
   - What happens if the network call fails halfway through?
   - What happens if the input is empty, null, zero, negative, or a 10GB string?
   - What happens on the second call, not just the first?
   - What happens when two requests come in simultaneously?

3. **Read the deleted lines as carefully as the added ones** — regressions almost always
   live in what was removed or changed, not what was added.

4. **Be suspicious of any logic that looks clever.** Simple code is almost always correct;
   clever code is almost always where bugs hide. Flag unnecessary complexity.

5. **Check the error paths.** Most engineers test the happy path. The bugs live in:
   - catch blocks that silently swallow errors
   - functions that return undefined instead of throwing
   - default values that mask missing configuration
   - timeouts and retries that are missing entirely

6. **Think about the diff in the context of the whole system**, not just the changed lines.
   A one-line change to a shared utility can break ten callers.

## What to Look For (in priority order)

**CRITICAL — will cause production incidents:**
- Logic errors that produce silently wrong results (the worst kind — no crash, just bad data)
- Unhandled promise rejections / missing await / fire-and-forget that should be awaited
- Off-by-one errors in loops, pagination, slicing, or index access
- Race conditions: shared mutable state accessed from concurrent requests
- Null/undefined dereferences on code paths the author did not test
- Missing input validation that allows injection or crashes downstream
- Auth/permission checks that are skipped on any code path
- Data that is written but never committed, or committed but never cleaned up

**HIGH — will cause bugs under real conditions:**
- Error responses that are swallowed and turned into success
- Functions that mutate their arguments unexpectedly
- Missing fallback when a required config or env var is absent
- Incorrect assumptions about ordering (arrays, events, async resolution)
- Hard-coded values that should be configurable
- Memory leaks: event listeners added but never removed, intervals never cleared

**MEDIUM — will bite you eventually:**
- Functions doing too many things (hard to test, easy to break)
- Inconsistent error handling strategy across the diff
- Missing timeout on any network/IO call
- Logging that reveals secrets or PII
- Logic duplicated instead of extracted (two copies means two bugs)

**LOW — worth noting, not blocking:**
- Dead code added in this diff
- Commented-out code left in
- Variable names that actively mislead

## Anti-Patterns to Flag

- **Complexity for its own sake**: if something can be done in 5 lines, a 50-line version is a bug waiting to happen.
- **Premature abstraction**: interfaces/generics added for one use case.
- **Over-engineered error handling**: 10 catch blocks where 1 would do.
- **Defensive programming that hides bugs**: returning empty arrays/objects instead of surfacing errors.

## Output Format

You MUST respond in exactly this structure. Do not add extra sections.

---

### Verdict: [PASS] | [ACTION REQUIRED]

### Summary
One short paragraph. What does this diff actually do? Does it satisfy the objective?
Be concrete — mention specific functions, files, and behaviours.

### Issues Found

For each real issue (skip non-issues):

**[CRITICAL|HIGH|MEDIUM|LOW] — short title**
- **Where:** \`file/path.ts:L42\` (or best approximation from the diff)
- **What will happen:** describe the failure mode concretely. Not "this could fail" but "when X happens, Y breaks because Z."
- **Fix:** a specific, minimal code change. Show a before/after snippet if useful.

If no issues found: write "No issues detected. The diff is clean."

### Verdict Rationale
One paragraph explaining why you gave PASS or ACTION REQUIRED.
If PASS: what gives you confidence? What did you check?
If ACTION REQUIRED: what is the single most important thing to fix before merging?

---

Remember: your job is to find bugs, not to find things to say.
If the code is correct and simple, say so clearly and move on.
Do not invent issues. Do not flag style preferences as bugs.
Do not suggest adding abstraction, patterns, or frameworks unless they fix a concrete bug.`;

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

  // 4. Build prompt — include diff stats for richer LLM context
  const diffLines = diff.split("\n");
  const addedLines = diffLines.filter(
    (l) => l.startsWith("+") && !l.startsWith("+++")
  ).length;
  const removedLines = diffLines.filter(
    (l) => l.startsWith("-") && !l.startsWith("---")
  ).length;
  const changedFiles = [
    ...diff.matchAll(/^\+\+\+ b\/(.+)$/gm),
  ].map((m) => m[1]);

  const userContent = [
    `## Objective (what this change is supposed to achieve)`,
    args.objective,
    ``,
    `## Diff Statistics`,
    `- Files changed: ${changedFiles.length}${changedFiles.length ? ` (${changedFiles.join(", ")})` : ""}`,
    `- Lines added: +${addedLines}`,
    `- Lines removed: -${removedLines}`,
    args.targetBranch
      ? `- Comparing: ${args.targetBranch}...HEAD`
      : `- Comparing: uncommitted/staged changes vs HEAD`,
    ``,
    `## Full Git Diff`,
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
    temperature: 0.1, // lower = more deterministic, less hallucination on bug-finding
    max_tokens: 8192, // enough for thorough multi-file reviews
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
