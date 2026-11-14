# Specs Index

## Navigation by Task

| If your task involves...                                       | Read these specs                                                                |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| sp4rk package layers, import rules                            | [architecture/layers.md](architecture/layers.md)                                |
| Request lifecycle inside the engine                           | [architecture/data-flow.md](architecture/data-flow.md)                          |
| Tool policies, confirmations, judge                           | [architecture/security-model.md](architecture/security-model.md)                |
| Tool capability groups, MCP group overrides, agent tool budgets | [contracts/tools.md](contracts/tools.md), [domains/tool-system/README.md](domains/tool-system/README.md), [domains/tool-system/mcp-gateway.md](domains/tool-system/mcp-gateway.md), [domains/agents.md](domains/agents.md) |
| Strict/session-aware ToolJudge evaluation                      | [architecture/security-model.md](architecture/security-model.md), [contracts/tools.md](contracts/tools.md), [domains/tool-system/README.md](domains/tool-system/README.md), [domains/oneshot.md](domains/oneshot.md) |
| Prompt-injection defense, untrusted-content wrapping          | [architecture/security-model.md](architecture/security-model.md)                |
| Orchestration overview (Conductor pipeline)                   | [domains/orchestration/README.md](domains/orchestration/README.md)              |
| Conductor (runs a task end-to-end)                            | [domains/orchestration/conductor.md](domains/orchestration/conductor.md)        |
| ReAct loop, circuit breakers, step limits                     | [domains/orchestration/executor.md](domains/orchestration/executor.md)          |
| Request classification, domain/complexity                     | [domains/orchestration/router.md](domains/orchestration/router.md)              |
| Plan generation, DAG, replan                                  | [domains/orchestration/planner.md](domains/orchestration/planner.md)            |
| Failure analysis, retry/replan/abort                          | [domains/orchestration/reflector.md](domains/orchestration/reflector.md)        |
| Subagent delegation, parallel steps                           | [domains/orchestration/subagents.md](domains/orchestration/subagents.md)        |
| Resuming execution from a checkpoint (ResumeSteps, SeedSteps) | [domains/orchestration/conductor.md](domains/orchestration/conductor.md), [domains/orchestration/executor.md](domains/orchestration/executor.md), [domains/memory/README.md](domains/memory/README.md) |
| Manual compaction of a resumed task (CompactOnStart)         | [domains/orchestration/conductor.md](domains/orchestration/conductor.md)                                        |
| Cooperative pause, live user messages, resume interjections    | [architecture/data-flow.md](architecture/data-flow.md), [domains/orchestration/conductor.md](domains/orchestration/conductor.md), [domains/orchestration/executor.md](domains/orchestration/executor.md), [domains/memory/README.md](domains/memory/README.md) |
| Post-edit verification after write_file/edit_file              | [contracts/agent-execution.md](contracts/agent-execution.md), [domains/orchestration/executor.md](domains/orchestration/executor.md), [domains/orchestration/conductor.md](domains/orchestration/conductor.md) |
| Tool registry, execution pipeline, pre-dispatch input validation | [domains/tool-system/README.md](domains/tool-system/README.md)              |
| Structural tool-input validation (ValidateToolInput): closed-set schemas, fail-open rules, disable handle | [domains/tool-system/README.md](domains/tool-system/README.md) |
| Tool-result caching, file-backed vs content-backed modes      | [domains/orchestration/executor.md](domains/orchestration/executor.md), [domains/tool-system/README.md](domains/tool-system/README.md) |
| Adding/modifying built-in tools                               | [domains/tool-system/builtins.md](domains/tool-system/builtins.md)              |
| Web search tool, provider backends (Brave, DuckDuckGo, Exa, SearXNG, Tavily) | [domains/tool-system/websearch.md](domains/tool-system/websearch.md), [domains/tool-system/builtins.md](domains/tool-system/builtins.md) |
| Safe file reads/writes (regular-files-only, size caps, atomic publish) | [domains/safeio.md](domains/safeio.md)                                          |
| Append-safe compressed logging (appendcodec: gzip/zstd, xz rejected) | [decisions/010-appendcodec-gzip-zstd-xz-rejected.md](decisions/010-appendcodec-gzip-zstd-xz-rejected.md) |
| UTF-8 string helpers (truncation, invisible content)           | [domains/strutil.md](domains/strutil.md)                                        |
| Ignore rules (.gitignore/.aiignore), glob & ripgrep filtering | [architecture/layers.md](architecture/layers.md), [contracts/tools.md](contracts/tools.md), [domains/tool-system/builtins.md](domains/tool-system/builtins.md) |
| MCP servers, dynamic tools                                    | [domains/tool-system/mcp-gateway.md](domains/tool-system/mcp-gateway.md)        |
| Context window, compaction strategies                         | [domains/memory/compaction.md](domains/memory/compaction.md)                    |
| Blackboard, shared state, facts, attachments                  | [domains/memory/blackboard.md](domains/memory/blackboard.md)                    |
| LLM providers, router, model registry, tokens                 | [domains/llm-providers.md](domains/llm-providers.md)                            |
| LLM response streaming (opt-in delta delivery)                | [domains/llm-providers.md](domains/llm-providers.md), [contracts/llm-providers.md](contracts/llm-providers.md), [domains/orchestration/executor.md](domains/orchestration/executor.md), [decisions/009-opt-in-request-scoped-streaming.md](decisions/009-opt-in-request-scoped-streaming.md) |
| One-shot structured LLM calls, the parse-repair loop, failure policies | [domains/oneshot.md](domains/oneshot.md), [decisions/008-unified-oneshot-service-client.md](decisions/008-unified-oneshot-service-client.md) |
| Adding a new structured service call (router/planner/reflector/judge-style) | [domains/oneshot.md](domains/oneshot.md), [contracts/llm-providers.md](contracts/llm-providers.md) |
| Purpose-aware sampling and provider parameter filtering       | [contracts/llm-providers.md](contracts/llm-providers.md), [domains/llm-providers.md](domains/llm-providers.md), [domains/prompt-building.md](domains/prompt-building.md) |
| Multi-protocol routing (Responses/Anthropic/Google/Chat), multimodal content blocks | [domains/llm-providers.md](domains/llm-providers.md)                            |
| Skill system, activation, resources                           | [domains/skills.md](domains/skills.md)                                          |
| Subagent Profiles, AGENT.md discovery, per-agent model/tools  | [domains/agents.md](domains/agents.md)                                          |
| Prompt builder, system-prompt composition                     | [domains/prompt-building.md](domains/prompt-building.md)                        |
| Embeddings, chunking, ONNX                                    | [domains/embedding.md](domains/embedding.md)                                    |
| GPU/CUDA inference, execution providers, GPU diagnostics      | [domains/embedding.md](domains/embedding.md), [decisions/011-runtime-hardware-cpu-degradation.md](decisions/011-runtime-hardware-cpu-degradation.md) |
| What an embedding application must provide (events, confirm)  | [contracts/agent-execution.md](contracts/agent-execution.md)                    |
| LLM provider contract                                         | [contracts/llm-providers.md](contracts/llm-providers.md)                        |
| Tool interface contract                                       | [contracts/tools.md](contracts/tools.md)                                        |
| "Why was X designed this way?"                                | [decisions/](decisions/)                                                        |

## Package Dependency Graph

sp4rk is a single Go module (`github.com/v0lka/sp4rk`). Arrows show import direction (`A → B` means package A imports package B). Import direction flows downward — upper orchestration layers depend on lower primitive layers, never the reverse.

```
Root — github.com/v0lka/sp4rk (Framework, Config, Execute, fluent builders) owns the
shared infrastructure (LLM router, tool registry, MCP gateway, tool cache), creates
per-task conductors, and imports every engine layer:

    root → agent, agent/reflector, llm, memory, orchestration, planner, skills,
           tools, tools/builtins, tools/mcp

Upper layers (import direction flows downward, never the reverse):

    orchestration   → agent, appendcodec, llm, safeio, skills, strutil, tools
    planner         → agent, llm, oneshot, orchestration, prompt, skills, strutil, tools
    memory          → agent, llm, prompt, security, strutil
    agent           → llm, strutil, tools
    agent/reflector → agent, llm, oneshot, orchestration, prompt, security, tools
    agent/router    → agent, llm, oneshot, prompt, tools

Tool layer:

    tools                    → llm, oneshot, pathutil, security, strutil, sysproc,
                               tools/internal/judge_prompts
    tools/builtins           → agent, pathutil, safeio, sysproc, tools
    tools/builtins/websearch → tools, tools/builtins
    tools/mcp                → sysproc, tools

Supporting / leaf packages (no engine imports beyond what is listed):

    oneshot   → llm                (one-shot structured-call client; consumed by planner,
                                    agent/router, agent/reflector, and the tools judges)
    appendcodec → safeio           (stdlib compress/gzip + klauspost/compress zstd otherwise;
                                    consumed by orchestration's StepDumpTracker and by host
                                    rotation wrappers)
    skills    → pathutil, safeio, tools
    agents    → safeio             (stdlib + yaml.v3 + log/slog otherwise; no coupling to
                                    tools/agent/orchestration)
    ignore    → pathutil, safeio   (external doublestar only otherwise)
    embedding → sysproc
    safeio    → pathutil
    llm       → strutil
    prompt, security, pathutil, strutil, sysproc, tools/internal/judge_prompts
              → nothing engine-internal (sysproc additionally uses golang.org/x/sys/windows
                behind a //go:build windows tag for process-tree containment; stdlib and
                a no-op on non-Windows)
```

Supporting packages — `prompt`, `oneshot`, `skills`, `agents`, `security`, `embedding`, `pathutil`, `strutil`, `sysproc`, `ignore`, `safeio`, `appendcodec` — are consumed across layers as needed and have no upward dependencies.

Import rule: every arrow is one-way. `tools` and `llm` are near-leaf dependencies; they never import `agent`, `orchestration`, `planner`, or the root package. This keeps the primitive layers free of higher-level concerns and prevents import cycles.

## Spec Workflow and Format Reference

See [META.md](META.md) for document templates, naming rules, and update protocol.

## Directory Listing

### architecture/

- [layers.md](architecture/layers.md) - sp4rk package layers, import rules, responsibilities
- [data-flow.md](architecture/data-flow.md) - Request lifecycle, ReAct loop, Plan&Execute, event flow
- [security-model.md](architecture/security-model.md) - Tool policies, judge, confirmations, prompt-injection defense

### domains/orchestration/

- [README.md](domains/orchestration/README.md) - Orchestration domain overview (Conductor pipeline)
- [conductor.md](domains/orchestration/conductor.md) - Conductor that runs a single task end-to-end
- [executor.md](domains/orchestration/executor.md) - Executor ReAct loop primitive (shared by Conductor and subagents)
- [router.md](domains/orchestration/router.md) - Request classification, skill matching, and optional semantic tool matching
- [planner.md](domains/orchestration/planner.md) - Plan generation, DAG, replan
- [reflector.md](domains/orchestration/reflector.md) - Failure analysis and retry/replan/abort
- [subagents.md](domains/orchestration/subagents.md) - Subagent delegation and parallel step execution

### domains/tool-system/

- [README.md](domains/tool-system/README.md) - Tool registry, execution pipeline, and pre-dispatch input validation
- [builtins.md](domains/tool-system/builtins.md) - Built-in tool catalog and extension guide
- [mcp-gateway.md](domains/tool-system/mcp-gateway.md) - MCP server lifecycle and dynamic tools
- [websearch.md](domains/tool-system/websearch.md) - Web search tool and provider backends (Brave, DuckDuckGo, Exa, SearXNG, Tavily)

### domains/memory/

- [README.md](domains/memory/README.md) - Context management overview
- [compaction.md](domains/memory/compaction.md) - Compaction strategies and thresholds
- [blackboard.md](domains/memory/blackboard.md) - Shared task state, facts, checkpoints

### domains/ (single-file)

- [llm-providers.md](domains/llm-providers.md) - Provider abstraction, router, model registry, token counting
- [oneshot.md](domains/oneshot.md) - One-shot structured-output LLM client: the Do loop, failure policies, parser toolkit, call-site catalog
- [skills.md](domains/skills.md) - Skill system, activation, resource access
- [agents.md](domains/agents.md) - Subagent Profiles (AGENT.md discovery, parsing, management)
- [prompt-building.md](domains/prompt-building.md) - Prompt builder and system-prompt composition
- [embedding.md](domains/embedding.md) - Embeddings, chunking, ONNX runtime
- [safeio.md](domains/safeio.md) - Safe file-read/write primitives (regular-files-only reads, size caps, atomic publish)
- [strutil.md](domains/strutil.md) - UTF-8-safe string helpers (truncation, invisible content)

### contracts/

- [agent-execution.md](contracts/agent-execution.md) - What an embedding application provides (events, confirmations, HITL)
- [llm-providers.md](contracts/llm-providers.md) - LLM provider interface contract
- [tools.md](contracts/tools.md) - Tool and ToolJudger interface contract

### decisions/

- [_template.md](decisions/_template.md) - ADR template
- [001-separate-module.md](decisions/001-separate-module.md) - sp4rk as a separate Go module from any host application
- [002-skills-mcp-in-sdk.md](decisions/002-skills-mcp-in-sdk.md) - Skills and MCP gateway live in sp4rk
- [003-shell-parser-symlink-detection.md](decisions/003-shell-parser-symlink-detection.md) - mvdan.cc/sh/v3/syntax for symlink-traversal detection
- [004-application-concept-extraction.md](decisions/004-application-concept-extraction.md) - sp4rk sheds application-specific concepts
- [005-conductor-orchestration-pipeline.md](decisions/005-conductor-orchestration-pipeline.md) - Conductor-driven orchestration pipeline
- [006-symlink-walk-literal-paths-only.md](decisions/006-symlink-walk-literal-paths-only.md) - The symlink walk is a literal-path extractor
- [007-c10-exec-scope-criterion.md](decisions/007-c10-exec-scope-criterion.md) - The C10 exec-scope criterion and digest v4
- [008-unified-oneshot-service-client.md](decisions/008-unified-oneshot-service-client.md) - Unified one-shot service client
- [009-opt-in-request-scoped-streaming.md](decisions/009-opt-in-request-scoped-streaming.md) - Opt-in request-scoped streaming channel
- [010-appendcodec-gzip-zstd-xz-rejected.md](decisions/010-appendcodec-gzip-zstd-xz-rejected.md) - Append-safe codec writers: gzip and zstd, xz rejected
- [011-runtime-hardware-cpu-degradation.md](decisions/011-runtime-hardware-cpu-degradation.md) - One-shot auto hardware→CPU degradation after consecutive inference failures
- [012-per-request-provider-override.md](decisions/012-per-request-provider-override.md) - Per-request provider override on ChatRequest
