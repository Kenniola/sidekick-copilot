import * as vscode from "vscode";
import * as fs from "fs";
import * as path from "path";
import * as os from "os";
import { FeedModel, FeedEntry, DetailNode, Alert, isHighPriority, typeTag, detailNodes, stripMarkdown, confidenceTag } from "./feedModel";

const ALERTS_PATH = path.join(
  os.homedir(),
  ".sidekick",
  "live",
  "alerts.jsonl"
);
const POLL_MS = 2000;
const STALE_MS = 10 * 60 * 1000; // findings older than 10 min render dimmed

let pollTimer: ReturnType<typeof setInterval> | undefined;
let lastSize = 0;
let statusBar: vscode.StatusBarItem;
const model = new FeedModel();
let feedProvider: SidekickFeedProvider;

// ── Feed TreeView (B2 + Phase 5 drill-down) ────────────────────────────

type FeedNode = FeedEntry | DetailNode;

function isDetail(node: FeedNode): node is DetailNode {
  return (node as DetailNode).kind === "detail";
}

const TYPE_ICON: Record<string, string> = {
  research: "search",
  prototype: "tools",
  roadmap: "map",
  sizing: "graph",
  diagnostic: "pulse",
  action_item: "checklist",
  suggestion: "lightbulb",
  deliverables: "package",
};

class SidekickFeedProvider implements vscode.TreeDataProvider<FeedNode> {
  private _onDidChange = new vscode.EventEmitter<void>();
  readonly onDidChangeTreeData = this._onDidChange.event;

  refresh(): void {
    this._onDidChange.fire();
  }

  getTreeItem(node: FeedNode): vscode.TreeItem {
    return isDetail(node) ? this.detailItem(node) : this.entryItem(node);
  }

  private entryItem(entry: FeedEntry): vscode.TreeItem {
    // Numbered (9.2) + category tag (5.1) + markdown-stripped question, on an
    // expandable row (5.2). Plain labels can't render markdown, so we clean it.
    const n = model.entries.indexOf(entry) + 1;
    const item = new vscode.TreeItem(
      `${n}. [${typeTag(entry.type)}] ${stripMarkdown(entry.headline)}`,
      vscode.TreeItemCollapsibleState.Collapsed
    );
    const stale = model.isStale(entry, Date.now(), STALE_MS);
    const highlight = isHighPriority(entry) && !entry.seen && !entry.superseded;
    // Colour the icon by confidence (green/amber/red); an unseen high-priority
    // finding keeps the orange attention colour.
    item.iconPath = new vscode.ThemeIcon(
      TYPE_ICON[entry.type] ?? "note",
      highlight
        ? new vscode.ThemeColor("charts.orange")
        : confidenceColor(entry.confidence)
    );
    const rel = relativeTime(entry.timestamp);
    const bits = [confidenceTag(entry.confidence), rel];
    if (entry.superseded) {
      bits.push("superseded");
    } else if (stale) {
      bits.push("stale");
    }
    item.description = bits.join(" · ");
    item.contextValue = "sidekickFinding";

    const md = new vscode.MarkdownString();
    md.appendMarkdown(`**[${typeTag(entry.type)}]** ${entry.headline}\n\n`);
    if (entry.rationale) {
      md.appendMarkdown(`_${entry.rationale}_\n\n`);
    }
    md.appendMarkdown(`priority: ${entry.priority} · ${entry.confidence} · ${rel}`);
    item.tooltip = md;
    // No command on the row: clicking expands it to reveal details (5.2 / 2c).
    return item;
  }

  private detailItem(node: DetailNode): vscode.TreeItem {
    const clean = stripMarkdown(node.value);
    const shortVal = clean.length > 90 ? clean.slice(0, 90) + "\u2026" : clean;
    const label = node.label ? `${node.label}: ${shortVal}` : shortVal;
    const item = new vscode.TreeItem(label, vscode.TreeItemCollapsibleState.None);
    // Full value (may be a multi-paragraph answer) shown on hover as markdown.
    item.tooltip = new vscode.MarkdownString(node.value);
    if (node.url && /^https?:\/\//.test(node.url)) {
      item.iconPath = new vscode.ThemeIcon("link-external");
      item.command = {
        command: "vscode.open",
        title: "Open Source",
        arguments: [vscode.Uri.parse(node.url)],
      };
    } else if (node.file) {
      item.iconPath = new vscode.ThemeIcon("file");
      item.command = {
        command: "vscode.open",
        title: "Open File",
        arguments: [vscode.Uri.file(node.file)],
      };
    } else if (node.chat) {
      item.iconPath = new vscode.ThemeIcon("comment-discussion");
      item.command = {
        command: "sidekick-notify.researchInChat",
        title: "Research in Chat",
        arguments: [node],
      };
    } else {
      item.iconPath = new vscode.ThemeIcon("info");
    }
    return item;
  }

  getChildren(node?: FeedNode): FeedNode[] {
    if (!node) {
      return model.entries;
    }
    if (isDetail(node)) {
      return [];
    }
    return detailNodes(node);
  }
}

// ── Lifecycle ──────────────────────────────────────────────────────────

export function activate(ctx: vscode.ExtensionContext): void {
  // Skip old alerts — start from the current file end.
  try {
    lastSize = fs.statSync(ALERTS_PATH).size;
  } catch {
    lastSize = 0;
  }

  statusBar = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Right,
    200
  );
  statusBar.command = "sidekick-notify.openFeed";
  setIdle();
  statusBar.show();
  ctx.subscriptions.push(statusBar);

  feedProvider = new SidekickFeedProvider();
  ctx.subscriptions.push(
    vscode.window.createTreeView("sidekickFeed", {
      treeDataProvider: feedProvider,
    })
  );

  ctx.subscriptions.push(
    vscode.commands.registerCommand("sidekick-notify.showStatus", () => {
      model.markAllSeen();
      feedProvider.refresh();
      updateBadge();
      vscode.commands.executeCommand("workbench.action.chat.open", {
        query: "@sidekick status",
        isPartialQuery: false,
      });
    }),
    vscode.commands.registerCommand("sidekick-notify.openFeed", () => {
      model.markAllSeen();
      feedProvider.refresh();
      updateBadge();
      vscode.commands.executeCommand("sidekickFeed.focus");
    }),
    vscode.commands.registerCommand("sidekick-notify.clearSeen", () => {
      model.markAllSeen();
      feedProvider.refresh();
      updateBadge();
    }),
    vscode.commands.registerCommand(
      "sidekick-notify.researchInChat",
      (node?: DetailNode) => {
        // Clean to a single plain-text line — markdown/newlines in the chat
        // query box behave badly. There is no "@sidekick" chat participant
        // (Sidekick is an MCP server), so send a plain instruction the agent
        // can act on with the Sidekick research tool it already has.
        const q = stripMarkdown(node?.question || "").slice(0, 300);
        vscode.commands.executeCommand("workbench.action.chat.open", {
          query: q
            ? `Research this question and answer with sources: ${q}`
            : "Show me the latest Sidekick findings.",
          isPartialQuery: false,
        });
      }
    )
  );

  pollTimer = setInterval(poll, POLL_MS);
  ctx.subscriptions.push({
    dispose: () => {
      if (pollTimer) {
        clearInterval(pollTimer);
      }
    },
  });
}

export function deactivate(): void {
  if (pollTimer) {
    clearInterval(pollTimer);
  }
}

// ── File polling ───────────────────────────────────────────────────────

function poll(): void {
  let stat: fs.Stats;
  try {
    stat = fs.statSync(ALERTS_PATH);
  } catch {
    return; // file doesn't exist yet — Sidekick hasn't started
  }

  if (stat.size < lastSize) {
    // File truncated / rotated (new session) — reset and clear the old feed.
    lastSize = 0;
    model.clear();
    feedProvider.refresh();
    updateBadge();
  }
  if (stat.size <= lastSize) {
    return; // no new data
  }

  const stream = fs.createReadStream(ALERTS_PATH, {
    start: lastSize,
    encoding: "utf-8",
  });

  let buf = "";
  stream.on("data", (chunk) => {
    buf += String(chunk);
  });
  stream.on("end", () => {
    lastSize = stat.size;
    let touched = false;
    for (const line of buf.split("\n")) {
      if (!line.trim()) {
        continue;
      }
      try {
        handleAlert(JSON.parse(line) as Alert);
        touched = true;
      } catch {
        // skip malformed lines
      }
    }
    if (touched) {
      feedProvider.refresh();
      updateBadge();
    }
  });
}

// ── Alert handling ─────────────────────────────────────────────────────

function handleAlert(alert: Alert): void {
  const { isNew } = model.addAlert(alert);
  // Gate toasts (B1): only critical/high, and only for genuinely new findings
  // (an update to an existing id refreshes the feed row without re-toasting).
  if (model.shouldToast(alert) && isNew) {
    showToast(alert);
  }
}

function showToast(alert: Alert): void {
  const icon: Record<string, string> = {
    research: "🔍",
    prototype: "🛠️",
    roadmap: "🗺️",
    suggestion: "💡",
    deliverables: "📦",
  };
  const emoji = icon[alert.type] ?? "📋";
  const headline = (alert.answer && alert.answer.trim()) || alert.summary;
  const msg = `${emoji} Sidekick: ${headline}`;

  const hasSource = !!(alert.source && /^https?:\/\//.test(alert.source));
  const hasFile = !!(alert.file && alert.file.trim());
  const actions: string[] = [];
  if (hasSource) {
    actions.push("Open Source");
  }
  if (hasFile) {
    actions.push("Open File");
  }
  actions.push("Open Feed");

  vscode.window.showWarningMessage(msg, ...actions).then((choice) => {
    if (choice === "Open Source" && alert.source) {
      vscode.env.openExternal(vscode.Uri.parse(alert.source));
    } else if (choice === "Open File" && alert.file) {
      vscode.commands.executeCommand("vscode.open", vscode.Uri.file(alert.file));
    } else if (choice === "Open Feed") {
      vscode.commands.executeCommand("sidekick-notify.openFeed");
    }
  });
}

// ── Status bar ─────────────────────────────────────────────────────────

function updateBadge(): void {
  const n = model.unseenHighCount();
  if (n > 0) {
    statusBar.text = `$(megaphone) Sidekick (${n})`;
    statusBar.backgroundColor = new vscode.ThemeColor(
      "statusBarItem.warningBackground"
    );
    statusBar.tooltip = `${n} unseen high-priority finding(s) — click to open the feed`;
  } else {
    setIdle();
  }
}

function setIdle(): void {
  statusBar.text = "$(megaphone) Sidekick";
  statusBar.backgroundColor = undefined;
  statusBar.tooltip = "Sidekick — click to open the feed";
}

function confidenceColor(confidence: string): vscode.ThemeColor | undefined {
  const c = (confidence || "").toLowerCase();
  if (c === "high") {
    return new vscode.ThemeColor("charts.green");
  }
  if (c === "low") {
    return new vscode.ThemeColor("charts.red");
  }
  return new vscode.ThemeColor("charts.yellow");
}

function relativeTime(iso: string): string {
  const t = Date.parse(iso);
  if (Number.isNaN(t)) {
    return "";
  }
  const s = Math.max(0, Math.floor((Date.now() - t) / 1000));
  if (s < 60) {
    return `${s}s ago`;
  }
  const m = Math.floor(s / 60);
  if (m < 60) {
    return `${m}m ago`;
  }
  const h = Math.floor(m / 60);
  return `${h}h ago`;
}
