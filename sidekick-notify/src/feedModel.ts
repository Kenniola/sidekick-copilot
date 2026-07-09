// feedModel.ts — pure, VS Code-free logic for the Sidekick Feed (Phase 3 / B5).
//
// Kept free of any `vscode` import so it can be unit-tested with `node --test`.
// The extension glue in extension.ts renders these entries in a TreeView and
// decides when to raise a toast.

export interface Alert {
  type: string;
  summary: string;
  answer?: string;
  answer_full?: string;
  source?: string;
  file?: string;
  confidence: string;
  priority: string;
  timestamp: string;
  id?: string;
  thread_id?: string;
  rationale?: string;
}

export interface FeedEntry {
  key: string;
  threadId: string;
  type: string;
  priority: string;
  confidence: string;
  headline: string;
  answer: string;
  answerFull: string;
  rationale: string;
  source: string;
  file: string;
  timestamp: string;
  seen: boolean;
  superseded: boolean;
}

/** A child row shown when a feed entry is expanded (Phase 5 / 5.2). */
export interface DetailNode {
  kind: "detail";
  label: string;
  value: string;
  url?: string;
  file?: string;
  chat?: boolean;
  question?: string;
}

// Short category tags shown beside the icon (Phase 5 / 5.1).
const TYPE_TAG: Record<string, string> = {
  research: "research",
  sizing: "sizing",
  roadmap: "roadmap",
  prototype: "proto",
  diagnostic: "diag",
  action_item: "action",
  suggestion: "ask",
  deliverables: "deliverable",
};

export function typeTag(type: string): string {
  return TYPE_TAG[type] ?? type;
}

/** Strip markdown so it reads cleanly in a plain-text TreeItem label (9.2). */
export function stripMarkdown(text: string): string {
  return (text || "")
    .replace(/\*\*(.*?)\*\*/g, "$1") // bold
    .replace(/`([^`]*)`/g, "$1") // inline code
    .replace(/^\s*#+\s*/gm, "") // headings
    .replace(/\*([^*]+)\*/g, "$1") // italic
    .replace(/_{1,2}([^_]+)_{1,2}/g, "$1") // underscore emphasis
    .replace(/\s+/g, " ")
    .trim();
}

/** Short uppercase confidence tag for the row description (9.2). */
export function confidenceTag(confidence: string): string {
  const c = (confidence || "").toLowerCase();
  if (c === "high") {
    return "HIGH";
  }
  if (c === "low") {
    return "LOW";
  }
  return "MED";
}

/** Build the expandable detail rows for a finding (pure — no vscode). */
export function detailNodes(entry: FeedEntry): DetailNode[] {
  const nodes: DetailNode[] = [];
  if (entry.answerFull) {
    nodes.push({ kind: "detail", label: "Answer", value: entry.answerFull });
  }
  if (entry.rationale) {
    nodes.push({ kind: "detail", label: "Why", value: entry.rationale });
  }
  if (entry.source) {
    nodes.push({ kind: "detail", label: "Source", value: entry.source, url: entry.source });
  }
  if (entry.file) {
    nodes.push({ kind: "detail", label: "File", value: entry.file, file: entry.file });
  }
  nodes.push({
    kind: "detail",
    label: "Meta",
    value: `${entry.priority} priority · ${entry.confidence} confidence`,
  });
  nodes.push({
    kind: "detail",
    label: "",
    value: "Research in Chat",
    chat: true,
    question: entry.headline,
  });
  return nodes;
}

const MAX_ENTRIES = 200;

/** Stable key for supersede/dedup — prefers the server-provided id. */
export function alertKey(alert: Alert): string {
  if (alert.id && alert.id.trim()) {
    return alert.id.trim();
  }
  return `${alert.type}:${(alert.summary || "").slice(0, 40)}`;
}

/** Toast floor (decision #4): only critical/high are worth interrupting for. */
export function isHighPriority(x: { priority?: string; confidence?: string }): boolean {
  return x.priority === "critical" || x.priority === "high" || x.confidence === "high";
}

export function headlineOf(alert: Alert): string {
  const answer = (alert.answer || "").trim();
  return answer || alert.summary || "(finding)";
}

/** The question a finding is answering — the feed row leads with this (8.5). */
export function questionOf(alert: Alert): string {
  return (alert.summary || "").trim() || headlineOf(alert);
}

export class FeedModel {
  entries: FeedEntry[] = [];

  /**
   * Add a new entry or update an existing one with the same key in place.
   * A newer entry on the same thread marks older thread entries superseded.
   * Returns whether the entry was genuinely new (used to dedup toasts).
   */
  addAlert(alert: Alert): { entry: FeedEntry; isNew: boolean } {
    const key = alertKey(alert);
    const entry: FeedEntry = {
      key,
      threadId: (alert.thread_id || "").trim(),
      type: alert.type,
      priority: alert.priority || "medium",
      confidence: alert.confidence || "medium",
      headline: questionOf(alert),
      answer: headlineOf(alert),
      answerFull: (alert.answer_full || alert.answer || "").trim(),
      rationale: (alert.rationale || "").trim(),
      source: alert.source || "",
      file: alert.file || "",
      timestamp: alert.timestamp || new Date().toISOString(),
      seen: false,
      superseded: false,
    };

    const existingIdx = this.entries.findIndex((e) => e.key === key);
    const isNew = existingIdx < 0;
    if (!isNew) {
      // Supersede in place: drop the stale copy, re-add at the front.
      this.entries.splice(existingIdx, 1);
    }
    this.entries.unshift(entry);

    // Thread-level supersede: a newer answer dims older ones on the same thread.
    if (entry.threadId) {
      for (const e of this.entries) {
        if (e !== entry && e.threadId === entry.threadId) {
          e.superseded = true;
        }
      }
    }

    if (this.entries.length > MAX_ENTRIES) {
      this.entries.length = MAX_ENTRIES;
    }
    return { entry, isNew };
  }

  /** Toast only critical/high — everything else lives silently in the feed. */
  shouldToast(alert: Alert): boolean {
    return isHighPriority(alert);
  }

  unseenHighCount(): number {
    return this.entries.filter(
      (e) => !e.seen && !e.superseded && isHighPriority(e),
    ).length;
  }

  markAllSeen(): void {
    for (const e of this.entries) {
      e.seen = true;
    }
  }

  /** Clear the feed — called when a new session starts (Phase 5 / 5.3). */
  clear(): void {
    this.entries = [];
  }

  /** True when an entry is older than ttlMs relative to nowMs (epoch ms). */
  isStale(entry: FeedEntry, nowMs: number, ttlMs: number): boolean {
    const t = Date.parse(entry.timestamp);
    if (Number.isNaN(t)) {
      return false;
    }
    return nowMs - t > ttlMs;
  }
}
