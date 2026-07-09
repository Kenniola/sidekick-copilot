// feedModel.test.ts — unit tests for the pure feed logic (Phase 3 / B5).
// Run with: npm test  (tsc then `node --test out/feedModel.test.js`).

import { test } from "node:test";
import assert from "node:assert";
import {
  FeedModel,
  alertKey,
  isHighPriority,
  headlineOf,
  questionOf,
  typeTag,
  detailNodes,
  stripMarkdown,
  confidenceTag,
  Alert,
} from "./feedModel";

function mkAlert(over: Partial<Alert> = {}): Alert {
  return {
    type: "research",
    summary: "What is F64?",
    confidence: "medium",
    priority: "medium",
    timestamp: new Date().toISOString(),
    ...over,
  };
}

test("alertKey prefers server id, else derives from type+summary", () => {
  assert.strictEqual(alertKey(mkAlert({ id: "research:abc" })), "research:abc");
  assert.strictEqual(
    alertKey(mkAlert({ id: "", summary: "hello world" })),
    "research:hello world",
  );
});

test("isHighPriority matches critical/high priority or high confidence", () => {
  assert.ok(isHighPriority({ priority: "high" }));
  assert.ok(isHighPriority({ priority: "critical" }));
  assert.ok(isHighPriority({ confidence: "high" }));
  assert.ok(!isHighPriority({ priority: "medium", confidence: "medium" }));
});

test("headlineOf prefers the answer, falls back to summary", () => {
  assert.strictEqual(headlineOf(mkAlert({ answer: "42" })), "42");
  assert.strictEqual(headlineOf(mkAlert({ answer: "" })), "What is F64?");
});

test("addAlert prepends new entries, newest first", () => {
  const m = new FeedModel();
  m.addAlert(mkAlert({ id: "a", summary: "A" }));
  const r = m.addAlert(mkAlert({ id: "b", summary: "B" }));
  assert.strictEqual(r.isNew, true);
  assert.strictEqual(m.entries.length, 2);
  assert.strictEqual(m.entries[0].key, "b");
});

test("addAlert supersedes same-key entry in place (dedup)", () => {
  const m = new FeedModel();
  m.addAlert(mkAlert({ id: "a", answer: "first" }));
  const r = m.addAlert(mkAlert({ id: "a", answer: "second" }));
  assert.strictEqual(r.isNew, false);
  assert.strictEqual(m.entries.length, 1);
  assert.strictEqual(m.entries[0].answer, "second");
});

test("feed row leads with the question, keeps the full answer", () => {
  const m = new FeedModel();
  const { entry } = m.addAlert(
    mkAlert({
      summary: "How do I size F64?",
      answer: "Start at F64...",
      answer_full: "Start at F64 and monitor CU. Sources: ...",
    }),
  );
  assert.strictEqual(entry.headline, "How do I size F64?");
  assert.strictEqual(entry.answer, "Start at F64...");
  assert.ok(entry.answerFull.startsWith("Start at F64 and monitor"));
});

test("questionOf prefers summary, falls back to answer", () => {
  assert.strictEqual(questionOf(mkAlert({ summary: "Q?" })), "Q?");
  assert.strictEqual(
    questionOf(mkAlert({ summary: "", answer: "A" })),
    "A",
  );
});

test("newer thread entry supersedes older ones on the same thread", () => {
  const m = new FeedModel();
  m.addAlert(mkAlert({ id: "a", thread_id: "t1" }));
  m.addAlert(mkAlert({ id: "b", thread_id: "t1" }));
  const older = m.entries.find((e) => e.key === "a");
  const newer = m.entries.find((e) => e.key === "b");
  assert.strictEqual(older?.superseded, true);
  assert.strictEqual(newer?.superseded, false);
});

test("shouldToast only for critical/high", () => {
  const m = new FeedModel();
  assert.ok(m.shouldToast(mkAlert({ priority: "high" })));
  assert.ok(m.shouldToast(mkAlert({ priority: "critical" })));
  assert.ok(!m.shouldToast(mkAlert({ priority: "medium" })));
  assert.ok(!m.shouldToast(mkAlert({ priority: "low" })));
});

test("unseenHighCount counts only unseen, non-superseded high entries", () => {
  const m = new FeedModel();
  m.addAlert(mkAlert({ id: "a", priority: "high" }));
  m.addAlert(mkAlert({ id: "b", priority: "medium" }));
  m.addAlert(mkAlert({ id: "c", priority: "critical" }));
  assert.strictEqual(m.unseenHighCount(), 2);
  m.markAllSeen();
  assert.strictEqual(m.unseenHighCount(), 0);
});

test("superseded high entries are not counted as unseen", () => {
  const m = new FeedModel();
  m.addAlert(mkAlert({ id: "a", priority: "high", thread_id: "t1" }));
  m.addAlert(mkAlert({ id: "b", priority: "high", thread_id: "t1" }));
  // "a" is superseded by "b"; only "b" counts.
  assert.strictEqual(m.unseenHighCount(), 1);
});

test("isStale compares timestamp against ttl", () => {
  const m = new FeedModel();
  const now = Date.parse("2026-07-07T12:00:00Z");
  const fresh = mkAlert({ timestamp: "2026-07-07T11:59:00Z" }); // 1 min old
  const old = mkAlert({ timestamp: "2026-07-07T11:40:00Z" }); // 20 min old
  const f = m.addAlert(fresh).entry;
  const o = m.addAlert(old).entry;
  const ttl = 10 * 60 * 1000;
  assert.strictEqual(m.isStale(f, now, ttl), false);
  assert.strictEqual(m.isStale(o, now, ttl), true);
});

test("entries are capped to avoid unbounded growth", () => {
  const m = new FeedModel();
  for (let i = 0; i < 250; i++) {
    m.addAlert(mkAlert({ id: `k${i}` }));
  }
  assert.ok(m.entries.length <= 200);
});

test("typeTag maps known types, falls back to the raw type", () => {
  assert.strictEqual(typeTag("research"), "research");
  assert.strictEqual(typeTag("action_item"), "action");
  assert.strictEqual(typeTag("prototype"), "proto");
  assert.strictEqual(typeTag("mystery"), "mystery");
});

test("detailNodes includes present fields and a chat action", () => {
  const m = new FeedModel();
  const { entry } = m.addAlert(
    mkAlert({
      summary: "How do I size F64?",
      answer: "Start at F64.",
      answer_full: "Start at F64 and monitor CU.",
      rationale: "why it matters",
      source: "https://x",
      file: "C:/a.md",
      priority: "high",
      confidence: "high",
    }),
  );
  const nodes = detailNodes(entry);
  assert.ok(nodes.some((n) => n.label === "Answer" && n.value.includes("monitor CU")));
  assert.ok(nodes.some((n) => n.label === "Why"));
  assert.ok(nodes.some((n) => n.url === "https://x"));
  assert.ok(nodes.some((n) => n.file === "C:/a.md"));
  assert.ok(nodes.some((n) => n.chat === true && n.question === "How do I size F64?"));
});

test("detailNodes omits absent optional fields", () => {
  const m = new FeedModel();
  const { entry } = m.addAlert(mkAlert({}));
  const nodes = detailNodes(entry);
  assert.ok(!nodes.some((n) => n.label === "Why"));
  assert.ok(!nodes.some((n) => n.url));
  assert.ok(!nodes.some((n) => n.file));
});

test("clear empties the feed", () => {
  const m = new FeedModel();
  m.addAlert(mkAlert({ id: "a" }));
  m.addAlert(mkAlert({ id: "b" }));
  m.clear();
  assert.strictEqual(m.entries.length, 0);
});

test("stripMarkdown removes bold/italic/code/headings", () => {
  assert.strictEqual(
    stripMarkdown("**Direct answer:** use `F64`"),
    "Direct answer: use F64",
  );
  assert.strictEqual(stripMarkdown("## Heading\n_emph_"), "Heading emph");
});

test("confidenceTag maps confidence levels", () => {
  assert.strictEqual(confidenceTag("high"), "HIGH");
  assert.strictEqual(confidenceTag("low"), "LOW");
  assert.strictEqual(confidenceTag("medium"), "MED");
  assert.strictEqual(confidenceTag(""), "MED");
});
