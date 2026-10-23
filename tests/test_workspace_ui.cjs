"use strict";

// Dependency-free interaction tests. The DOM double exercises application
// state and API contracts, not browser layout or native dialog rendering.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const web = path.join(__dirname, "../analyst/web");

class Element {
    constructor(tag, document) {
        this.tagName = tag.toLowerCase();
        this.document = document;
        this.children = [];
        this.parentElement = null;
        this.attributes = new Map();
        this.dataset = {};
        this.style = {};
        this.listeners = new Map();
        this.className = "";
        this.value = "";
        this.hidden = false;
        this.open = false;
        this.disabled = false;
        this.checked = false;
        this._text = "";
        this.tabIndex = ["button", "input", "textarea", "summary"].includes(tag) ? 0 : -1;
        this.classList = {
            contains: (value) => this.className.split(/\s+/).includes(value),
            toggle: (value, enabled) => {
                const classes = new Set(this.className.split(/\s+/).filter(Boolean));
                const add = enabled === undefined ? !classes.has(value) : enabled;
                if (add) classes.add(value); else classes.delete(value);
                this.className = Array.from(classes).join(" ");
                return add;
            },
            remove: (...values) => values.forEach((value) => this.classList.toggle(value, false))
        };
    }
    set textContent(value) { this._text = String(value); this.replaceChildren(); }
    get textContent() { return this._text + this.children.map((child) => child.textContent).join(""); }
    set innerHTML(value) { throw new Error("UI must not interpret untrusted HTML: " + value); }
    get childElementCount() { return this.children.length; }
    append(...children) {
        for (const child of children) {
            child.remove();
            child.parentElement = this;
            this.children.push(child);
        }
    }
    appendChild(child) { this.append(child); return child; }
    replaceChildren(...children) {
        for (const child of this.children) child.parentElement = null;
        this.children = [];
        this.append(...children);
    }
    remove() {
        if (this.parentElement) {
            const siblings = this.parentElement.children;
            siblings.splice(siblings.indexOf(this), 1);
            this.parentElement = null;
        }
    }
    setAttribute(name, value) {
        this.attributes.set(name, String(value));
        if (name === "class") this.className = value;
        if (name === "id") this.document.ids.set(value, this);
        if (name === "hidden") this.hidden = true;
        if (name === "disabled") this.disabled = true;
        if (name === "name") this.name = value;
    }
    getAttribute(name) { return this.attributes.get(name) ?? null; }
    removeAttribute(name) { this.attributes.delete(name); }
    matches(selector) {
        if (selector.includes(",")) return selector.split(",").some((item) => this.matches(item.trim()));
        if (selector === ".context-actions > summary") {
            return this.tagName === "summary" && this.parentElement?.classList.contains("context-actions");
        }
        if (selector === ".context-actions[open]") return this.classList.contains("context-actions") && this.open;
        if (selector === 'input[name="remove_skill"]:checked') {
            return this.tagName === "input" && this.name === "remove_skill" && this.checked;
        }
        if (selector.endsWith(":not([disabled])")) {
            return this.tagName === selector.split(":")[0] && !this.disabled;
        }
        if (selector.startsWith(".")) return this.classList.contains(selector.slice(1));
        return this.tagName === selector;
    }
    querySelectorAll(selector) {
        const result = [];
        for (const child of this.children) {
            if (child.matches(selector)) result.push(child);
            result.push(...child.querySelectorAll(selector));
        }
        return result;
    }
    querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
    closest(selector) { return this.matches(selector) ? this : this.parentElement?.closest(selector) || null; }
    contains(element) { return element === this || this.children.some((child) => child.contains(element)); }
    addEventListener(type, listener) {
        if (!this.listeners.has(type)) this.listeners.set(type, []);
        this.listeners.get(type).push(listener);
    }
    async dispatch(type, extra = {}) {
        const event = { target: this, preventDefault() { this.defaultPrevented = true; }, ...extra };
        for (const listener of this.listeners.get(type) || []) await listener(event);
        return event;
    }
    async click() { if (!this.disabled) await this.dispatch("click"); }
    requestSubmit() { return this.dispatch("submit"); }
    focus() { this.document.activeElement = this; }
    select() {}
    showModal() { this.open = true; }
    close() { this.open = false; }
    getClientRects() {
        if (this.hidden) return [];
        if (this.parentElement?.tagName === "details" && !this.parentElement.open && this.tagName !== "summary") return [];
        return this.parentElement ? this.parentElement.getClientRects() : [{}];
    }
}

const A = "a".repeat(32);
const B = "b".repeat(32);
const C = "c".repeat(32);
function record(id, title) {
    return { id, title, scope: "local-operator", created_at: "2026-10-20T12:00:00+00:00", updated_at: "2026-10-20T12:00:00+00:00" };
}
function measuredSummary(id, exchanges = 1, contextSize = 1000, modelName = "test-model") {
    return {
        conversation_id: id, exchange_count: exchanges, recorded_exchanges: exchanges,
        input_tokens: 100 * exchanges, output_tokens: 25 * exchanges, total_tokens: 125 * exchanges,
        input_tokens_measured_exchanges: exchanges, output_tokens_measured_exchanges: exchanges,
        total_tokens_measured_exchanges: exchanges, is_partial: false,
        latest_usage: exchanges ? {
            input_tokens: 100, output_tokens: 25, total_tokens: 125,
            context_size: contextSize, model_name: modelName
        } : null
    };
}
function response(data, status = 200) { return { ok: status < 400, status, json: async () => data }; }
async function flush() { for (let n = 0; n < 4; n++) await new Promise(setImmediate); }

async function harness({ narrow = false, disabled = false, incognito = false, records = [record(A, "Blue Lantern"), record(B, "Red Maple")] } = {}) {
    const document = new Element("document", null);
    document.document = document;
    document.ids = new Map();
    document.createElement = (tag) => new Element(tag, document);
    document.getElementById = (id) => {
        const element = document.ids.get(id);
        assert.ok(element, "HTML is missing #" + id);
        return element;
    };
    const stack = [document];
    const html = fs.readFileSync(path.join(web, "index.html"), "utf8");
    for (const match of html.matchAll(/<(\/?)([a-z][a-z0-9-]*)([^>]*)>/gi)) {
        const [, closing, name, attributes] = match;
        if (closing) { stack.pop(); continue; }
        const element = document.createElement(name);
        for (const attr of attributes.matchAll(/([\w-]+)(?:="([^"]*)")?/g)) element.setAttribute(attr[1], attr[2] ?? "");
        stack[stack.length - 1].append(element);
        if (!["meta", "link", "input", "br"].includes(name) && !/\/\s*$/.test(attributes)) stack.push(element);
    }
    document.body = document.querySelector("body");
    document.activeElement = document;
    const calls = [];
    const database = new Map(records.map((item) => [item.id, item]));
    const media = { matches: narrow, addEventListener(type, listener) { this.listener = listener; } };
    const h = { document, database, calls, media, confirm: true, override: null };
    h.savedUsage = new Map(records.map((item) => [item.id, measuredSummary(item.id)]));
    h.get = (id) => document.getElementById(id);
    h.fetch = async (url, options = {}) => {
        const call = { url, method: options.method || "GET", body: options.body ? JSON.parse(options.body) : undefined };
        calls.push(call);
        if (h.override) {
            const overridden = await h.override(call);
            if (overridden) return overridden;
        }
        if (url.includes("?limit=")) return disabled ? response({ error: "Disabled" }, 409) : response({ conversations: Array.from(database.values()) });
        if (url === "/v1/conversations" && call.method === "POST") {
            const item = record(C, call.body.title);
            database.set(C, item);
            return response({ conversation: item }, 201);
        }
        if (url === "/v1/chat/completions" || url === "/v1/chat/selection") {
            const id = call.body.conversation_id;
            if (id) h.savedUsage.set(id, measuredSummary(id, (h.savedUsage.get(id)?.exchange_count || 0) + 1));
            return response({
                choices: [{ message: { content: "Generated answer" } }],
                ...(id ? { conversation_id: id, memory_persistence: { saved: true }, conversation_usage: h.savedUsage.get(id) } : {}),
                stats: { input_tokens: 100, output_tokens: 25, total_tokens: 125, tokens_used: 125, ctx_size: 1000, model_name: "test-model" }
            });
        }
        const id = url.split("/")[3];
        if (!database.has(id)) return response({ error: "Not found" }, 404);
        if (url.endsWith("/usage")) return response({ usage: h.savedUsage.get(id) || measuredSummary(id, 0) });
        if (url.includes("/messages?")) return response({ conversation_id: id, messages: [{ conversation_id: id, role: "user", content: "Stored question" }] });
        if (url.endsWith("/title")) {
            const item = { ...database.get(id), title: call.body.title };
            database.set(id, item);
            return response({ conversation: item });
        }
        if (call.method === "DELETE") {
            database.delete(id);
            return response({ conversation_id: id, deleted: true });
        }
        return response({ conversation: database.get(id) });
    };
    const context = vm.createContext({
        document,
        window: { matchMedia: () => media, confirm: () => h.confirm },
        fetch: h.fetch,
        console
    });
    h.run = (code) => vm.runInContext(code, context);
    h.function = (name, ...args) => context[name](...args);
    vm.runInContext(fs.readFileSync(path.join(web, "app.js"), "utf8"), context);
    await flush();
    if (incognito) h.function("startIndependentConversation");
    return h;
}

test("default mode saves and renders searchable titles as text", async () => {
    const h = await harness({ records: [record(A, '<img src=x onerror="alert(1)">'), record(B, "Red Maple")] });
    assert.equal(h.get("conversation-enabled").checked, true);
    assert.equal(h.get("session-title").textContent, "New Conversation");
    assert.equal(h.get("conversation-list").querySelectorAll(".conversation-row").length, 2);
    assert.equal(h.get("conversation-list").querySelector(".conversation-row-title").textContent, '<img src=x onerror="alert(1)">');
    h.get("conversation-search").value = "RED";
    await h.get("conversation-search").dispatch("input");
    assert.equal(h.get("conversation-list").querySelectorAll(".conversation-row").length, 1);
    assert.equal(h.get("conversation-list").querySelector(".conversation-row-title").textContent, "Red Maple");
});

test("independent submission omits conversation ID and updates real metrics", async () => {
    const h = await harness({ incognito: true });
    h.get("prompt-input").value = "This is a test";
    await h.function("sendPrompt");
    assert.deepEqual(h.calls.at(-1).body, { messages: [{ role: "user", content: "This is a test" }] });
    assert.equal(h.get("tokens-used").textContent, "125");
    assert.equal(h.get("chat-box").children.length, 2);
});

test("new independent conversation requires no create request", async () => {
    const h = await harness({ incognito: true });
    const count = h.calls.length;
    await h.get("conversation-new").click();
    assert.equal(h.calls.length, count);
    assert.equal(h.get("conversation-dialog").open, false);
});

test("one row click opens history, highlights it, and submits to its ID", async () => {
    const h = await harness({ incognito: true });
    await h.get("conversation-list").querySelector(".conversation-link").click();
    assert.equal(h.get("session-title").textContent, "Blue Lantern");
    assert.equal(h.get("conversation-enabled").checked, true);
    assert.equal(h.get("conversation-list").querySelector(".conversation-link").getAttribute("aria-current"), "true");
    assert.equal(h.get("chat-box").children.length, 1);
    h.get("prompt-input").value = "Follow-up";
    await h.function("sendPrompt");
    assert.equal(h.calls.at(-1).body.conversation_id, A);
});

test("enabling saving and creating a name makes exactly one create request", async () => {
    const h = await harness({ incognito: true });
    h.get("conversation-enabled").checked = true;
    await h.get("conversation-enabled").dispatch("change");
    h.get("conversation-title").value = "New investigation";
    await h.get("conversation-name-form").dispatch("submit");
    assert.equal(h.calls.filter((call) => call.method === "POST" && call.url === "/v1/conversations").length, 1);
    assert.equal(h.get("session-title").textContent, "New investigation");
    assert.equal(h.run("activeConversationId"), C);
    assert.equal(h.get("conversation-dialog").open, false);
});

test("canceling opt-in preserves independent messages and restores saving off", async () => {
    const h = await harness({ incognito: true });
    h.function("addMessage", "user", "User", "Unsaved message");
    h.get("conversation-enabled").checked = true;
    await h.get("conversation-enabled").dispatch("change");
    await h.get("conversation-dialog-cancel").click();
    assert.equal(h.get("conversation-enabled").checked, false);
    assert.equal(h.get("chat-box").children.length, 1);
    assert.equal(h.run("activeConversationId"), null);
});

test("failed creation leaves dialog and current view intact", async () => {
    const h = await harness({ incognito: true });
    h.function("addMessage", "user", "User", "Existing message");
    h.get("conversation-enabled").checked = true;
    await h.get("conversation-enabled").dispatch("change");
    h.override = (call) => call.url === "/v1/conversations" ? response({ error: "Storage unavailable" }, 500) : null;
    h.get("conversation-title").value = "New investigation";
    await h.get("conversation-name-form").dispatch("submit");
    assert.equal(h.get("chat-box").children.length, 1);
    assert.equal(h.get("conversation-dialog").open, true);
    assert.equal(h.get("conversation-dialog-error").textContent, "Storage unavailable");
    assert.equal(h.run("activeConversationId"), null);
});

test("opening failure preserves the active conversation and messages", async () => {
    const h = await harness({ incognito: true });
    await h.function("openConversation", A);
    h.override = (call) => call.url.includes(B) ? response({ error: "Not found" }, 404) : null;
    await h.function("openConversation", B);
    assert.equal(h.get("session-title").textContent, "Blue Lantern");
    assert.equal(h.get("chat-box").children.length, 1);
    assert.equal(h.get("conversation-notice").dataset.error, "true");
});

test("renaming another row preserves the current conversation", async () => {
    const h = await harness({ incognito: true });
    await h.function("openConversation", A);
    h.function("openNameDialog", "rename", h.database.get(B));
    h.get("conversation-title").value = "Renamed Maple";
    await h.get("conversation-name-form").dispatch("submit");
    assert.equal(h.database.get(B).title, "Renamed Maple");
    assert.equal(h.get("session-title").textContent, "Blue Lantern");
    assert.equal(h.get("chat-box").children.length, 1);
});

test("header rename updates title and list exactly once", async () => {
    const h = await harness({ incognito: true });
    await h.function("openConversation", A);
    await h.get("conversation-rename").click();
    h.get("conversation-title").value = "Lantern updated";
    await h.get("conversation-name-form").dispatch("submit");
    assert.equal(h.calls.filter((call) => call.url.endsWith("/title")).length, 1);
    assert.equal(h.get("session-title").textContent, "Lantern updated");
    assert.equal(h.get("conversation-list").querySelector(".conversation-row-title").textContent, "Lantern updated");
});

test("deleting inactive history preserves active view; deleting active clears it", async () => {
    const h = await harness({ incognito: true });
    await h.function("openConversation", A);
    await h.function("deleteConversation", B);
    assert.equal(h.run("activeConversationId"), A);
    assert.equal(h.get("chat-box").children.length, 1);
    await h.get("conversation-delete").click();
    assert.equal(h.run("activeConversationId"), null);
    assert.equal(h.get("chat-box").children.length, 0);
    assert.equal(h.get("conversation-enabled").checked, true);
});

test("declining deletion leaves storage untouched", async () => {
    const h = await harness({ incognito: true });
    h.confirm = false;
    await h.function("deleteConversation", A);
    assert.equal(h.database.has(A), true);
    assert.equal(h.calls.some((call) => call.method === "DELETE"), false);
});

test("disabled server saving does not block independent prompts", async () => {
    const h = await harness({ disabled: true });
    assert.equal(h.get("conversation-enabled").disabled, true);
    assert.equal(h.get("send-button").disabled, false);
    assert.match(h.get("conversation-list-status").textContent, /disabled/i);
    h.get("prompt-input").value = "Hello";
    await h.function("sendPrompt");
    assert.equal(h.get("chat-box").children.length, 2);
});

test("save failures keep the answer and remain visible after another saved reply", async () => {
    const h = await harness({ incognito: true });
    await h.function("openConversation", A);
    h.override = (call) => call.url === "/v1/chat/completions" ? response({
        choices: [{ message: { content: "Answer remains visible" } }],
        conversation_id: A, memory_persistence: { saved: false, message: "Could not save" }
    }) : null;
    h.get("prompt-input").value = "Question";
    await h.function("sendPrompt");
    assert.equal(h.get("session-mode").textContent, "Save Failed");
    assert.match(h.get("chat-box").textContent, /Answer remains visible/);
    h.override = null;
    h.get("prompt-input").value = "Another question";
    await h.function("sendPrompt");
    assert.equal(h.get("session-mode").textContent, "Save Failed");
    assert.match(h.get("conversation-notice").textContent, /earlier exchange/i);
});

test("pending selection locks navigation; discard releases without saving", async () => {
    const h = await harness({ incognito: true });
    h.function("renderSkillSelection", { request_id: "pending", input_tokens: 120, input_budget: 100, context_size: 150, skills: [] });
    assert.equal(h.get("conversation-new").disabled, true);
    assert.equal(h.get("send-button").disabled, true);
    assert.equal(h.get("pending-request-bar").hidden, false);
    const count = h.calls.length;
    await h.function("openConversation", A);
    assert.equal(h.calls.length, count);
    await h.get("conversation-discard").click();
    assert.equal(h.get("send-button").disabled, false);
    assert.equal(h.get("pending-request-bar").hidden, true);
    assert.equal(h.calls.length, count);
});

test("skill exclusion sends original request ID without a conversation override", async () => {
    const h = await harness({ incognito: true });
    h.function("renderSkillSelection", {
        request_id: "original", input_tokens: 120, input_budget: 100, context_size: 150,
        skills: [{ name: "example", score: 0.8, characters: 200 }]
    });
    h.get("chat-box").querySelector("input").checked = true;
    await h.get("chat-box").querySelector("form").dispatch("submit");
    assert.deepEqual(h.calls.at(-1).body, { request_id: "original", remove_skill_names: ["example"] });
    assert.equal(h.get("send-button").disabled, false);
});

test("expired selections release the browser lock and show the error", async () => {
    const h = await harness({ incognito: true });
    h.function("renderSkillSelection", {
        request_id: "expired", input_tokens: 120, input_budget: 100, context_size: 150,
        skills: [{ name: "example", score: 0.8, characters: 200 }]
    });
    h.override = (call) => call.url === "/v1/chat/selection" ? response({ error: "Selection expired" }, 404) : null;
    h.get("chat-box").querySelector("input").checked = true;
    await h.get("chat-box").querySelector("form").dispatch("submit");
    assert.equal(h.get("send-button").disabled, false);
    assert.equal(h.get("conversation-notice").textContent, "Selection expired");
});

test("narrow navigation isolates main content and restores focus when closed", async () => {
    const h = await harness({ narrow: true });
    assert.equal(h.get("conversation-panel").hidden, true);
    await h.get("conversation-sidebar-toggle").click();
    assert.equal(h.get("conversation-workspace").inert, true);
    assert.equal(h.document.activeElement, h.get("sidebar-close"));
    await h.get("sidebar-close").click();
    assert.equal(h.get("conversation-workspace").inert, false);
    assert.equal(h.document.activeElement, h.get("conversation-sidebar-toggle"));
});

test("canceling view replacement preserves the draft and avoids requests", async () => {
    const h = await harness({ incognito: true });
    h.get("prompt-input").value = "Unsent draft";
    h.confirm = false;
    const count = h.calls.length;
    await h.function("openConversation", A);
    assert.equal(h.get("prompt-input").value, "Unsent draft");
    assert.equal(h.calls.length, count);
});

test("failed deletion preserves the active view and its navigation entry", async () => {
    const h = await harness({ incognito: true });
    await h.function("openConversation", A);
    h.override = (call) => call.method === "DELETE" ? response({ error: "Deletion failed" }, 500) : null;
    await h.function("deleteConversation", A);
    assert.equal(h.run("activeConversationId"), A);
    assert.equal(h.get("chat-box").children.length, 1);
    assert.equal(h.get("conversation-list").querySelectorAll(".conversation-row").length, 2);
    assert.equal(h.get("conversation-notice").textContent, "Deletion failed");
});

test("failed rename preserves metadata and keeps the editable dialog open", async () => {
    const h = await harness({ incognito: true });
    await h.function("openConversation", A);
    await h.get("conversation-rename").click();
    h.override = (call) => call.url.endsWith("/title") ? response({ error: "Rename failed" }, 500) : null;
    h.get("conversation-title").value = "New title";
    await h.get("conversation-name-form").dispatch("submit");
    assert.equal(h.get("session-title").textContent, "Blue Lantern");
    assert.equal(h.get("conversation-dialog").open, true);
    assert.equal(h.get("conversation-dialog-error").textContent, "Rename failed");
});

test("blank names make no create request", async () => {
    const h = await harness({ incognito: true });
    h.function("openNameDialog", "new");
    h.get("conversation-title").value = "   ";
    const count = h.calls.length;
    await h.get("conversation-name-form").dispatch("submit");
    assert.equal(h.calls.length, count);
    assert.equal(h.get("conversation-dialog-error").hidden, false);
});

test("an in-flight prompt prevents duplicate sends and conversation switching", async () => {
    const h = await harness({ incognito: true });
    let resolveRequest;
    h.override = (call) => call.url === "/v1/chat/completions"
        ? new Promise((resolve) => { resolveRequest = resolve; }) : null;
    h.get("prompt-input").value = "Question";
    const submission = h.function("sendPrompt");
    assert.equal(h.get("conversation-new").disabled, true);
    assert.equal(h.get("send-button").disabled, true);
    const count = h.calls.length;
    await h.function("openConversation", A);
    await h.function("sendPrompt");
    assert.equal(h.calls.length, count);
    resolveRequest(response({ choices: [{ message: { content: "Answer" } }] }));
    await submission;
    assert.equal(h.get("send-button").disabled, false);
});

test("repeated overflow responses retain the pending lock until completion", async () => {
    const h = await harness({ incognito: true });
    const pending = {
        request_id: "original", selection_required: true,
        input_tokens: 120, input_budget: 100, context_size: 150,
        skills: [{ name: "example", score: 0.8, characters: 200 }]
    };
    h.function("renderSkillSelection", pending);
    h.override = (call) => call.url === "/v1/chat/selection" ? response(pending) : null;
    h.get("chat-box").querySelector("input").checked = true;
    await h.get("chat-box").querySelector("form").dispatch("submit");
    assert.equal(h.get("send-button").disabled, true);
    assert.equal(h.get("chat-box").querySelectorAll(".skill-selection").length, 1);
    h.override = null;
    h.get("chat-box").querySelector("input").checked = true;
    await h.get("chat-box").querySelector("form").dispatch("submit");
    assert.equal(h.get("send-button").disabled, false);
    assert.equal(h.get("pending-request-bar").hidden, true);
});

test("mobile drawer Escape closes it and returns focus", async () => {
    const h = await harness({ narrow: true });
    await h.get("conversation-sidebar-toggle").click();
    await h.document.dispatch("keydown", { key: "Escape" });
    assert.equal(h.get("conversation-panel").hidden, true);
    assert.equal(h.get("conversation-workspace").inert, false);
    assert.equal(h.document.activeElement, h.get("conversation-sidebar-toggle"));
});

test("renaming inside the mobile drawer keeps focus out of inert main content", async () => {
    const h = await harness({ narrow: true });
    await h.get("conversation-sidebar-toggle").click();
    h.function("openNameDialog", "rename", h.database.get(B));
    h.get("conversation-title").value = "Maple renamed";
    await h.get("conversation-name-form").dispatch("submit");
    assert.equal(h.get("conversation-workspace").inert, true);
    assert.equal(h.document.activeElement, h.get("conversation-search"));
});

test("session totals and the progress bar accumulate across completed exchanges", async () => {
    const h = await harness({ incognito: true });
    for (const prompt of ["First", "Second"]) {
        h.get("prompt-input").value = prompt;
        await h.function("sendPrompt");
    }
    assert.equal(h.get("conversation-total-tokens").textContent, "250");
    assert.equal(h.get("conversation-input-tokens").textContent, "200");
    assert.equal(h.get("conversation-output-tokens").textContent, "50");
    assert.equal(h.get("context-progress-fill").style.width, "25%");
    assert.equal(h.get("tokens-used").textContent, "250");
});

test("opening saved history restores full totals rather than visible-message totals", async () => {
    const h = await harness({ incognito: true });
    h.savedUsage.set(A, measuredSummary(A, 20));
    await h.function("openConversation", A);
    assert.equal(h.get("chat-box").children.length, 1);
    assert.equal(h.get("conversation-total-tokens").textContent, (2500).toLocaleString());
    assert.match(h.get("usage-coverage").textContent, /20 of 20/);
    assert.equal(h.get("context-progress-fill").style.width, "100%");
    assert.equal(h.get("context-percent").textContent, "250%");
    assert.equal(h.get("conversation-usage-label").textContent, "Saved Conversation Usage");
});

test("restoration uses the historical request model and capacity", async () => {
    const h = await harness({ incognito: true });
    h.savedUsage.set(A, measuredSummary(A, 1, 4000, "previous-model"));
    await h.function("openConversation", A);
    assert.equal(h.get("context-size").textContent, (4000).toLocaleString());
    assert.equal(h.get("metrics-model").textContent, "Latest Model: previous-model");
    assert.equal(h.get("context-progress-fill").style.width, "3.125%");
    h.function("setBusy", true, "Processing request…");
    assert.equal(h.get("context-progress-fill").style.width, "3.125%");
    assert.equal(h.get("stats-panel").getAttribute("aria-busy"), "true");
    h.function("setBusy", false);
});

test("legacy exchanges show unavailable counts rather than fabricated zero usage", async () => {
    const h = await harness({ incognito: true });
    h.savedUsage.set(A, {
        conversation_id: A, exchange_count: 2, recorded_exchanges: 0,
        input_tokens: null, output_tokens: null, total_tokens: null,
        input_tokens_measured_exchanges: 0, output_tokens_measured_exchanges: 0,
        total_tokens_measured_exchanges: 0, is_partial: true, latest_usage: null
    });
    await h.function("openConversation", A);
    assert.equal(h.get("conversation-total-tokens").textContent, "--");
    assert.equal(h.get("tokens-used").textContent, "--");
    assert.equal(h.get("context-progress").getAttribute("aria-valuenow"), null);
    assert.match(h.get("usage-coverage").textContent, /0 of 2/);
});

test("partially measured history identifies totals as lower bounds", async () => {
    const h = await harness({ incognito: true });
    h.savedUsage.set(A, { ...measuredSummary(A), exchange_count: 2, is_partial: true });
    await h.function("openConversation", A);
    assert.equal(h.get("conversation-total-tokens").textContent, "≥ 125");
    assert.match(h.get("usage-coverage").textContent, /1 of 2/);
});

test("unknown output leaves total context usage unknown", async () => {
    const h = await harness({ incognito: true });
    h.override = (call) => call.url === "/v1/chat/completions" ? response({
        choices: [{ message: { content: "Answer" } }],
        stats: { input_tokens: 10, output_tokens: null, total_tokens: null, tokens_used: null, ctx_size: 1000 }
    }) : null;
    h.get("prompt-input").value = "Question";
    await h.function("sendPrompt");
    assert.equal(h.get("request-input-tokens").textContent, "10");
    assert.equal(h.get("request-output-tokens").textContent, "--");
    assert.equal(h.get("tokens-used").textContent, "--");
    assert.equal(h.get("conversation-total-tokens").textContent, "--");
    assert.equal(h.get("context-progress").getAttribute("aria-valuenow"), null);
});

test("a save failure leaves the cumulative bar and saved totals unchanged", async () => {
    const h = await harness({ incognito: true });
    await h.function("openConversation", A);
    h.override = (call) => call.url === "/v1/chat/completions" ? response({
        choices: [{ message: { content: "Unsaved answer" } }],
        conversation_id: A, memory_persistence: { saved: false },
        stats: { input_tokens: 200, output_tokens: 50, total_tokens: 250, tokens_used: 250, ctx_size: 1000 }
    }) : null;
    h.get("prompt-input").value = "Question";
    await h.function("sendPrompt");
    assert.equal(h.get("context-progress-fill").style.width, "12.5%");
    assert.equal(h.get("conversation-total-tokens").textContent, "125");
    assert.match(h.get("usage-coverage").textContent, /not saved/i);
});

test("a usage endpoint failure does not prevent opening saved messages", async () => {
    const h = await harness({ incognito: true });
    h.override = (call) => call.url.endsWith("/usage") ? response({ error: "Unavailable" }, 500) : null;
    await h.function("openConversation", A);
    assert.equal(h.get("session-title").textContent, "Blue Lantern");
    assert.equal(h.get("chat-box").children.length, 1);
    assert.equal(h.get("conversation-total-tokens").textContent, "--");
    assert.match(h.get("usage-coverage").textContent, /unavailable/i);
});

test("totals returned for another conversation are not displayed", async () => {
    const h = await harness({ incognito: true });
    h.savedUsage.set(A, measuredSummary(B, 100));
    await h.function("openConversation", A);
    assert.equal(h.get("conversation-total-tokens").textContent, "--");
    assert.equal(h.get("context-size").textContent, "--");
});

test("capacity overrun is labeled while the visual fill is bounded", async () => {
    const h = await harness({ incognito: true });
    h.function("handleResponse", { choices: [{ message: { content: "Answer" } }], stats: { tokens_used: 1500, ctx_size: 1000 } });
    assert.equal(h.get("context-progress-fill").style.width, "100%");
    assert.equal(h.get("context-percent").textContent, "150%");
    assert.equal(h.get("context-progress").getAttribute("aria-valuenow"), "100");
    assert.match(h.get("context-remaining").textContent, /Reference Size Exceeded/);
});

test("zero output tokens are displayed as measured zero", async () => {
    const h = await harness({ incognito: true });
    h.function("handleResponse", { choices: [{ message: { content: "Answer" } }], stats: { input_tokens: 100, output_tokens: 0, total_tokens: 100, ctx_size: 1000 } });
    assert.equal(h.get("request-output-tokens").textContent, "0");
    assert.equal(h.get("context-progress-fill").style.width, "10%");
});

test("missing totals after a committed save are not reported as a failed save", async () => {
    const h = await harness({ incognito: true });
    await h.function("openConversation", A);
    h.override = (call) => call.url === "/v1/chat/completions" ? response({
        choices: [{ message: { content: "Saved answer" } }],
        conversation_id: A, memory_persistence: { saved: true }, conversation_usage: null,
        stats: { input_tokens: 100, output_tokens: 25, total_tokens: 125, ctx_size: 1000 }
    }) : null;
    h.get("prompt-input").value = "Question";
    await h.function("sendPrompt");
    assert.equal(h.get("session-mode").textContent, "Saved Mode");
    assert.equal(h.get("conversation-total-tokens").textContent, "--");
    assert.equal(h.get("context-progress-fill").style.width, "0%");
    assert.equal(h.get("context-progress").getAttribute("aria-valuenow"), null);
});

test("the cumulative progress bar is inside Usage Details", async () => {
    const h = await harness();
    const details = h.document.querySelector(".metrics-details");
    assert.equal(details.contains(h.get("context-progress")), true);
    assert.equal(details.contains(h.get("context-percent")), true);
    assert.equal(details.open, false);
    assert.equal(h.get("context-progress").getAttribute("aria-describedby").includes("usage-reference-note"), true);
});

test("partial cumulative totals mark both the percentage and accessible value as lower bounds", async () => {
    const h = await harness();
    h.savedUsage.set(A, { ...measuredSummary(A), exchange_count: 3, is_partial: true });
    await h.function("openConversation", A);
    assert.equal(h.get("tokens-used").textContent, "≥ 125");
    assert.equal(h.get("context-percent").textContent, "≥ 12.5%");
    assert.match(h.get("context-progress").getAttribute("aria-valuetext"), /At Least/);
});

test("new Incognito sessions clear accumulated totals and the reference", async () => {
    const h = await harness({ incognito: true });
    h.get("prompt-input").value = "Question";
    await h.function("sendPrompt");
    assert.equal(h.get("tokens-used").textContent, "125");
    await h.get("conversation-new").click();
    assert.equal(h.get("tokens-used").textContent, "0");
    assert.equal(h.get("context-size").textContent, "--");
    assert.equal(h.get("context-progress-fill").style.width, "0%");
});

test("a failed save cannot replace the saved reference with the unsaved response capacity", async () => {
    const h = await harness();
    await h.function("openConversation", A);
    h.override = (call) => call.url === "/v1/chat/completions" ? response({
        choices: [{ message: { content: "Unsaved answer" } }],
        conversation_id: A, memory_persistence: { saved: false },
        stats: { input_tokens: 200, output_tokens: 50, total_tokens: 250, ctx_size: 2000 }
    }) : null;
    h.get("prompt-input").value = "Question";
    await h.function("sendPrompt");
    assert.equal(h.get("context-size").textContent, (1000).toLocaleString());
    assert.equal(h.get("context-progress-fill").style.width, "12.5%");
});

test("switching saved conversations restores each conversation's own totals and reference", async () => {
    const h = await harness();
    h.savedUsage.set(A, measuredSummary(A, 4, 1000));
    h.savedUsage.set(B, measuredSummary(B, 2, 2000));
    await h.function("openConversation", A);
    assert.equal(h.get("context-progress-fill").style.width, "50%");
    await h.function("openConversation", B);
    assert.equal(h.get("context-progress-fill").style.width, "12.5%");
    assert.equal(h.get("tokens-used").textContent, "250");
    await h.function("openConversation", A);
    assert.equal(h.get("tokens-used").textContent, "500");
    assert.equal(h.get("context-progress-fill").style.width, "50%");
});

test("an unmeasured latest exchange does not borrow an earlier model reference", async () => {
    const h = await harness();
    h.savedUsage.set(A, { ...measuredSummary(A), exchange_count: 2, latest_usage: null, is_partial: true });
    await h.function("openConversation", A);
    assert.equal(h.get("tokens-used").textContent, "≥ 125");
    assert.equal(h.get("context-size").textContent, "--");
    assert.equal(h.get("context-progress").getAttribute("aria-valuenow"), null);
});

test("zero measured usage renders a genuine zero percent", async () => {
    const h = await harness({ incognito: true });
    h.function("handleResponse", { choices: [{ message: { content: "Answer" } }], stats: {
        input_tokens: 0, output_tokens: 0, total_tokens: 0, ctx_size: 1000
    } });
    assert.equal(h.get("tokens-used").textContent, "0");
    assert.equal(h.get("context-percent").textContent, "0%");
    assert.equal(h.get("context-progress").getAttribute("aria-valuenow"), "0");
});

test("unsafe cumulative sums become unavailable instead of rounded totals", async () => {
    const h = await harness({ incognito: true });
    for (const total of [Number.MAX_SAFE_INTEGER, 1]) {
        h.function("handleResponse", { choices: [{ message: { content: "Answer" } }], stats: {
            total_tokens: total, ctx_size: 1000
        } });
    }
    assert.equal(h.get("tokens-used").textContent, "--");
    assert.equal(h.get("conversation-total-tokens").textContent, "--");
    assert.equal(h.get("context-progress").getAttribute("aria-valuenow"), null);
});

test("conversation action disclosure toggles and opens rename", async () => {
    const h = await harness();
    const actions = h.get("conversation-list").querySelector(".context-actions");
    const summary = actions.querySelector("summary");
    await summary.click();
    assert.equal(actions.open, true);
    await summary.click();
    assert.equal(actions.open, false);
    await summary.click();
    await actions.querySelector("button").click();
    assert.equal(actions.open, false);
    assert.equal(h.get("conversation-dialog").open, true);
});

test("pending selection prevents conversation action disclosure", async () => {
    const h = await harness();
    h.function("renderSkillSelection", { request_id: "pending", input_tokens: 120, input_budget: 100, context_size: 150, skills: [] });
    const actions = h.get("conversation-list").querySelector(".context-actions");
    await actions.querySelector("summary").click();
    assert.equal(actions.open, false);
});

test("failed chat restores the prompt without automatically resending", async () => {
    const h = await harness({ incognito: true });
    h.override = (call) => call.url === "/v1/chat/completions" ? response({ error: "Inference unavailable" }, 503) : null;
    h.get("prompt-input").value = "Investigate this alert";
    await h.function("sendPrompt");
    assert.equal(h.get("prompt-input").value, "Investigate this alert");
    assert.equal(h.get("send-button").disabled, false);
    assert.match(h.get("chat-box").textContent, /Inference unavailable/);
    assert.equal(h.calls.filter((call) => call.url === "/v1/chat/completions").length, 1);
});

test("non-object API responses fail clearly and release the composer", async () => {
    const h = await harness({ incognito: true });
    for (const data of [null, [], "invalid"]) {
        h.override = (call) => call.url === "/v1/chat/completions" ? response(data) : null;
        h.get("prompt-input").value = "Question";
        await h.function("sendPrompt");
        assert.equal(h.get("send-button").disabled, false);
        assert.match(h.get("chat-box").textContent, /Invalid Response/);
    }
});

test("missing answers do not update usage from an invalid response", async () => {
    const h = await harness({ incognito: true });
    h.override = (call) => call.url === "/v1/chat/completions" ? response({ stats: { total_tokens: 999, ctx_size: 1000 } }) : null;
    h.get("prompt-input").value = "Question";
    await h.function("sendPrompt");
    assert.equal(h.get("tokens-used").textContent, "0");
    assert.match(h.get("chat-box").textContent, /No Assistant Response/);
});

test("malformed skill reviews never lock the conversation", async () => {
    const h = await harness({ incognito: true });
    h.override = (call) => call.url === "/v1/chat/completions" ? response({ selection_required: true, skills: [] }) : null;
    h.get("prompt-input").value = "Question";
    await h.function("sendPrompt");
    assert.equal(h.get("send-button").disabled, false);
    assert.equal(h.get("pending-request-bar").hidden, true);
    assert.match(h.get("chat-box").textContent, /Invalid Skill Selection/);
});

test("terminal budget rejection releases the review and shows the backend error", async () => {
    const h = await harness({ incognito: true });
    h.function("renderSkillSelection", { request_id: "pending", skills: [{ name: "example", score: 0.8, characters: 200 }] });
    h.override = (call) => call.url === "/v1/chat/selection" ? response({ error: "Shorten the prompt" }, 400) : null;
    h.get("chat-box").querySelector("input").checked = true;
    await h.get("chat-box").querySelector("form").dispatch("submit");
    assert.equal(h.get("send-button").disabled, false);
    assert.equal(h.get("pending-request-bar").hidden, true);
    assert.match(h.get("conversation-notice").textContent, /Shorten the prompt/);
});

test("assistant content stays text and uses CORVEUS branding", async () => {
    const h = await harness({ incognito: true });
    const content = '<img src=x onerror="alert(1)">';
    h.override = (call) => call.url === "/v1/chat/completions" ? response({ choices: [{ message: { content } }] }) : null;
    h.get("prompt-input").value = "Question";
    await h.function("sendPrompt");
    assert.match(h.document.title, /CORVEUS/);
    assert.equal(h.get("chat-box").querySelectorAll("img").length, 0);
    assert.ok(h.get("chat-box").textContent.includes(content));
    assert.ok(h.get("chat-box").textContent.includes("Corveus"));
});

test("inspector follows saved conversation selection and usage restoration", async () => {
    const h = await harness();
    h.savedUsage.set(A, measuredSummary(A, 2, 4000, "historical-model"));
    await h.function("openConversation", A);
    assert.equal(h.get("inspector-conversation-title").textContent, "Blue Lantern");
    assert.equal(h.get("inspector-saving-state").textContent, "Saving to This Conversation");
    assert.equal(h.get("inspector-usage-total").textContent, "250 Total Tokens");
    assert.match(h.get("inspector-request-model").textContent, /historical-model/);
    assert.equal(h.get("inspector-usage-coverage").textContent, h.get("usage-coverage").textContent);
});

test("inspector clears saved measurements when starting incognito", async () => {
    const h = await harness();
    h.savedUsage.set(A, measuredSummary(A, 2));
    await h.function("openConversation", A);
    h.function("startIndependentConversation");
    assert.equal(h.get("inspector-saving-state").textContent, "Incognito · Not Saved");
    assert.equal(h.get("inspector-usage-total").textContent, "0 Total Tokens");
    assert.equal(h.get("inspector-request-model").textContent, "Latest Model: --");
});

test("inspector exposes unavailable totals without blocking history", async () => {
    const h = await harness();
    h.override = (call) => call.url.endsWith("/usage") ? response({ error: "Unavailable" }, 500) : null;
    await h.function("openConversation", A);
    assert.equal(h.get("inspector-usage-total").textContent, "-- Total Tokens");
    assert.match(h.get("inspector-usage-coverage").textContent, /Unavailable/);
    assert.match(h.get("chat-box").textContent, /Stored question/);
});

test("inspector marks save failures and keeps committed totals", async () => {
    const h = await harness();
    h.savedUsage.set(A, measuredSummary(A, 2));
    await h.function("openConversation", A);
    h.override = (call) => call.url === "/v1/chat/completions" ? response({
        choices: [{ message: { content: "Unsaved reply" } }], conversation_id: A,
        memory_persistence: { saved: false }, stats: { total_tokens: 999 }
    }) : null;
    h.get("prompt-input").value = "Question";
    await h.function("sendPrompt");
    assert.match(h.get("inspector-saving-state").textContent, /Save Warning/);
    assert.equal(h.get("inspector-usage-total").textContent, "250 Total Tokens");
    assert.match(h.get("inspector-usage-coverage").textContent, /Not Saved/);
});
