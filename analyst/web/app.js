"use strict";
const durableJobs = typeof window.createJobClient === "function"
    ? window.createJobClient()
    : null;
let durableActionRunning = false;
let durableCancelRunning = false;
// Session state stays in this tab. Nothing is written to browser storage.
let isSending = false;
let pendingRequestId = null;
let activeConversationId = null;
let activeConversationTitle = "";
let saveWarning = false;
let hasUnrecordedRequest = false;
let memoryAvailable = null;
let historyLoaded = false;
let historyError = "";
let dialogAction = null;
const savedConversationMetadata = new Map();
let localUsageRecords = [];
let latestRequestStats = null;
let currentUsageSummary = emptyUsageSummary();
let usageSummaryAvailable = true;
let usageSummaryNote = "";

const input = document.getElementById("prompt-input");
const chatBox = document.getElementById("chat-box");
const inputForm = document.getElementById("input-container");
const sendButton = document.getElementById("send-button");
const conversationEnabled = document.getElementById("conversation-enabled");
const savedDarkTheme = document.getElementById("saved-dark-theme");
const savedModeIndicator = document.getElementById("saved-mode-indicator");
const conversationNew = document.getElementById("conversation-new");
const conversationRefresh = document.getElementById("conversation-refresh");
const conversationList = document.getElementById("conversation-list");
const conversationSearch = document.getElementById("conversation-search");
const conversationListStatus = document.getElementById("conversation-list-status");
const conversationDelete = document.getElementById("conversation-delete");
const conversationRename = document.getElementById("conversation-rename");
const conversationDiscard = document.getElementById("conversation-discard");
const conversationStatus = document.getElementById("conversation-status");
const conversationNotice = document.getElementById("conversation-notice");
const conversationTitle = document.getElementById("conversation-title");
const sessionTitle = document.getElementById("session-title");
const sessionMode = document.getElementById("session-mode");
const sessionDescription = document.getElementById("session-description");
const sessionActions = document.getElementById("session-actions");
const requestStatus = document.getElementById("request-status");
const pendingBar = document.getElementById("pending-request-bar");
const terminal = document.querySelector(".terminal");
const workspace = document.getElementById("conversation-workspace");
const conversationPanel = document.getElementById("conversation-panel");
const sidebarToggle = document.getElementById("conversation-sidebar-toggle");
const sidebarClose = document.getElementById("sidebar-close");
const sidebarBackdrop = document.getElementById("sidebar-backdrop");
const narrowLayout = window.matchMedia("(max-width: 760px)");
const nameDialog = document.getElementById("conversation-dialog");
const nameForm = document.getElementById("conversation-name-form");
const dialogTitle = document.getElementById("conversation-dialog-title");
const dialogError = document.getElementById("conversation-dialog-error");
const dialogCancel = document.getElementById("conversation-dialog-cancel");
const dialogSave = document.getElementById("conversation-dialog-save");

conversationEnabled.checked = true;

function locked() {
    return isSending || pendingRequestId !== null;
}

function setConversationNotice(message, isError = false) {
    conversationNotice.textContent = message;
    conversationNotice.hidden = !message;
    conversationNotice.dataset.error = String(isError);
}

function setConversationSidebarVisible(visible, moveFocus = true) {
    const focusWasInside = conversationPanel.contains(document.activeElement);
    conversationPanel.hidden = !visible;
    terminal.classList.toggle("sidebar-collapsed", !visible);
    sidebarToggle.setAttribute("aria-expanded", String(visible));
    sidebarBackdrop.hidden = !(visible && narrowLayout.matches);
    workspace.inert = visible && narrowLayout.matches;
    if (moveFocus && !nameDialog.open) {
        if (visible && narrowLayout.matches) sidebarClose.focus();
        else if (!visible && focusWasInside) sidebarToggle.focus();
    }
}

function updateConversationControls() {
    const unavailable = memoryAvailable === false;
    conversationEnabled.disabled = locked() || (unavailable && !conversationEnabled.checked);
    conversationNew.disabled = locked() || (unavailable && conversationEnabled.checked);
    conversationRefresh.disabled = locked();
    conversationRename.disabled = locked() || activeConversationId === null;
    conversationDelete.disabled = locked() || activeConversationId === null;
    conversationDiscard.disabled = isSending || pendingRequestId === null;
    sendButton.disabled = locked();
    input.readOnly = isSending;
    input.setAttribute("aria-busy", String(isSending));
    pendingBar.hidden = pendingRequestId === null;
    sessionActions.hidden = activeConversationId === null;
    conversationTitle.disabled = isSending;
    dialogCancel.disabled = isSending;
    dialogSave.disabled = isSending;

    for (const button of conversationList.querySelectorAll("button")) button.disabled = locked();
    for (const summary of document.querySelectorAll(".context-actions > summary")) {
        summary.setAttribute("aria-disabled", String(locked()));
        summary.tabIndex = locked() ? -1 : 0;
    }
    for (const row of conversationList.querySelectorAll(".conversation-row")) {
        const active = row.dataset.conversationId === activeConversationId;
        row.classList.toggle("is-active", active);
        const link = row.querySelector(".conversation-link");
        if (active) link.setAttribute("aria-current", "true");
        else link.removeAttribute("aria-current");
    }

    const incognito = !conversationEnabled.checked;
    const savedDark = !incognito && savedDarkTheme.checked;
    document.body.classList.toggle("incognito", incognito);
    document.body.classList.toggle("black-theme", incognito || savedDark);
    savedDarkTheme.disabled = incognito;
    savedModeIndicator.hidden = !savedDark || saveWarning;

    sessionTitle.textContent = activeConversationId === null
        ? (incognito ? "Incognito Conversation" : "New Conversation")
        : activeConversationTitle;

    sessionMode.textContent = saveWarning
        ? "Save Failed"
        : incognito ? "Incognito" : "Saved Mode";

    sessionMode.dataset.mode = saveWarning
        ? "warning"
        : incognito ? "incognito" : "saved";

    sessionDescription.textContent = activeConversationId !== null
        ? "New Exchanges Are Saved. Earlier Messages Are Not Used for Model Recall Yet."
        : incognito
            ? "Exchanges Are Not Saved to Conversation History."
            : "Choose a Conversation Name Before Sending Your First Message.";

    conversationStatus.textContent = incognito
        ? "Incognito · Conversation History Is Not Saved"
        : saveWarning
            ? "Saved Mode · Some Exchanges Were Not Saved"
            : activeConversationId !== null
                ? "Saved Mode"
                : "Saved Mode · Ready for a New Conversation";

    conversationNew.textContent = incognito
        ? "+ New Incognito Conversation"
        : "+ New Conversation";

    document.title = sessionTitle.textContent + " | CORVEUS";
    if (!isSending) requestStatus.textContent = pendingRequestId !== null ? "Awaiting Skill Selection" : "Ready";
    document.getElementById("stats-panel").setAttribute("aria-busy", String(isSending));
    renderConversationUsage();
    document.getElementById("inspector-conversation-title").textContent = sessionTitle.textContent;
    document.getElementById("inspector-saving-state").textContent = saveWarning
        ? "Save Warning · Some Exchanges Were Not Saved"
        : memoryAvailable === false ? "Conversation Storage Disabled"
            : activeConversationId !== null ? "Saving to This Conversation"
                : conversationEnabled.checked ? "Saving On · Name Required" : "Incognito · Not Saved";
}

function setBusy(busy, activity = "Working…") {
    isSending = busy;
    if (busy) requestStatus.textContent = activity;
    updateConversationControls();
}

async function requestJSON(url, method = "GET", payload) {
    const options = { method, cache: "no-store" };
    if (payload !== undefined) {
        options.headers = { "Content-Type": "application/json" };
        options.body = JSON.stringify(payload);
    }
    const response = await fetch(url, options);
    let data;
    try {
        data = await response.json();
    } catch {
        const error = new Error("The Server Returned an Unreadable Response (HTTP " + response.status + ").");
        error.status = response.status;
        throw error;
    }
    if (data === null || typeof data !== "object" || Array.isArray(data)) {
        const error = new Error("The Server Returned an Invalid Response (HTTP " + response.status + ").");
        error.status = response.status;
        throw error;
    }
    if (!response.ok) {
        const error = new Error(typeof data.error === "string"
            ? data.error : data.error?.message || "HTTP " + response.status);
        error.status = response.status;
        throw error;
    }
    return data;
}

function postJSON(url, payload) {
    return requestJSON(url, "POST", payload);
}

function requireConversationId(value) {
    if (typeof value !== "string" || !/^[0-9a-f]{32}$/.test(value)) {
        throw new Error("The Server Returned an Invalid Conversation ID.");
    }
    return value;
}

function validateConversationMetadata(conversation) {
    requireConversationId(conversation?.id);
    if (
        typeof conversation.title !== "string" || !conversation.title.trim() ||
        typeof conversation.created_at !== "string" ||
        Number.isNaN(Date.parse(conversation.created_at)) ||
        typeof conversation.updated_at !== "string" ||
        Number.isNaN(Date.parse(conversation.updated_at))
    ) throw new Error("The Server Returned Invalid Conversation Details.");
    return conversation;
}

function resetVisibleChat() {
    chatBox.replaceChildren();
    input.value = "";
    saveWarning = false;
    hasUnrecordedRequest = false;
    localUsageRecords = [];
    currentUsageSummary = emptyUsageSummary();
    usageSummaryAvailable = true;
    usageSummaryNote = "";
    updateStats(null);
    renderConversationUsage();
}

function confirmViewReplacement() {
    const unsaved = Boolean(input.value.trim()) || hasUnrecordedRequest ||
        (chatBox.childElementCount > 0 && (activeConversationId === null || saveWarning));
    return !unsaved || window.confirm("Unsaved Messages or Drafts in This View Will Be Cleared. Continue?");
}

function startIndependentConversation() {
    activeConversationId = null;
    activeConversationTitle = "";
    conversationEnabled.checked = false;
    resetVisibleChat();
    setConversationNotice("");
    updateConversationControls();
    if (narrowLayout.matches) setConversationSidebarVisible(false);
    input.focus();
}

function renderConversationList() {
    const query = conversationSearch.value.trim().toLocaleLowerCase();
    const conversations = Array.from(savedConversationMetadata.values())
        .sort((a, b) => b.updated_at.localeCompare(a.updated_at) || a.id.localeCompare(b.id))
        .filter((conversation) => conversation.title.toLocaleLowerCase().includes(query));
    const rows = [];
    for (const conversation of conversations) {
        const row = document.createElement("li");
        row.className = "conversation-row";
        row.dataset.conversationId = conversation.id;

        const link = document.createElement("button");
        link.type = "button";
        link.className = "conversation-link";
        link.title = conversation.title;
        link.setAttribute("aria-label", "Open " + conversation.title);
        const title = document.createElement("span");
        title.className = "conversation-row-title";
        title.textContent = conversation.title;
        const date = document.createElement("time");
        date.className = "conversation-row-date";
        date.dateTime = conversation.created_at;
        date.textContent = new Intl.DateTimeFormat(undefined, {
            year: "numeric", month: "short", day: "2-digit"
        }).format(new Date(conversation.created_at));
        link.append(title, date);
        link.addEventListener("click", () => openConversation(conversation.id));

        const actions = document.createElement("details");
        actions.className = "context-actions";
        const summary = document.createElement("summary");
        summary.textContent = "•••";
        summary.setAttribute("aria-label", "Actions for " + conversation.title);
        summary.addEventListener("click", (event) => {
            event.preventDefault();
            if (!locked()) actions.open = !actions.open;
        });
        const menu = document.createElement("div");
        menu.className = "action-menu";
        const rename = document.createElement("button");
        rename.type = "button";
        rename.textContent = "Rename";
        rename.addEventListener("click", () => {
            if (locked()) return;
            actions.open = false;
            openNameDialog("rename", conversation);
        });
        const remove = document.createElement("button");
        remove.type = "button";
        remove.className = "danger-text";
        remove.textContent = "Delete";
        remove.addEventListener("click", () => {
            actions.open = false;
            deleteConversation(conversation.id);
        });
        menu.append(rename, remove);
        actions.append(summary, menu);
        row.append(link, actions);
        rows.push(row);
    }
    conversationList.replaceChildren(...rows);
    conversationListStatus.textContent = historyError ||
        (!historyLoaded ? "Loading Saved Conversations…"
            : !savedConversationMetadata.size ? "No Saved Conversations Yet."
                : !rows.length ? "No Conversations Match Your Search."
                    : savedConversationMetadata.size >= 1000 ? "Showing the Latest 1,000. Search Filters This List." : "");
    conversationListStatus.hidden = !conversationListStatus.textContent;
    updateConversationControls();
}

async function runConversationAction(action, activity = "Working…") {
    if (locked()) return;
    setBusy(true, activity);
    setConversationNotice("");
    try {
        await action();
    } catch (error) {
        setConversationNotice(error.message, true);
    } finally {
        setBusy(false);
    }
}

async function refreshConversations() {
    await runConversationAction(async () => {
        try {
            const data = await requestJSON("/v1/conversations?limit=1000&offset=0");
            if (!Array.isArray(data.conversations)) throw new Error("The Server Returned an Invalid Conversation List.");
            const records = data.conversations.map(validateConversationMetadata);
            savedConversationMetadata.clear();
            for (const conversation of records) savedConversationMetadata.set(conversation.id, conversation);
            memoryAvailable = true;
            historyLoaded = true;
            historyError = "";
            const active = savedConversationMetadata.get(activeConversationId);
            if (active) activeConversationTitle = active.title;
        } catch (error) {
            historyLoaded = true;
                     if (error.status === 409) {
                memoryAvailable = false;
                if (activeConversationId === null) {
                    conversationEnabled.checked = false;
                }
                historyError = "Saving Is Disabled on This Server. Incognito Is Available.";
                setConversationNotice(
                    "Saving Is Disabled on This Server. New Conversations Use Incognito."
                );
            } else {
                historyError = "Could Not Refresh Saved Conversations. Use Refresh to Retry.";
            }
        }
        renderConversationList();
    }, "Loading Conversations…");
}

async function openConversation(id) {
    if (locked() || nameDialog.open) return;
    if (id === activeConversationId) return;
    if (!confirmViewReplacement()) return;
    await runConversationAction(async () => {
        requireConversationId(id);
        const [metadata, data, usageData] = await Promise.all([
            requestJSON("/v1/conversations/" + id),
            requestJSON("/v1/conversations/" + id + "/messages?recent_exchanges=50"),
            requestJSON("/v1/conversations/" + id + "/usage").catch(() => null)
        ]);
        const conversation = validateConversationMetadata(metadata.conversation);
        if (
            conversation.id !== id || data.conversation_id !== id ||
            !Array.isArray(data.messages) ||
            !data.messages.every((message) =>
                message.conversation_id === id &&
                ["user", "assistant"].includes(message.role) &&
                typeof message.content === "string"
            )
        ) throw new Error("The Server Returned Invalid Conversation Messages.");
        savedConversationMetadata.set(id, conversation);
        activeConversationId = id;
        activeConversationTitle = conversation.title;
        conversationEnabled.checked = true;
        memoryAvailable = true;
        resetVisibleChat();
        restoreConversationUsage(usageData?.usage, id);
        for (const message of data.messages) {
            addMessage(message.role, message.role === "user" ? "Analyst" : "Corveus", message.content);
        }
        renderConversationList();
        setConversationNotice("Opened “" + conversation.title + "”. Showing Up to 50 Recent Exchanges. History Recall Is Not Enabled.");
        if (narrowLayout.matches) setConversationSidebarVisible(false);
        input.focus();
    }, "Opening Conversation…");
}

function openNameDialog(kind, conversation = null, restoreSaving = conversationEnabled.checked) {
    if (locked() || nameDialog.open) return;
    dialogAction = { kind, id: conversation?.id, restoreSaving };
    dialogTitle.textContent = kind === "rename" ? "Rename Conversation" : "New Conversation";
    dialogSave.textContent = kind === "rename" ? "Save Name" : "Create Conversation";
    conversationTitle.value = conversation?.title || "";
    dialogError.hidden = true;
    nameDialog.showModal();
    conversationTitle.focus();
    if (kind === "rename") conversationTitle.select();
}

function cancelNameDialog() {
    if (isSending || !dialogAction) return;
    conversationEnabled.checked = dialogAction.restoreSaving;
    dialogAction = null;
    nameDialog.close();
    updateConversationControls();
}

async function submitConversationName(event) {
    event.preventDefault();
    if (isSending || !dialogAction) return;
    const title = conversationTitle.value.trim();
    if (!title || Array.from(title).length > 120) {
        dialogError.textContent = "Enter a Name Between 1 and 120 Characters.";
        dialogError.hidden = false;
        conversationTitle.focus();
        return;
    }
    const action = dialogAction;
    setBusy(true, action.kind === "rename" ? "Renaming Conversation…" : "Creating Conversation…");
    dialogError.hidden = true;
    try {
        const url = action.kind === "rename"
            ? "/v1/conversations/" + requireConversationId(action.id) + "/title" : "/v1/conversations";
        const data = await postJSON(url, { title });
        const conversation = validateConversationMetadata(data.conversation);
        if (action.kind === "rename" && conversation.id !== action.id) {
            throw new Error("The Server Returned a Different Conversation.");
        }
        savedConversationMetadata.set(conversation.id, conversation);
        memoryAvailable = true;
        historyLoaded = true;
        historyError = "";
        if (action.kind === "new") {
            activeConversationId = conversation.id;
            activeConversationTitle = conversation.title;
            conversationEnabled.checked = true;
            conversationSearch.value = "";
            resetVisibleChat();
        } else if (activeConversationId === conversation.id) activeConversationTitle = conversation.title;
        dialogAction = null;
        nameDialog.close();
        renderConversationList();
        setConversationNotice(action.kind === "rename"
            ? "Renamed to “" + conversation.title + "”." : "Created “" + conversation.title + "”.");
        if (action.kind === "new") {
            if (narrowLayout.matches) setConversationSidebarVisible(false);
            input.focus();
        } else {
            if (narrowLayout.matches && !conversationPanel.hidden) {
                conversationSearch.focus();
            } else {
                sessionTitle.tabIndex = -1;
                sessionTitle.focus();
            }
        }
    } catch (error) {
        dialogError.textContent = error.message;
        dialogError.hidden = false;
    } finally {
        setBusy(false);
    }
}

async function deleteConversation(id) {
    if (locked() || nameDialog.open) return;
    const conversation = savedConversationMetadata.get(id);
    if (!conversation) return;
    if (!window.confirm("Delete “" + conversation.title + "” and All Its Saved Messages?")) return;
    await runConversationAction(async () => {
        requireConversationId(id);
        const data = await requestJSON("/v1/conversations/" + id, "DELETE");
        if (data.deleted !== true || data.conversation_id !== id) throw new Error("The Server Did Not Confirm Deletion.");
        savedConversationMetadata.delete(id);
        if (activeConversationId === id) {
            startIndependentConversation();
            conversationEnabled.checked = memoryAvailable !== false;
            updateConversationControls();
        }
        renderConversationList();
        setConversationNotice("Deleted “" + conversation.title + "”.");
    }, "Deleting Conversation…");
}

conversationNew.addEventListener("click", () => {
    if (locked() || !confirmViewReplacement()) return;
    if (conversationEnabled.checked) openNameDialog("new");
    else startIndependentConversation();
});
savedDarkTheme.addEventListener("change", updateConversationControls);
conversationRefresh.addEventListener("click", refreshConversations);
conversationSearch.addEventListener("input", renderConversationList);
conversationRename.addEventListener("click", () => {
    sessionActions.open = false;
    const conversation = savedConversationMetadata.get(activeConversationId);
    if (conversation) openNameDialog("rename", conversation);
});
conversationDelete.addEventListener("click", () => {
    sessionActions.open = false;
    return deleteConversation(activeConversationId);
});
conversationEnabled.addEventListener("change", () => {
    if (locked()) {
        conversationEnabled.checked = !conversationEnabled.checked;
        return;
    }

    const requestedSaving = conversationEnabled.checked;

    if (!confirmViewReplacement()) {
        conversationEnabled.checked = !requestedSaving;
        updateConversationControls();
        return;
    }

    if (requestedSaving) {
        openNameDialog("new", null, false);
    } else {
        startIndependentConversation();
    }

    updateConversationControls();
});
nameForm.addEventListener("submit", submitConversationName);
dialogCancel.addEventListener("click", cancelNameDialog);
nameDialog.addEventListener("cancel", (event) => {
    event.preventDefault();
    cancelNameDialog();
});
sidebarToggle.addEventListener("click", () => setConversationSidebarVisible(conversationPanel.hidden));
sidebarClose.addEventListener("click", () => setConversationSidebarVisible(false));
sidebarBackdrop.addEventListener("click", () => setConversationSidebarVisible(false));
narrowLayout.addEventListener("change", (event) => setConversationSidebarVisible(!event.matches));

document.addEventListener("click", (event) => {
    const summary = event.target.closest(".context-actions > summary");
    if (summary && locked()) event.preventDefault();
    for (const menu of document.querySelectorAll(".context-actions[open]")) {
        if (!menu.contains(event.target)) menu.open = false;
    }
});
document.addEventListener("keydown", (event) => {
    if (nameDialog.open) return;
    if (event.key === "Escape") {
        for (const menu of document.querySelectorAll(".context-actions[open]")) menu.open = false;
        if (narrowLayout.matches && !conversationPanel.hidden) setConversationSidebarVisible(false);
    }
    if (event.key === "Tab" && narrowLayout.matches && !conversationPanel.hidden) {
        const focusable = Array.from(conversationPanel.querySelectorAll(
            "button:not([disabled]), input:not([disabled]), summary"
        )).filter((element) => element.tabIndex >= 0 && element.getClientRects().length > 0);
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (event.shiftKey && document.activeElement === first) {
            event.preventDefault();
            last.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
            event.preventDefault();
            first.focus();
        }
    }
});
conversationDiscard.addEventListener("click", () => {
    if (durableJobs?.pending) {
        cancelDurableRequest();
        return;
    }
    if (isSending || pendingRequestId === null) return;
    for (const panel of chatBox.querySelectorAll(".skill-selection")) panel.remove();
    pendingRequestId = null;
    hasUnrecordedRequest = true;
    setConversationNotice("Pending Request Discarded in This Browser. No Exchange Was Saved; The Server Selection Expires Automatically.");
    updateConversationControls();
    input.focus();
});
inputForm.addEventListener("submit", (event) => {
    event.preventDefault();
    sendPrompt();
});
input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        inputForm.requestSubmit();
    }
});

async function sendPrompt() {
    if (locked() || nameDialog.open) return;
    const prompt = input.value.trim();
    if (!prompt) {
        input.focus();
        return;
    }
    if (conversationEnabled.checked && activeConversationId === null) {
        openNameDialog("new");
        return;
    }
    setBusy(true, "Processing Request…");
    setConversationNotice("");
    addMessage("user", "Analyst", prompt);
    input.value = "";
    const loading = addMessage(
        "assistant", "Corveus", "Processing Request…"
    );

    try {
        const enabled = durableJobs ? await durableJobs.enabled() : false;
        if (enabled) {
            loading.remove();
            await runDurableAction(() => durableJobs.start(
                prompt, activeConversationId, showDurableStatus
            ));
        } else {
            const payload = {
                messages: [{ role: "user", content: prompt }]
            };
            if (activeConversationId !== null) {
                payload.conversation_id = activeConversationId;
            }
            const data = await postJSON("/v1/chat/completions", payload);
            loading.remove();
            handleResponse(data);
        }
    } catch (error) {
        loading.remove();
        hasUnrecordedRequest = true;
        addMessage("error", "[ERROR]:", error.message);
        input.value = prompt;
    } finally {
        setBusy(false);
        if (durableJobs?.pending) {
            requestStatus.textContent = "Request Pending — Review or Check Status";
        }
        input.focus();
    }
}

function handleResponse(data) {
    if (data.selection_required) {
        renderSkillSelection(data);
        return;
    }
    const reply = data.choices?.[0]?.message?.content;
    if (typeof reply !== "string" || !reply.trim()) throw new Error("The Server Returned No Assistant Response.");
    updateStats(data.stats);
    addMessage("assistant", "Corveus", reply);
    recordResponseUsage(data);
    if (activeConversationId !== null) {
        if (data.conversation_id !== activeConversationId || data.memory_persistence?.saved !== true) {
            saveWarning = true;
            const warning = typeof data.memory_persistence?.message === "string"
                ? data.memory_persistence.message : "The Server Did Not Confirm That This Exchange Was Saved.";
            addMessage("error", "[SAVE WARNING]:", warning);
            setConversationNotice(warning, true);
        } else {
            // Preserve warnings for any earlier failed save in this view.
            setConversationNotice(saveWarning
                ? "This Exchange Was Saved. An Earlier Exchange in This View Was Not Saved."
                : "The Completed Exchange Was Saved.", saveWarning);
        }
    }
    updateConversationControls();
}

setConversationSidebarVisible(!narrowLayout.matches, false);
updateConversationControls();
refreshConversations();

function renderSkillSelection(data) {
    if (typeof data.request_id !== "string" || !data.request_id.trim() ||
        !Array.isArray(data.skills) || data.skills.some((skill) =>
            !skill || typeof skill.name !== "string" || !skill.name.trim())) {
        throw new Error("The Server Returned an Invalid Skill Selection.");
    }
    pendingRequestId = data.request_id;
    updateConversationControls();
    const panel = document.createElement("section");
    panel.className = "message assistant skill-selection";

    const heading = document.createElement("strong");
    heading.textContent = "[Context Budget — Skill Selection]";
    panel.appendChild(heading);

    const summary = document.createElement("p");
    summary.textContent =
        `Input: ${Number(data.input_tokens).toLocaleString()} Tokens · ` +
        `Budget: ${Number(data.input_budget).toLocaleString()} Tokens · ` +
        `Context: ${Number(data.context_size).toLocaleString()} Tokens`;

    panel.appendChild(summary);

    const explanation = document.createElement("p");
    explanation.textContent =
        data.message ||
        "Select One or More Skills to Remove, Then Continue.";

    panel.appendChild(explanation);

    const form = document.createElement("form");
    form.className = "skill-selection-form";

    const selectedSkills = data.skills || [];

    if (selectedSkills.length === 0) {
        const notice = document.createElement("p");
        notice.textContent =
            "No Removable Skills Remain. The Prompt May Be Too Large Even Without Skills.";
        panel.appendChild(notice);
        chatBox.appendChild(panel);
        chatBox.scrollTop = chatBox.scrollHeight;
        return;
    }

    for (const skill of selectedSkills) {
        const label = document.createElement("label");
        label.className = "skill-option";

        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.name = "remove_skill";
        checkbox.value = skill.name;

        const details = document.createElement("span");
        details.className = "skill-details";

        const name = document.createElement("span");
        name.className = "skill-name";
        name.textContent = skill.name;

        const metadata = document.createElement("span");
        metadata.className = "skill-metadata";
        metadata.textContent =
            `Score: ${Number(skill.score).toFixed(3)} · ` +
            `${Number(skill.characters).toLocaleString()} Characters`;

        details.append(name, metadata);
        label.append(checkbox, details);
        form.appendChild(label);
    }

    const error = document.createElement("p");
    error.className = "selection-error";
    error.hidden = true;

    const submit = document.createElement("button");
    submit.type = "submit";
    submit.className = "primary-button";
    submit.textContent = "Remove Selected Skills";

    form.append(error, submit);

    form.addEventListener("submit", async (event) => {
        event.preventDefault();

        if (isSending || pendingRequestId !== data.request_id) return;

        const removeNames = Array.from(
            form.querySelectorAll('input[name="remove_skill"]:checked')
        ).map((checkbox) => checkbox.value);

        if (removeNames.length === 0) {
            error.textContent = "Select at Least One Skill to Remove.";
            error.hidden = false;
            return;
        }

        setBusy(true, "Rechecking Context Budget…");
        submit.disabled = true;
        error.hidden = true;

        const status = document.createElement("p");
        status.textContent = "Rechecking Context Budget…";
        form.appendChild(status);

        try {
            const result = await postJSON("/v1/chat/selection", {
                request_id: data.request_id,
                remove_skill_names: removeNames
            });

            pendingRequestId = null;
            handleResponse(result);
            panel.remove();

        } catch (err) {
            if (err.status === 404 || err.status === 400 || pendingRequestId === null) {
                panel.remove();
                pendingRequestId = null;
                hasUnrecordedRequest = true;
                setConversationNotice(err.message, true);
                return;
            }
            error.textContent = err.message;
            error.hidden = false;
            submit.disabled = false;
            status.remove();
        } finally {
            setBusy(false);
            if (pendingRequestId === null) input.focus();
        }
    });

    panel.appendChild(form);
    chatBox.appendChild(panel);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function addMessage(type, label, content) {
    const message = document.createElement("div");
    message.className = `message ${type}`;

    const heading = document.createElement("strong");
    heading.textContent = label;

    const text = document.createElement("span");
    text.textContent = content;

    message.append(heading, text);
    chatBox.appendChild(message);
    chatBox.scrollTop = chatBox.scrollHeight;

    return message;
}

function validTokenCount(value) {
    return Number.isSafeInteger(value) && value >= 0 ? value : null;
}

function emptyUsageSummary() {
    const summary = { exchange_count: 0, recorded_exchanges: 0, latest_usage: null, is_partial: false };
    for (const field of ["input_tokens", "output_tokens", "total_tokens"]) {
        summary[field] = 0;
        summary[field + "_measured_exchanges"] = 0;
    }
    return summary;
}

function statsCounts(stats) {
    return {
        input_tokens: validTokenCount(stats?.input_tokens),
        output_tokens: validTokenCount(stats?.output_tokens),
        total_tokens: validTokenCount(stats?.total_tokens) ?? validTokenCount(stats?.tokens_used)
    };
}

function validUsageSummary(summary, conversationId) {
    if (!summary || summary.conversation_id !== conversationId ||
        typeof summary.is_partial !== "boolean" ||
        validTokenCount(summary.exchange_count) === null ||
        validTokenCount(summary.recorded_exchanges) === null ||
        summary.recorded_exchanges > summary.exchange_count) return false;
    for (const field of ["input_tokens", "output_tokens", "total_tokens"]) {
        if (summary[field] !== null && validTokenCount(summary[field]) === null) return false;
        const measured = validTokenCount(summary[field + "_measured_exchanges"]);
        if (measured === null || measured > summary.exchange_count) return false;
    }
    if (summary.latest_usage !== null) {
        const latest = summary.latest_usage;
        if (!latest || typeof latest.model_name !== "string" ||
            !latest.model_name.trim() || !validTokenCount(latest.context_size)) return false;
        for (const field of ["input_tokens", "output_tokens", "total_tokens"]) {
            if (latest[field] !== null && validTokenCount(latest[field]) === null) return false;
        }
    }
    return true;
}

function restoreConversationUsage(summary, conversationId, restoreLatest = true) {
    if (!validUsageSummary(summary, conversationId)) {
        usageSummaryAvailable = false;
        usageSummaryNote = "Conversation Totals Are Unavailable.";
        if (restoreLatest) updateStats(null);
    } else {
        currentUsageSummary = summary;
        usageSummaryAvailable = true;
        usageSummaryNote = "";
        if (restoreLatest) {
            const latest = summary.latest_usage;
            updateStats(latest ? {
                ...latest, ctx_size: latest.context_size, tokens_used: latest.total_tokens
            } : null);
        }
    }
    renderConversationUsage();
}

function recordResponseUsage(data) {
    if (activeConversationId !== null) {
        if (data.memory_persistence?.saved === true && data.conversation_id === activeConversationId) {
            restoreConversationUsage(data.conversation_usage, activeConversationId, false);
        } else {
            usageSummaryNote = "Latest Request Not Saved; Totals Cover Saved Exchanges Only.";
            renderConversationUsage();
        }
        return;
    }
    localUsageRecords.push(statsCounts(data.stats));
    const summary = emptyUsageSummary();
    summary.exchange_count = localUsageRecords.length;
    summary.recorded_exchanges = localUsageRecords.filter(
        (record) => Object.values(record).some((value) => value !== null)
    ).length;
    for (const field of ["input_tokens", "output_tokens", "total_tokens"]) {
        const values = localUsageRecords.map((record) => record[field]).filter((value) => value !== null);
        let sum = values.length ? 0 : null;
        for (const value of values) {
            sum = sum === null ? null : validTokenCount(sum + value);
        }
        summary[field] = sum;
        summary[field + "_measured_exchanges"] = values.length;
    }
    summary.is_partial = ["input_tokens", "output_tokens", "total_tokens"].some(
        (field) => summary[field + "_measured_exchanges"] < summary.exchange_count
    );
    currentUsageSummary = summary;
    usageSummaryAvailable = true;
    renderConversationUsage();
}

function renderConversationUsage() {
    renderConversationProgress();
    document.getElementById("inspector-usage-total").textContent =
        document.getElementById("tokens-used").textContent + " Total Tokens";
    document.getElementById("inspector-request-model").textContent = document.getElementById("metrics-model").textContent;
    document.getElementById("inspector-request-tokens").textContent =
        "Input " + document.getElementById("request-input-tokens").textContent +
        " · Output " + document.getElementById("request-output-tokens").textContent;
    document.getElementById("conversation-usage-label").textContent =
        activeConversationId === null ? "Session Usage" : "Saved Conversation Usage";
    for (const [field, id] of [
        ["input_tokens", "conversation-input-tokens"],
        ["output_tokens", "conversation-output-tokens"],
        ["total_tokens", "conversation-total-tokens"]
    ]) {
        const value = usageSummaryAvailable ? currentUsageSummary[field] : null;
        const partial = currentUsageSummary[field + "_measured_exchanges"] < currentUsageSummary.exchange_count;
        document.getElementById(id).textContent = value === null ? "--"
            : (partial ? "≥ " : "") + value.toLocaleString();
    }
    const coverage = document.getElementById("usage-coverage");
    if (!usageSummaryAvailable) {
        coverage.textContent = usageSummaryNote || "Conversation Totals Are Unavailable.";
        document.getElementById("inspector-usage-coverage").textContent = coverage.textContent;
        return;
    }
    const summary = currentUsageSummary;
    const text = summary.exchange_count === 0 ? "0 Completed Exchanges"
        : summary.total_tokens_measured_exchanges + " of " + summary.exchange_count +
            " Exchanges Have Total-Token Measurements." +
            (summary.is_partial ? " Missing Measurements Are Unavailable; Partial Totals Are Lower Bounds." : "");
    coverage.textContent = text + (usageSummaryNote ? " " + usageSummaryNote : "");
    document.getElementById("inspector-usage-coverage").textContent = coverage.textContent;
}

function updateStats(stats) {
    latestRequestStats = stats && typeof stats === "object" ? stats : null;
    const counts = statsCounts(latestRequestStats);
    document.getElementById("request-input-tokens").textContent =
        counts.input_tokens === null ? "--" : counts.input_tokens.toLocaleString();
    document.getElementById("request-output-tokens").textContent =
        counts.output_tokens === null ? "--" : counts.output_tokens.toLocaleString();
    document.getElementById("metrics-model").textContent =
        "Latest Model: " + (typeof latestRequestStats?.model_name === "string" && latestRequestStats.model_name.trim()
            ? latestRequestStats.model_name : "--");
    renderConversationProgress();
}

function conversationReferenceCapacity() {
    // Saved totals use the latest persisted snapshot. A failed save must not
    // change the denominator of the saved conversation's comparison.
    const value = activeConversationId === null
        ? latestRequestStats?.ctx_size : currentUsageSummary.latest_usage?.context_size;
    const capacity = validTokenCount(value);
    return capacity !== null && capacity > 0 ? capacity : null;
}

function renderConversationProgress() {
    const tokens = usageSummaryAvailable ? validTokenCount(currentUsageSummary.total_tokens) : null;
    const capacity = usageSummaryAvailable ? conversationReferenceCapacity() : null;
    const partial = tokens !== null &&
        currentUsageSummary.total_tokens_measured_exchanges < currentUsageSummary.exchange_count;
    const percent = tokens !== null && capacity !== null ? tokens / capacity * 100 : null;
    const prefix = partial ? "≥ " : "";
    const track = document.getElementById("context-progress");
    const percentElement = document.getElementById("context-percent");
    document.getElementById("tokens-used").textContent = tokens === null ? "--"
        : prefix + tokens.toLocaleString();
    document.getElementById("context-size").textContent = capacity === null ? "--" : capacity.toLocaleString();
    percentElement.textContent = percent === null ? "--"
        : prefix + percent.toLocaleString(undefined, { maximumFractionDigits: 1 }) + "%";
    percentElement.classList.toggle("context-warning", percent !== null && percent >= 80 && percent < 100);
    percentElement.classList.toggle("context-critical", percent !== null && percent >= 100);
    document.getElementById("context-progress-fill").style.width =
        (percent === null ? 0 : Math.min(100, Math.max(0, percent))) + "%";
    track.dataset.level = percent !== null && percent >= 100 ? "critical"
        : percent !== null && percent >= 80 ? "warning" : "normal";
    if (percent === null) {
        track.removeAttribute("aria-valuenow");
        track.setAttribute("aria-valuetext", "Cumulative Usage or Context Reference Unavailable");
    } else {
        track.setAttribute("aria-valuenow", String(Math.min(100, percent)));
        track.setAttribute("aria-valuetext", (partial ? "At Least " : "") +
            tokens.toLocaleString() + " Cumulative Tokens Compared With " +
            capacity.toLocaleString() + " Reference Tokens; Not Current Context Occupancy");
    }
    document.getElementById("context-remaining").textContent = tokens === null
        ? "Cumulative Usage Unavailable"
        : capacity === null ? "Context Reference Unavailable"
            : (partial ? "Measured Usage Only · " : "") +
              (tokens > capacity ? "Reference Size Exceeded by " + (tokens - capacity).toLocaleString()
                  : (capacity - tokens).toLocaleString() + " Tokens Below Reference Size");
}
function clearDurablePanels() {
    for (const panel of chatBox.querySelectorAll(".durable-controls")) panel.remove();
    for (const panel of chatBox.querySelectorAll(".durable-review")) panel.remove();
}

function showDurableStatus(job) {
    const text = job.cancel_requested
        ? "Cancellation Requested — Waiting for Execution to Stop"
        : "Request: " + job.state.replaceAll("_", " ");
    requestStatus.textContent = text;
    const status = chatBox.querySelector(".durable-status");
    if (status) status.textContent = text;
}

function renderDurableControls() {
    const panel = document.createElement("section");
    panel.className = "message assistant durable-controls";
    panel.setAttribute("aria-label", "Pending request controls");

    const status = document.createElement("p");
    status.className = "durable-status";
    status.setAttribute("role", "status");
    status.textContent = "Checking Request…";

    const resume = document.createElement("button");
    resume.type = "button";
    resume.className = "secondary-button";
    resume.textContent = "Check Existing Request";
    resume.disabled = true;
    resume.addEventListener("click", () => {
        if (!isSending && durableJobs.pending) {
            runDurableAction(() => durableJobs.resume(showDurableStatus));
        }
    });

    const cancel = document.createElement("button");
    cancel.type = "button";
    cancel.className = "secondary-button";
    cancel.textContent = "Cancel Request";
    cancel.addEventListener("click", () => cancelDurableRequest(cancel));

    panel.append(status, resume, cancel);
    chatBox.appendChild(panel);
    return { resume, status };
}

async function cancelDurableRequest(button = null) {
    if (!durableJobs?.pending || durableCancelRunning) return;
    durableCancelRunning = true;
    if (button) button.disabled = true;
    try {
        await durableJobs.cancel();
        setConversationNotice(
            "Cancellation Requested. The Request Remains Tracked Until Its Status Is Confirmed."
        );
        if (!durableActionRunning && durableJobs.pending) {
            await runDurableAction(() => durableJobs.resume(showDurableStatus));
        }
    } catch (error) {
        setConversationNotice(
            error.message + " Check the Existing Request Before Sending Another.",
            true
        );
    } finally {
        durableCancelRunning = false;
        if (button) button.disabled = false;
    }
}

async function displayDurableResult(data) {
    const usage = data.usage;
    let conversationUsage = null;
    if (data.memory_persistence === "saved" && data.job.conversation_id) {
        try {
            const totals = await requestJSON(
                `/v1/conversations/${data.job.conversation_id}/usage`
            );
            conversationUsage = totals.usage;
        } catch {
            // An unavailable usage summary must not hide a completed answer.
        }
    }
    handleResponse({
        choices: [{ message: { content: data.answer } }],
        stats: usage ? {
            ...usage,
            ctx_size: usage.context_size,
            tokens_used: usage.total_tokens
        } : null,
        conversation_id: data.job.conversation_id,
        conversation_usage: conversationUsage,
        memory_persistence: {
            saved: data.memory_persistence === "saved",
            message: data.memory_persistence === "saved"
                ? "The Exchange Was Saved."
                : "The Exchange Was Not Saved to Conversation History."
        },
        selection_required: false
    });
}

function renderDurableReview(data) {
    const review = data.review;
    const panel = document.createElement("section");
    panel.className = "message assistant skill-selection durable-review";

    const heading = document.createElement("strong");
    heading.textContent = "Context Budget — Skill Review";

    const summary = document.createElement("p");
    summary.textContent =
        `Input: ${review.input_tokens} Tokens · Budget: ${review.input_budget} Tokens`;

    const expiry = document.createElement("p");
    expiry.textContent =
        "Review Expires: " + review.expires_at +
        ". Select Skills to Remove; Reviewing Does Not Extend the Deadline.";

    const form = document.createElement("form");
    form.className = "skill-selection-form";

    for (const skill of review.skills) {
        const label = document.createElement("label");
        label.className = "skill-option";

        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.name = "remove_skill";
        checkbox.value = skill.name;

        const details = document.createElement("span");
        details.className = "skill-details";
        details.textContent =
            `${skill.name} · ${skill.characters} Characters`;

        label.append(checkbox, details);
        form.appendChild(label);
    }

    const error = document.createElement("p");
    error.className = "selection-error";
    error.hidden = true;

    const submit = document.createElement("button");
    submit.type = "submit";
    submit.className = "primary-button";
    submit.textContent = "Remove Selected Skills";

    form.addEventListener("submit", (event) => {
        event.preventDefault();
        if (isSending || !durableJobs.pending) return;
        const names = Array.from(
            form.querySelectorAll('input[name="remove_skill"]:checked')
        ).map((checkbox) => checkbox.value);
        if (!names.length) {
            error.textContent = "Select at Least One Skill to Remove.";
            error.hidden = false;
            return;
        }
        runDurableAction(() => durableJobs.review(
            data.job.version, names, showDurableStatus
        ));
    });

    form.append(error, submit);
    panel.append(heading, summary, expiry, form);
    chatBox.appendChild(panel);
    chatBox.scrollTop = chatBox.scrollHeight;
}

async function runDurableAction(action) {
    if (durableActionRunning) return;
    durableActionRunning = true;
    pendingRequestId = durableJobs.jobId || "durable-pending";
    clearDurablePanels();
    setBusy(true, "Checking Request…");
    const controls = renderDurableControls();
    let awaitingReview = false;

    try {
        const data = await action();
        if (data.review) {
            awaitingReview = true;
            pendingRequestId = data.job.job_id;
            renderDurableReview(data);
        } else {
            await displayDurableResult(data);
        }
    } catch (error) {
        hasUnrecordedRequest = true;
        addMessage("error", "[REQUEST STATUS]:", error.message);
        setConversationNotice(
            durableJobs.pending
                ? "The Request Remains Tracked. Check Its Status Before Retrying."
                : error.message,
            true
        );
    } finally {
        durableActionRunning = false;
        pendingRequestId = durableJobs.pending
            ? durableJobs.jobId || "durable-pending"
            : null;
        if (!durableJobs.pending) clearDurablePanels();
        setBusy(false);
        conversationDiscard.textContent = durableJobs.pending
            ? "Cancel Request" : "Discard Pending Request";
        if (durableJobs.pending) {
            controls.resume.disabled = false;
            controls.status.textContent = awaitingReview
                ? "Awaiting Skill Review"
                : "Status Check Needed — Use Check Existing Request";
            requestStatus.textContent = controls.status.textContent;
        } else {
            input.focus();
        }
    }
}
