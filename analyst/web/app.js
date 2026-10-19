"use strict";

let isSending = false;

const input = document.getElementById("prompt-input");
const chatBox = document.getElementById("chat-box");
const inputForm = document.getElementById("input-container");
const sendButton = document.getElementById("send-button");

inputForm.addEventListener("submit", (event) => {
    event.preventDefault();
    sendPrompt();
});

input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        inputForm.requestSubmit();
    }
});

function setBusy(busy) {
    isSending = busy;
    sendButton.disabled = busy;
    input.setAttribute("aria-busy", String(busy));
}

async function postJSON(url, payload) {
    const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    });

    const data = await response.json();

    if (!response.ok) {
        const message = typeof data.error === "string"
            ? data.error
            : data.error?.message || `HTTP ${response.status}`;

        throw new Error(message);
    }

    return data;
}

async function sendPrompt() {
    if (isSending) return;

    const prompt = input.value.trim();

    if (!prompt) {
        input.focus();
        return;
    }

    setBusy(true);

    addMessage("user", "[USER]:", prompt);
    input.value = "";

    const loading = addMessage(
        "assistant",
        "[ANALYST]:",
        "Evaluating skills and checking context budget..."
    );

    try {
        const data = await postJSON("/v1/chat/completions", {
            messages: [{ role: "user", content: prompt }]
        });

        loading.remove();
        handleResponse(data);

    } catch (error) {
        loading.remove();
        addMessage("error", "[ERROR]:", error.message);
    } finally {
        setBusy(false);
        input.focus();
    }
}

function handleResponse(data) {
    if (data.selection_required) {
        renderSkillSelection(data);
        return;
    }

    updateStats(data.stats);

    const reply = data.choices?.[0]?.message?.content;

    if (typeof reply !== "string" || !reply.trim()) {
        throw new Error("The server returned no assistant response.");
    }

    addMessage("assistant", "[ANALYST WORKFLOW]:", reply);
}

function renderSkillSelection(data) {
    const panel = document.createElement("section");
    panel.className = "message assistant skill-selection";

    const heading = document.createElement("strong");
    heading.textContent = "[CONTEXT BUDGET — SKILL SELECTION]";
    panel.appendChild(heading);

    const summary = document.createElement("p");
    summary.textContent =
        `Input: ${Number(data.input_tokens).toLocaleString()} tokens · ` +
        `Budget: ${Number(data.input_budget).toLocaleString()} tokens · ` +
        `Context: ${Number(data.context_size).toLocaleString()} tokens`;

    panel.appendChild(summary);

    const explanation = document.createElement("p");
    explanation.textContent =
        data.message ||
        "Select one or more skills to remove, then continue.";

    panel.appendChild(explanation);

    const form = document.createElement("form");
    form.className = "skill-selection-form";

    const selectedSkills = data.skills || [];

    if (selectedSkills.length === 0) {
        const notice = document.createElement("p");
        notice.textContent =
            "No removable skills remain. The prompt may be too large even without skills.";
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
            `${Number(skill.characters).toLocaleString()} characters`;

        details.append(name, metadata);
        label.append(checkbox, details);
        form.appendChild(label);
    }

    const error = document.createElement("p");
    error.className = "selection-error";
    error.hidden = true;

    const submit = document.createElement("button");
    submit.type = "submit";
    submit.textContent = "REMOVE SELECTED SKILLS";

    form.append(error, submit);

    form.addEventListener("submit", async (event) => {
        event.preventDefault();

        if (isSending) return;

        const removeNames = Array.from(
            form.querySelectorAll('input[name="remove_skill"]:checked')
        ).map((checkbox) => checkbox.value);

        if (removeNames.length === 0) {
            error.textContent = "Select at least one skill to remove.";
            error.hidden = false;
            return;
        }

        setBusy(true);
        submit.disabled = true;
        error.hidden = true;

        const status = document.createElement("p");
        status.textContent = "Rechecking context budget...";
        form.appendChild(status);

        try {
            const result = await postJSON("/v1/chat/selection", {
                request_id: data.request_id,
                remove_skill_names: removeNames
            });

            panel.remove();
            handleResponse(result);

        } catch (err) {
            error.textContent = err.message;
            error.hidden = false;
            submit.disabled = false;
            status.remove();
        } finally {
            setBusy(false);
            sendButton.disabled = false;
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

function updateStats(stats) {
    if (!stats) return;

    const tokensElement = document.getElementById("tokens-used");
    const contextElement = document.getElementById("context-size");
    const percentElement = document.getElementById("context-percent");

    const tokens = Number(stats.tokens_used);
    const contextSize = Number(stats.ctx_size);

    const validTokens =
        stats.tokens_used != null &&
        stats.tokens_used !== "" &&
        Number.isFinite(tokens) &&
        tokens >= 0;

    const validContext =
        stats.ctx_size != null &&
        stats.ctx_size !== "" &&
        Number.isFinite(contextSize) &&
        contextSize > 0;

    tokensElement.textContent = validTokens
        ? tokens.toLocaleString()
        : "--";

    contextElement.textContent = validContext
        ? contextSize.toLocaleString()
        : "--";

    if (validTokens && validContext) {
        const percentage = (tokens / contextSize) * 100;

        percentElement.textContent = `${percentage.toFixed(1)}%`;

        percentElement.classList.toggle(
            "context-warning",
            percentage >= 80 && percentage < 100
        );

        percentElement.classList.toggle(
            "context-critical",
            percentage >= 100
        );
    } else {
        percentElement.textContent = "--";
        percentElement.classList.remove(
            "context-warning",
            "context-critical"
        );
    }
}