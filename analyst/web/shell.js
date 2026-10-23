"use strict";

// Native modal focus containment and Escape behavior for the narrow inspector.
(() => {
    const inspector = document.getElementById("workspace-inspector");
    const trigger = document.getElementById("inspector-toggle");
    const close = document.getElementById("inspector-close");
    const narrow = window.matchMedia("(max-width: 1100px)");
    trigger.addEventListener("click", () => {
        if (narrow.matches && !inspector.open) inspector.showModal();
    });
    close.addEventListener("click", () => inspector.close());
    inspector.addEventListener("click", (event) => {
        if (event.target !== inspector) return;
        const bounds = inspector.getBoundingClientRect();
        if (event.clientX < bounds.left || event.clientX > bounds.right ||
            event.clientY < bounds.top || event.clientY > bounds.bottom) inspector.close();
    });
    inspector.addEventListener("close", () => {
        if (narrow.matches) trigger.focus();
    });
    narrow.addEventListener("change", () => {
        if (inspector.open) inspector.close();
        if (!narrow.matches) inspector.show();
    });
    if (!narrow.matches) inspector.show();
})();

// Poll independently of chat; a busy synchronous API is unknown until checked.
(() => {
    const names = ["analyst", "router", "model", "persistence"];
    function render(states) {
        for (const name of names) {
            const element = document.getElementById(`status-${name}`);
            const state = ["online", "offline", "disabled"].includes(states[name]) ? states[name] : "unknown";
            element.dataset.state = state;
            element.querySelector(".status-label").textContent = {
                online: "Online", offline: "Offline", disabled: "Disabled", unknown: "Unavailable"
            }[state];
        }
    }
    async function refresh() {
        try {
            // The API serves requests synchronously; avoid queueing checks behind inference.
            if (typeof isSending !== "undefined" && isSending) return;
            const response = await fetch("/v1/status", { cache: "no-store", signal: AbortSignal.timeout(5000) });
            if (!response.ok) throw new Error("Status unavailable");
            const data = await response.json();
            render(data.components || {});
            document.getElementById("inspector-model").textContent =
                typeof data.model?.name === "string" && data.model.name.trim() ? data.model.name : "Unavailable";
            document.getElementById("inspector-runtime").textContent =
                typeof data.model?.runtime === "string" ? data.model.runtime : "Unavailable";
            document.getElementById("inspector-connection").textContent =
                data.components?.model === "online" ? "Online" : "Offline";
        } catch {
            // Do not claim the model or storage stopped when the API cannot be reached.
            render({ analyst: "offline" });
            document.getElementById("inspector-model").textContent = "Unavailable";
            document.getElementById("inspector-runtime").textContent = "Unavailable";
            document.getElementById("inspector-connection").textContent = "Unavailable";
        } finally {
            window.setTimeout(refresh, 15000);
        }
    }
    refresh();
})();
