"use strict";

// Request state stays in this tab; never write prompts to browser storage.
window.createJobClient = function () {
    let current = null;
    let busy = false;

    async function request(path, method = "GET", payload, headers = {}) {
        const options = {
            method,
            cache: "no-store",
            headers: { ...headers }
        };
        if (payload !== undefined) {
            options.headers["Content-Type"] = "application/json";
            options.body = JSON.stringify(payload);
        }
        const response = await fetch(path, options);
        let data;
        try {
            data = await response.json();
        } catch {
            throw new Error(
                "Unreadable server response. Check the existing request before retrying."
            );
        }
        if (!data || typeof data !== "object" || Array.isArray(data)) {
            throw new Error("Invalid server response.");
        }
        if (!response.ok) {
            const error = new Error(
                data.error?.message || `Request failed (HTTP ${response.status}).`
            );
            error.status = response.status;
            error.code = data.error?.code;
            throw error;
        }
        return data;
    }

    function requireJob(job) {
        if (
            !job || !/^[0-9a-f]{32}$/.test(job.job_id) ||
            !Number.isSafeInteger(job.version) || job.version < 1 ||
            typeof job.state !== "string"
        ) {
            throw new Error("Invalid job status returned by the server.");
        }
        if (current?.id && current.id !== job.job_id) {
            throw new Error("The server returned a different job.");
        }
        return job;
    }

    async function exclusive(action) {
        if (busy) throw new Error("This request is already being checked.");
        busy = true;
        try {
            return await action();
        } finally {
            busy = false;
        }
    }

     async function submitExisting() {
        if (!current) throw new Error("No pending request.");
        if (!current.id) {
            let data;
            try {
                data = await request(
                    "/v1/jobs",
                    "POST",
                    current.payload,
                    { "Idempotency-Key": current.key }
                );
            } catch (error) {
                // Rejected or expired requests cannot be resumed.
                // Preserve uncertain submissions and temporary failures.
                if ([400, 404, 410].includes(error.status)) {
                    current = null;
                }
                throw error;
            }
            current.id = requireJob(data.job).job_id;
        }
    }

    async function follow(onStatus = () => {}) {
        await submitExisting();
        const path = `/v1/jobs/${current.id}`;

        while (true) {
            const { job: rawJob } = await request(path);
            const job = requireJob(rawJob);
            onStatus(job);

            if (job.state === "awaiting_review") {
                const data = await request(`${path}/review`);
                requireJob(data.job);
                if (
                    !data.review ||
                    !Array.isArray(data.review.skills) ||
                    !Number.isSafeInteger(data.review.input_tokens) ||
                    !Number.isSafeInteger(data.review.input_budget)
                ) {
                    throw new Error("Invalid review returned by the server.");
                }
                return data;
            }

            if (job.state === "completed") {
                const data = await request(`${path}/result`);
                requireJob(data.job);
                if (
                    typeof data.answer !== "string" ||
                    !data.answer.trim()
                ) {
                    throw new Error("The server returned no answer.");
                }
                current = null;
                return data;
            }

            if ([
                "failed", "cancelled", "expired", "interrupted"
            ].includes(job.state)) {
                current = null;
                const error = new Error(
                    job.state === "interrupted"
                        ? "Execution was interrupted. Upstream execution may still be active."
                        : `Request ${job.state}.`
                );
                error.jobState = job.state;
                throw error;
            }

            if (![
                "queued", "preparing", "waiting_inference",
                "running", "finalizing"
            ].includes(job.state)) {
                throw new Error("Unknown job state returned by the server.");
            }

            await new Promise((resolve) => setTimeout(resolve, 1000));
        }
    }

    return {
        async enabled() {
            const data = await request("/v1/job-capabilities");
            if (typeof data.enabled !== "boolean") {
                throw new Error("Invalid job capability response.");
            }
            return data.enabled;
        },

        get pending() {
            return current !== null;
        },

        get jobId() {
            return current?.id || null;
        },

        start(prompt, conversationId, onStatus) {
            return exclusive(async () => {
                if (current) {
                    throw new Error("Check or cancel the existing request first.");
                }
                if (typeof prompt !== "string" || !prompt.trim()) {
                    throw new Error("A prompt is required.");
                }
                const key = window.crypto.randomUUID();
                current = {
                    key,
                    id: null,
                    payload: {
                        prompt,
                        conversation_id: conversationId || null
                    }
                };
                return follow(onStatus);
            });
        },

        resume(onStatus) {
            // An uncertain submission uses its original key and exact payload.
            return exclusive(() => follow(onStatus));
        },

        review(expectedVersion, removeNames, onStatus) {
            return exclusive(async () => {
                if (!current?.id) throw new Error("No pending review.");
                await request(
                    `/v1/jobs/${current.id}/review`,
                    "POST",
                    {
                        expected_version: expectedVersion,
                        remove_skill_names: removeNames
                    }
                );
                return follow(onStatus);
            });
        },

        async cancel() {
            // Cancellation can run while follow() is polling.
            // It does not abandon local tracking or claim inference stopped.
            await submitExisting();
            const path = `/v1/jobs/${current.id}`;
            const { job: rawJob } = await request(path);
            const job = requireJob(rawJob);
            return request(`${path}/cancel`, "POST", {
                expected_version: job.version
            });
        }
    };
};