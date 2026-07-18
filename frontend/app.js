/* ==========================================================================
   Noor Al-Hidayah Javascript Application Logic
   ========================================================================== */

const API_BASE_URL = "http://localhost:8000/api/v1";

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const chatForm = document.getElementById("chatForm");
    const chatInput = document.getElementById("chatInput");
    const messageStream = document.getElementById("messageStream");
    const welcomeState = document.getElementById("welcomeState");
    const mockupCard = document.getElementById("mockupCard");
    const themeSwitch = document.getElementById("themeSwitch");
    document.body.classList.add("light-theme");
    themeSwitch.checked = false;
    const retrieveModeBtn = document.getElementById("retrieveModeBtn");
    const suggestionChips = document.querySelectorAll(".suggestion-chips .chip-btn");
    const leftColumn = document.querySelector(".left-column");
    const leftSidebarToggle = document.getElementById("leftSidebarToggle");
    const leftSidebarOverlay = document.getElementById("leftSidebarOverlay");

    // 0. Mobile left-sidebar toggle
    function openLeftSidebar() {
        leftColumn.classList.add("open");
        leftSidebarOverlay.classList.add("open");
    }
    function closeLeftSidebar() {
        leftColumn.classList.remove("open");
        leftSidebarOverlay.classList.remove("open");
    }
    leftSidebarToggle.addEventListener("click", () => {
        if (leftColumn.classList.contains("open")) {
            closeLeftSidebar();
        } else {
            openLeftSidebar();
        }
    });
    leftSidebarOverlay.addEventListener("click", closeLeftSidebar);

    let retrieveOnlySession = false; // true = call /retrieve instead of /ask

    // 1. Theme Toggle implementation
    themeSwitch.addEventListener("change", (e) => {
        if (!e.target.checked) {
            document.body.classList.add("light-theme");
        } else {
            document.body.classList.remove("light-theme");
        }
    });

    // 2. Toggle retrieve-only mode
    retrieveModeBtn.addEventListener("click", () => {
        retrieveOnlySession = !retrieveOnlySession;
        if (retrieveOnlySession) {
            retrieveModeBtn.classList.add("active");
            chatInput.placeholder = "استرجاع الأحاديث والبحث فقط...";
        } else {
            retrieveModeBtn.classList.remove("active");
            chatInput.placeholder = "اسألني عن الإسلام...";
        }
    });

    // 3. Setup suggestion chips action
    suggestionChips.forEach(chip => {
        chip.addEventListener("click", () => {
            const query = chip.getAttribute("data-query");
            if (query) {
                chatInput.value = query;
                submitMessage();
            }
        });
    });

    // 4. Form Submit handler
    chatForm.addEventListener("submit", (e) => {
        e.preventDefault();
        submitMessage();
    });

    // 5. Message action buttons (like / dislike / regenerate / copy) via event delegation
    messageStream.addEventListener("click", async (e) => {
        const btn = e.target.closest(".msg-action-btn");
        if (!btn) return;

        const action = btn.dataset.action;
        const bubble = btn.closest(".chat-bubble");

        if (action === "like") {
            const wasActive = btn.classList.contains("active");
            const dislikeBtn = bubble.querySelector('.msg-action-btn[data-action="dislike"]');
            if (dislikeBtn) dislikeBtn.classList.remove("active");
            btn.classList.toggle("active", !wasActive);
        }

        if (action === "dislike") {
            const wasActive = btn.classList.contains("active");
            const likeBtn = bubble.querySelector('.msg-action-btn[data-action="like"]');
            if (likeBtn) likeBtn.classList.remove("active");
            btn.classList.toggle("active", !wasActive);
        }

        if (action === "copy") {
            const contentEl = bubble.querySelector(".markdown-content") || bubble.querySelector(".source-document-details");
            const textToCopy = contentEl ? contentEl.innerText.trim() : "";
            try {
                await navigator.clipboard.writeText(textToCopy);
                btn.classList.add("copied");
                setTimeout(() => btn.classList.remove("copied"), 1500);
            } catch (err) {
                console.error("Copy failed:", err);
            }
        }

        if (action === "regenerate") {
            const question = bubble.dataset.question;
            const mode = bubble.dataset.mode === "retrieve";
            if (!question) return;
            bubble.remove();
            await fetchAndRespond(question, mode);
        }
    });

    // Main send message orchestration
    async function submitMessage() {
        const text = chatInput.value.trim();
        if (!text) return;

        // Transition from welcome state to conversation stream
        if (welcomeState.style.display !== "none") {
            welcomeState.style.display = "none";
            messageStream.style.display = "flex";
        }

        // 1. Append user bubble
        appendBubble("user", text);
        chatInput.value = "";

        // 2. Fetch + render assistant response
        await fetchAndRespond(text, retrieveOnlySession);
    }

    // Fetches from the RAG backend and renders the assistant response.
    // Reused by both the initial send and the "regenerate" action.
    async function fetchAndRespond(text, useRetrieveMode) {
        const loaderId = appendLoadingBubble();
        messageStream.scrollIntoView({ behavior: "smooth", block: "end" });

        try {
            if (useRetrieveMode) {
                // Call /retrieve endpoint
                const res = await fetch(`${API_BASE_URL}/retrieve`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ query: text, k: 5 })
                });

                if (!res.ok) throw new Error("HTTP error " + res.status);
                const data = await res.json();

                removeLoadingBubble(loaderId);
                renderRetrieveResponse(data, text, useRetrieveMode);
            } else {
                // Call /ask endpoint
                const res = await fetch(`${API_BASE_URL}/ask`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ question: text, k: 5, rewrite: true })
                });

                if (!res.ok) throw new Error("HTTP error " + res.status);
                const data = await res.json();

                removeLoadingBubble(loaderId);
                renderAskResponse(data, text, useRetrieveMode);
            }
        } catch (err) {
            console.error(err);
            removeLoadingBubble(loaderId);
            appendBubble("assistant", "نعتذر، حدث خطأ أثناء الاتصال بالخادم. الرجاء التأكد من تشغيل خادم RAG وإعادة المحاولة.", text, useRetrieveMode);
        }

        // Scroll viewport to bottom
        const viewport = document.querySelector(".chat-viewport");
        viewport.scrollTop = viewport.scrollHeight;
    }

    // Generic Bubble renderer
    function appendBubble(sender, text, sourceQuestion, sourceMode) {
        const bubble = document.createElement("div");
        bubble.className = `chat-bubble ${sender}`;

        if (sender === "assistant") {
            bubble.dataset.question = sourceQuestion || "";
            bubble.dataset.mode = sourceMode ? "retrieve" : "ask";
            bubble.innerHTML = `
                <div class="bot-avatar-badge">
                    <svg viewBox="0 0 24 24" class="avatar-svg" fill="currentColor">
                        <path d="M12 3L2 12h3v8h14v-8h3z"/>
                        <path d="M12 5l-7 7h3v6h8v-6h3z" fill="#cca46c"/>
                    </svg>
                </div>
                <div class="text-wrapper">
                    <div class="markdown-content">${renderMarkdown(text)}</div>
                    ${actionBarHTML()}
                </div>
            `;
        } else {
            bubble.innerHTML = `
                <div class="text-wrapper">
                    <p>${escapeHTML(text)}</p>
                </div>
            `;
        }

        messageStream.appendChild(bubble);
        return bubble;
    }

    // Markup for the response action bar (like / dislike / regenerate / copy)
    function actionBarHTML() {
        return `
            <div class="message-actions">
                <button type="button" class="msg-action-btn" data-action="like" title="إعجاب" aria-label="إعجاب">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M7 10v12"/>
                        <path d="M15 5.88 14 10h5.83a2 2 0 0 1 1.92 2.56l-2.33 8A2 2 0 0 1 17.5 22H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2.76a2 2 0 0 0 1.79-1.11L12 2h0a3.13 3.13 0 0 1 3 3.88Z"/>
                    </svg>
                </button>
                <button type="button" class="msg-action-btn" data-action="dislike" title="عدم إعجاب" aria-label="عدم إعجاب">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M17 14V2"/>
                        <path d="M9 18.12 10 14H4.17a2 2 0 0 1-1.92-2.56l2.33-8A2 2 0 0 1 6.5 2H20a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-2.76a2 2 0 0 0-1.79 1.11L12 22h0a3.13 3.13 0 0 1-3-3.88Z"/>
                    </svg>
                </button>
                <button type="button" class="msg-action-btn" data-action="regenerate" title="إعادة المحاولة" aria-label="إعادة المحاولة">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/>
                        <path d="M21 3v5h-5"/>
                        <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/>
                        <path d="M8 16H3v5"/>
                    </svg>
                </button>
                <button type="button" class="msg-action-btn" data-action="copy" title="نسخ" aria-label="نسخ">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <rect width="14" height="14" x="8" y="8" rx="2" ry="2"/>
                        <path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/>
                    </svg>
                </button>
            </div>
        `;
    }

    // Loading indicator renderer
    function appendLoadingBubble() {
        const id = "loader_" + Date.now();
        const bubble = document.createElement("div");
        bubble.className = "chat-bubble assistant";
        bubble.id = id;

        bubble.innerHTML = `
            <div class="bot-avatar-badge">
                <svg viewBox="0 0 24 24" class="avatar-svg" fill="currentColor">
                    <path d="M12 3L2 12h3v8h14v-8h3z"/>
                    <path d="M12 5l-7 7h3v6h8v-6h3z" fill="#cca46c"/>
                </svg>
            </div>
            <div class="text-wrapper">
                <div class="chat-loading">
                    <div class="dot"></div>
                    <div class="dot"></div>
                    <div class="dot"></div>
                </div>
            </div>
        `;
        messageStream.appendChild(bubble);
        return id;
    }

    function removeLoadingBubble(id) {
        const loader = document.getElementById(id);
        if (loader) loader.remove();
    }

    // Renders the structured RAG output details
    function renderAskResponse(data, sourceQuestion, sourceMode) {
        const bubble = document.createElement("div");
        bubble.className = "chat-bubble assistant";
        bubble.dataset.question = sourceQuestion || "";
        bubble.dataset.mode = sourceMode ? "retrieve" : "ask";

        let sourcesHTML = "";
        if (data.sources && data.sources.length > 0) {
            sourcesHTML = `
                <div class="source-document-details">
                    <div class="sources-title">المصادر والأحاديث المستند إليها:</div>
                    ${data.sources.map((src, i) => `
                        <div class="source-item">
                            [${i + 1}] <strong>الحديث:</strong> ${escapeHTML(src.hadith)} <br>
                            <strong>الحكم:</strong> <span style="color: var(--gold-bright)">${escapeHTML(src.hokm)}</span> |
                            <strong>الراوي:</strong> ${escapeHTML(src.rawy)} |
                            <strong>المصدر:</strong> ${escapeHTML(src.source)}
                        </div>
                    `).join('')}
                </div>
            `;
        }

        let queryRewrittenHTML = "";
        if (data.query_rewritten) {
            queryRewrittenHTML = `<div style="font-size: 11px; color: var(--gold-primary); margin-bottom: 6px;">تم البحث بالاستعلام الفصيح: "${escapeHTML(data.query_rewritten)}"</div>`;
        }

        bubble.innerHTML = `
            <div class="bot-avatar-badge">
                <svg viewBox="0 0 24 24" class="avatar-svg" fill="currentColor">
                    <path d="M12 3L2 12h3v8h14v-8h3z"/>
                    <path d="M12 5l-7 7h3v6h8v-6h3z" fill="#cca46c"/>
                </svg>
            </div>
            <div class="text-wrapper" style="width: 100%;">
                ${queryRewrittenHTML}
                <div class="markdown-content">${renderMarkdown(data.answer)}</div>
                ${sourcesHTML}
                ${actionBarHTML()}
            </div>
        `;
        messageStream.appendChild(bubble);
    }

    // Renders the retrieve-only output cards
    function renderRetrieveResponse(data, sourceQuestion, sourceMode) {
        const bubble = document.createElement("div");
        bubble.className = "chat-bubble assistant";
        bubble.dataset.question = sourceQuestion || "";
        bubble.dataset.mode = sourceMode ? "retrieve" : "ask";

        let resultsHTML = "";
        if (data.results && data.results.length > 0) {
            resultsHTML = `
                <div class="source-document-details" style="margin-top: 0; border: none; padding: 0;">
                    <div class="sources-title">نتائج البحث والاسترجاع المباشر:</div>
                    ${data.results.map((src, i) => `
                        <div class="source-item" style="margin-bottom: 12px; font-size: 12px;">
                            [${i + 1}] <strong>الحديث:</strong> ${escapeHTML(src.hadith)} <br>
                            <strong>الشرح:</strong> ${escapeHTML(src.sharh)} <br>
                            <strong>الحكم:</strong> <span style="color: var(--gold-bright)">${escapeHTML(src.hokm)}</span> |
                            <strong>الراوي:</strong> ${escapeHTML(src.rawy)} |
                            <strong>المصدر:</strong> ${escapeHTML(src.source)}
                        </div>
                    `).join('')}
                </div>
            `;
        } else {
            resultsHTML = `<p>لم يتم العثور على نتائج استرجاع مطابقة للطلب.</p>`;
        }

        bubble.innerHTML = `
            <div class="bot-avatar-badge">
                <svg viewBox="0 0 24 24" class="avatar-svg" fill="currentColor">
                    <path d="M12 3L2 12h3v8h14v-8h3z"/>
                    <path d="M12 5l-7 7h3v6h8v-6h3z" fill="#cca46c"/>
                </svg>
            </div>
            <div class="text-wrapper" style="width: 100%;">
                ${resultsHTML}
                ${actionBarHTML()}
            </div>
        `;
        messageStream.appendChild(bubble);
    }

    // Simple HTML escaping helper for security
    function escapeHTML(str) {
        if (!str) return "";
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // Renders assistant text as sanitized markdown (ChatGPT-style, no bubble box)
    function renderMarkdown(str) {
        if (!str) return "";
        try {
            const rawHTML = marked.parse(str, { breaks: true, gfm: true });
            return DOMPurify.sanitize(rawHTML);
        } catch (err) {
            console.error("Markdown render error:", err);
            return `<p>${escapeHTML(str)}</p>`;
        }
    }
});