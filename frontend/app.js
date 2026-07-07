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
    const retrieveModeBtn = document.getElementById("retrieveModeBtn");
    const suggestionChips = document.querySelectorAll(".suggestion-chips .chip-btn");

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

    // Helper: format dates to regional Arabic timestamp
    function getArabicTime() {
        const d = new Date();
        let hours = d.getHours();
        const minutes = String(d.getMinutes()).padStart(2, '0');
        const ampm = hours >= 12 ? 'م' : 'ص';
        hours = hours % 12;
        hours = hours ? hours : 12; // 0 should be 12
        return `${hours}:${minutes} ${ampm}`;
    }

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

        // 2. Append loading indicator bubble
        const loaderId = appendLoadingBubble();
        messageStream.scrollIntoView({ behavior: "smooth", block: "end" });

        try {
            if (retrieveOnlySession) {
                // Call /retrieve endpoint
                const res = await fetch(`${API_BASE_URL}/retrieve`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ query: text, k: 5 })
                });

                if (!res.ok) throw new Error("HTTP error " + res.status);
                const data = await res.json();

                removeLoadingBubble(loaderId);
                renderRetrieveResponse(data);
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
                renderAskResponse(data);
            }
        } catch (err) {
            console.error(err);
            removeLoadingBubble(loaderId);
            appendBubble("assistant", "نعتذر، حدث خطأ أثناء الاتصال بالخادم. الرجاء التأكد من تشغيل خادم RAG وإعادة المحاولة.");
        }

        // Scroll viewport to bottom
        const viewport = document.querySelector(".chat-viewport");
        viewport.scrollTop = viewport.scrollHeight;
    }

    // Generic Bubble renderer
    function appendBubble(sender, text) {
        const bubble = document.createElement("div");
        bubble.className = `chat-bubble ${sender}`;

        let avatarHTML = "";
        if (sender === "assistant") {
            avatarHTML = `
                <div class="bot-avatar-badge">
                    <svg viewBox="0 0 24 24" class="avatar-svg" fill="currentColor">
                        <path d="M12 3L2 12h3v8h14v-8h3z"/>
                        <path d="M12 5l-7 7h3v6h8v-6h3z" fill="#cca46c"/>
                    </svg>
                </div>
            `;
        }

        bubble.innerHTML = `
            ${avatarHTML}
            <div class="text-wrapper">
                <p>${escapeHTML(text)}</p>
                <span class="timestamp">${getArabicTime()}</span>
            </div>
        `;

        messageStream.appendChild(bubble);
        return bubble;
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
    function renderAskResponse(data) {
        const bubble = document.createElement("div");
        bubble.className = "chat-bubble assistant";

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
                <p style="white-space: pre-wrap;">${escapeHTML(data.answer)}</p>
                ${sourcesHTML}
                <span class="timestamp">${getArabicTime()}</span>
            </div>
        `;
        messageStream.appendChild(bubble);
    }

    // Renders the retrieve-only output cards
    function renderRetrieveResponse(data) {
        const bubble = document.createElement("div");
        bubble.className = "chat-bubble assistant";

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
                <span class="timestamp">${getArabicTime()}</span>
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
});
