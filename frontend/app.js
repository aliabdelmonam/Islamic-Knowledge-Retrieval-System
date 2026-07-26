/* ==========================================================================
   Noor Al-Hidayah Javascript Application Logic
   ========================================================================== */

const API_BASE_URL = "http://localhost:8000/api/v1";

document.addEventListener("DOMContentLoaded", () => {
    // If the user reloaded the page, clear the session storage
    const navEntry = performance.getEntriesByType('navigation')[0];
    if (navEntry && navEntry.type === 'reload') {
        sessionStorage.removeItem('noor-session-id');
        console.log('Session ID cleared on reload.');
    }

    // DOM Elements
    const chatForm = document.getElementById("chatForm");
    const chatInput = document.getElementById("chatInput");
    const messageStream = document.getElementById("messageStream");
    const welcomeState = document.getElementById("welcomeState");
    const mockupCard = document.getElementById("mockupCard");
    const themeSwitch = document.getElementById("themeSwitch");
    const savedTheme = localStorage.getItem("noor-theme");
    const isDark = savedTheme === "dark";
    document.body.classList.toggle("light-theme", !isDark);
    themeSwitch.checked = isDark;
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

    // ---- Master sidebar toggle: hides the left widgets column, collapses the right nav to icons ----
    const sidebarMasterToggle = document.getElementById("sidebarMasterToggle");
    const appContainer = document.querySelector(".app-container");
    if (sidebarMasterToggle && appContainer) {
        const collapsedSaved = localStorage.getItem("noor-sidebar-collapsed") === "true";
        appContainer.classList.toggle("sidebar-collapsed", collapsedSaved);

        sidebarMasterToggle.addEventListener("click", () => {
            const collapsed = !appContainer.classList.contains("sidebar-collapsed");
            appContainer.classList.toggle("sidebar-collapsed", collapsed);
            localStorage.setItem("noor-sidebar-collapsed", collapsed ? "true" : "false");
        });
    }

    // ---- Left sidebar visibility (set from the Settings page) ----
    if (appContainer) {
        const leftHidden = localStorage.getItem("noor-leftbar-hidden") === "true";
        appContainer.classList.toggle("left-hidden", leftHidden);
    }

    let retrieveOnlySession = false; // true = call /retrieve instead of /ask

    // ---- Favorites store helpers (shared format with duas.js / ahadith.js / favorites.js) ----
    const FAVORITES_KEY = "noor-favorites";

    function getFavorites() {
        try {
            return JSON.parse(localStorage.getItem(FAVORITES_KEY) || "[]");
        } catch (err) {
            return [];
        }
    }

    function saveFavorites(list) {
        localStorage.setItem(FAVORITES_KEY, JSON.stringify(list));
    }

    function addFavorite(entry) {
        const list = getFavorites();
        if (!list.some((item) => item.id === entry.id)) {
            list.push({ ...entry, date: new Date().toISOString() });
            saveFavorites(list);
        }
    }


    // 1. Theme Toggle implementation
    themeSwitch.addEventListener("change", (e) => {
        if (!e.target.checked) {
            document.body.classList.add("light-theme");
            localStorage.setItem("noor-theme", "light");
        } else {
            document.body.classList.remove("light-theme");
            localStorage.setItem("noor-theme", "dark");
        }
    });

    // ---- Prayer Times Widget (left sidebar) ----
    function initPrayerWidget() {
        const widget = document.querySelector(".prayer-widget");
        if (!widget) return;

        const cityNameEl = widget.querySelector(".city-name");
        const hijriTextEl = widget.querySelector(".hijri-date-text");
        const prayerItems = widget.querySelectorAll(".prayer-time-item");
        const scheduleBtn = widget.querySelector(".show-schedule-btn");
        const PRAYER_ORDER = ["Fajr", "Sunrise", "Dhuhr", "Asr", "Maghrib", "Isha"];

        if (scheduleBtn) {
            scheduleBtn.addEventListener("click", () => {
                window.location.href = "mawaqit.html";
            });
        }

        function formatTime12h(hhmm) {
            const [hStr, mStr] = hhmm.split(":");
            let h = parseInt(hStr, 10);
            const period = h >= 12 ? "م" : "ص";
            h = h % 12;
            if (h === 0) h = 12;
            return `${h}:${mStr} ${period}`;
        }

        function minutesSinceMidnight(hhmm) {
            const [h, m] = hhmm.split(":").map((v) => parseInt(v, 10));
            return h * 60 + m;
        }

        async function loadWidgetTimings() {
            const saved = JSON.parse(localStorage.getItem("noor-mawaqit-location") || "null");
            const countryEn = saved?.countryEn || "Saudi Arabia";
            const cityEn = saved?.cityEn || "Makkah";
            const cityAr = saved?.cityAr || "مكة المكرمة";
            const countryAr = saved?.countryAr || "السعودية";

            try {
                const url = `https://api.aladhan.com/v1/timingsByCity?city=${encodeURIComponent(cityEn)}&country=${encodeURIComponent(countryEn)}&method=5`;
                const res = await fetch(url);
                if (!res.ok) throw new Error("HTTP error " + res.status);
                const json = await res.json();
                if (json.code !== 200 || !json.data) throw new Error("Invalid response");

                const timings = json.data.timings;
                const now = new Date();
                const nowMinutes = now.getHours() * 60 + now.getMinutes();
                let nextIndex = 0;
                let found = false;
                PRAYER_ORDER.forEach((key, i) => {
                    const raw = timings[key].split(" ")[0];
                    if (!found && minutesSinceMidnight(raw) > nowMinutes) {
                        nextIndex = i;
                        found = true;
                    }
                });

                prayerItems.forEach((item, i) => {
                    const key = PRAYER_ORDER[i];
                    if (!key) return;
                    const raw = timings[key].split(" ")[0];
                    const valEl = item.querySelector(".prayer-val");
                    if (valEl) valEl.textContent = formatTime12h(raw);
                    item.classList.toggle("active", i === nextIndex);
                });

                if (cityNameEl) cityNameEl.textContent = `${cityAr}، ${countryAr}`;
                if (hijriTextEl) {
                    const h = json.data.date.hijri;
                    hijriTextEl.textContent = `${h.day} ${h.month.ar} ${h.year} هـ`;
                }
            } catch (err) {
                console.error("Prayer widget fetch failed:", err);
                // Keep the default placeholder values shown in the markup as a fallback
            }
        }

        loadWidgetTimings();
    }
    initPrayerWidget();

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

        if (action === "favorite") {
            const contentEl = bubble.querySelector(".markdown-content") || bubble.querySelector(".source-document-details");
            const textToSave = contentEl ? contentEl.innerText.trim() : "";
            const msgId = bubble.dataset.msgId || ("msg_" + Date.now() + "_" + Math.random().toString(36).slice(2, 8));
            bubble.dataset.msgId = msgId;

            addFavorite({
                id: msgId,
                type: "chat",
                title: bubble.dataset.question || "",
                text: textToSave,
                meta: null
            });

            btn.classList.add("active");
            window.location.href = "favorites.html";
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
                // Fetch session_id from sessionStorage if exists
                let sessionId = sessionStorage.getItem("noor-session-id");
                let body = { question: text, k: 5, rewrite: true };
                if (sessionId) {
                    body.session_id = sessionId;
                }

                // Call /ask endpoint
                const res = await fetch(`${API_BASE_URL}/ask`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(body)
                });

                if (!res.ok) throw new Error("HTTP error " + res.status);
                const data = await res.json();

                if (data.session_id) {
                    sessionStorage.setItem("noor-session-id", data.session_id);
                }

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

    // Markup for the response action bar (like / dislike / favorite / regenerate / copy)
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
                <button type="button" class="msg-action-btn" data-action="favorite" title="إضافة للمفضلة" aria-label="إضافة للمفضلة">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"/>
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

    const DUMMY_MESSAGES = [
        "جاري إرسال الطلب...",
        "جاري البحث عن أحاديث...",
        "جاري الكشف عن جودة الأحاديث...",
        "جاري صياغة الإجابة...",
        "يرجى الانتظار قليلاً..."
    ];

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
                <div class="chat-loading" style="display: inline-block;">
                    <div class="dot"></div>
                    <div class="dot"></div>
                    <div class="dot"></div>
                </div>
                <div class="loading-text" style="font-size: 13px; color: #888; margin-top: 8px;">جاري إرسال الطلب...</div>
            </div>
        `;
        messageStream.appendChild(bubble);

        let msgIdx = 1;
        const textEl = bubble.querySelector(".loading-text");
        const intervalId = setInterval(() => {
            if (textEl) {
                textEl.textContent = DUMMY_MESSAGES[msgIdx % DUMMY_MESSAGES.length];
                msgIdx++;
            }
        }, 5000); // Change text every 5 seconds

        bubble.dataset.intervalId = intervalId;

        return id;
    }

    function removeLoadingBubble(id) {
        const loader = document.getElementById(id);
        if (loader) {
            if (loader.dataset.intervalId) {
                clearInterval(loader.dataset.intervalId);
            }
            loader.remove();
        }
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