/* ==========================================================================
   Noor Al-Hidayah — المفضلة Page Logic
   ========================================================================== */

document.addEventListener("DOMContentLoaded", () => {
    // ---- Shared shell elements (theme + mobile sidebar) ----
    const themeSwitch = document.getElementById("themeSwitch");
    const savedTheme = localStorage.getItem("noor-theme");
    const isDark = savedTheme === "dark";
    document.body.classList.toggle("light-theme", !isDark);
    themeSwitch.checked = isDark;

    const leftColumn = document.querySelector(".left-column");
    const leftSidebarToggle = document.getElementById("leftSidebarToggle");
    const leftSidebarOverlay = document.getElementById("leftSidebarOverlay");

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

    // ---- Favorites store helpers (shared format with app.js / duas.js / ahadith.js) ----
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

    const SOURCE_LABELS = {
        chat: "من المحادثة",
        hadith: "من الأحاديث",
        dua: "من الأدعية"
    };

    // ---- Favorites page elements ----
    const filterChips = document.querySelectorAll("#favoritesFilters .category-chip");
    const searchInput = document.getElementById("favoritesSearchInput");
    const gridEl = document.getElementById("favoritesGrid");
    const emptyState = document.getElementById("favoritesEmptyState");
    const emptyText = document.getElementById("favoritesEmptyText");

    let activeSource = "all";

    function normalizeArabic(str) {
        return (str || "")
            .replace(/[\u064B-\u0652\u0670\u06D6-\u06ED]/g, "")
            .replace(/[إأآا]/g, "ا")
            .replace(/ى/g, "ي")
            .replace(/ة/g, "ه")
            .trim();
    }

    function escapeHTML(str) {
        if (!str) return "";
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    function renderList() {
        const favorites = getFavorites().sort((a, b) => (b.date || "").localeCompare(a.date || ""));
        const query = normalizeArabic(searchInput.value.trim());

        const filtered = favorites.filter((item) => {
            const matchesSource = activeSource === "all" || item.type === activeSource;
            if (!matchesSource) return false;
            if (!query) return true;
            const haystack = normalizeArabic([item.title, item.text, item.meta].filter(Boolean).join(" "));
            return haystack.includes(query);
        });

        if (filtered.length === 0) {
            gridEl.innerHTML = "";
            emptyState.style.display = "flex";
            emptyText.textContent = favorites.length === 0
                ? "لم تقم بإضافة أي عنصر إلى المفضلة بعد. اضغط على أيقونة الحفظ الموجودة في أي رد أو دعاء أو حديث لإضافته إلى هذه القائمة."
                : "لا توجد عناصر مطابقة لبحثك أو للفلتر المحدد.";
            return;
        }

        emptyState.style.display = "none";

        gridEl.innerHTML = filtered.map((item) => `
            <div class="dua-card favorite-card" data-fav-id="${escapeHTML(item.id)}">
                <div class="dua-card-header">
                    <span class="favorite-source-badge favorite-badge-${item.type}">${SOURCE_LABELS[item.type] || item.type}</span>
                    <div class="dua-card-actions">
                        <button type="button" class="dua-action-btn" data-action="remove" title="إزالة من المفضلة">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <polyline points="3 6 5 6 21 6"></polyline>
                                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                                <line x1="10" y1="11" x2="10" y2="17"></line>
                                <line x1="14" y1="11" x2="14" y2="17"></line>
                            </svg>
                        </button>
                    </div>
                </div>
                ${item.title ? `<p class="favorite-title">${escapeHTML(item.title)}</p>` : ""}
                <p class="dua-text favorite-text">${escapeHTML(item.text)}</p>
                ${item.meta ? `<div class="hadith-meta favorite-meta">${escapeHTML(item.meta)}</div>` : ""}
            </div>
        `).join("");
    }

    filterChips.forEach((btn) => {
        btn.addEventListener("click", () => {
            filterChips.forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");
            activeSource = btn.dataset.source;
            renderList();
        });
    });

    searchInput.addEventListener("input", renderList);

    // ---- Delete confirmation modal ----
    const confirmOverlay = document.getElementById("confirmDeleteOverlay");
    const confirmCancelBtn = document.getElementById("confirmDeleteCancel");
    const confirmConfirmBtn = document.getElementById("confirmDeleteConfirm");
    let pendingDeleteId = null;

    function openDeleteConfirm(id) {
        pendingDeleteId = id;
        confirmOverlay.style.display = "flex";
    }

    function closeDeleteConfirm() {
        pendingDeleteId = null;
        confirmOverlay.style.display = "none";
    }

    gridEl.addEventListener("click", (e) => {
        const btn = e.target.closest('.dua-action-btn[data-action="remove"]');
        if (!btn) return;
        const card = btn.closest(".favorite-card");
        const id = card?.dataset.favId;
        if (!id) return;
        openDeleteConfirm(id);
    });

    confirmCancelBtn.addEventListener("click", closeDeleteConfirm);
    confirmOverlay.addEventListener("click", (e) => {
        if (e.target === confirmOverlay) closeDeleteConfirm();
    });

    confirmConfirmBtn.addEventListener("click", () => {
        if (!pendingDeleteId) return;
        const updated = getFavorites().filter((item) => item.id !== pendingDeleteId);
        saveFavorites(updated);
        closeDeleteConfirm();
        renderList();
    });

    renderList();
});
