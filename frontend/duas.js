/* ==========================================================================
   Noor Al-Hidayah — الأدعية Page Logic
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

    // ---- Left sidebar visibility (set from the Settings page) ----
    if (appContainer) {
        const leftHidden = localStorage.getItem("noor-leftbar-hidden") === "true";
        appContainer.classList.toggle("left-hidden", leftHidden);
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

    // ---- Favorites store helpers (shared format with app.js / ahadith.js / favorites.js) ----
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

    function simpleHash(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            hash = (hash << 5) - hash + str.charCodeAt(i);
            hash |= 0;
        }
        return Math.abs(hash).toString(36);
    }


    // ---- Duas page elements ----
    const categoryButtons = document.querySelectorAll(".category-chip");
    const duaCards = document.querySelectorAll(".dua-card");
    const searchInput = document.getElementById("duaSearchInput");
    const emptyState = document.getElementById("duasEmptyState");

    let activeCategory = "all";

    function normalizeArabic(str) {
        return str
            .replace(/[\u064B-\u0652\u0670\u06D6-\u06ED]/g, "") // strip tashkeel/diacritics
            .replace(/[إأآا]/g, "ا")
            .replace(/ى/g, "ي")
            .replace(/ة/g, "ه")
            .trim();
    }

    function applyFilters() {
        const query = normalizeArabic(searchInput.value.trim());
        let visibleCount = 0;

        duaCards.forEach((card) => {
            const matchesCategory = activeCategory === "all" || card.dataset.category === activeCategory;
            const cardText = normalizeArabic(card.querySelector(".dua-text").textContent);
            const tagText = normalizeArabic(card.querySelector(".dua-category-tag").textContent);
            const matchesSearch = !query || cardText.includes(query) || tagText.includes(query);
            const visible = matchesCategory && matchesSearch;
            card.style.display = visible ? "" : "none";
            if (visible) visibleCount++;
        });

        emptyState.style.display = visibleCount === 0 ? "flex" : "none";
    }

    // Category filter chips
    categoryButtons.forEach((btn) => {
        btn.addEventListener("click", () => {
            categoryButtons.forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");
            activeCategory = btn.dataset.category;
            applyFilters();
        });
    });

    // Live search
    searchInput.addEventListener("input", applyFilters);

    // Dua card actions: copy / favorite (event delegation)
    document.getElementById("duasGrid").addEventListener("click", async (e) => {
        const btn = e.target.closest(".dua-action-btn");
        if (!btn) return;

        const card = btn.closest(".dua-card");
        const action = btn.dataset.action;

        if (action === "copy") {
            const text = card.querySelector(".dua-text").textContent.trim();
            try {
                await navigator.clipboard.writeText(text);
                btn.classList.add("copied");
                setTimeout(() => btn.classList.remove("copied"), 1500);
            } catch (err) {
                console.error("Copy failed:", err);
            }
        }

        if (action === "favorite") {
            const text = card.querySelector(".dua-text").textContent.trim();
            const tag = card.querySelector(".dua-category-tag").textContent.trim();
            const source = card.querySelector(".dua-source")?.textContent.trim() || null;
            const id = "dua_" + simpleHash(tag + "|" + text);

            const favorites = getFavorites();
            const exists = favorites.some((item) => item.id === id);
            if (exists) {
                saveFavorites(favorites.filter((item) => item.id !== id));
                btn.classList.remove("active");
            } else {
                favorites.push({ id, type: "dua", title: tag, text, meta: source, date: new Date().toISOString() });
                saveFavorites(favorites);
                btn.classList.add("active");
            }
        }
    });

    // Reflect already-saved favorites on page load
    (function initFavoriteStates() {
        const favIds = new Set(getFavorites().map((item) => item.id));
        duaCards.forEach((card) => {
            const text = card.querySelector(".dua-text").textContent.trim();
            const tag = card.querySelector(".dua-category-tag").textContent.trim();
            const id = "dua_" + simpleHash(tag + "|" + text);
            const favBtn = card.querySelector('.dua-action-btn[data-action="favorite"]');
            if (favBtn && favIds.has(id)) favBtn.classList.add("active");
        });
    })();
});