/* ==========================================================================
   Noor Al-Hidayah — الأدعية Page Logic
   ========================================================================== */

document.addEventListener("DOMContentLoaded", () => {
    // ---- Shared shell elements (theme + mobile sidebar) ----
    const themeSwitch = document.getElementById("themeSwitch");
    document.body.classList.add("light-theme");
    themeSwitch.checked = false;

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

    themeSwitch.addEventListener("change", (e) => {
        if (!e.target.checked) {
            document.body.classList.add("light-theme");
        } else {
            document.body.classList.remove("light-theme");
        }
    });

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
            btn.classList.toggle("active");
        }
    });
});