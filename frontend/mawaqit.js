/* ==========================================================================
   Noor Al-Hidayah — مواقيت الصلاة Page Logic
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

    // ---- Prayer Times (مواقيت الصلاة) page logic ----

    const API_BASE = "https://api.aladhan.com/v1";

    // Countries + cities (Arabic display label + English params for the Aladhan API)
    const LOCATIONS = {
        "السعودية": {
            country: "Saudi Arabia",
            cities: [
                { ar: "مكة المكرمة", en: "Makkah" },
                { ar: "المدينة المنورة", en: "Madinah" },
                { ar: "الرياض", en: "Riyadh" },
                { ar: "جدة", en: "Jeddah" },
                { ar: "الدمام", en: "Dammam" }
            ]
        },
        "مصر": {
            country: "Egypt",
            cities: [
                { ar: "القاهرة", en: "Cairo" },
                { ar: "الإسكندرية", en: "Alexandria" },
                { ar: "الجيزة", en: "Giza" },
                { ar: "أسوان", en: "Aswan" },
                { ar: "المنصورة", en: "Mansoura" }
            ]
        },
        "الإمارات": {
            country: "United Arab Emirates",
            cities: [
                { ar: "دبي", en: "Dubai" },
                { ar: "أبوظبي", en: "Abu Dhabi" },
                { ar: "الشارقة", en: "Sharjah" }
            ]
        },
        "الأردن": {
            country: "Jordan",
            cities: [
                { ar: "عمّان", en: "Amman" },
                { ar: "إربد", en: "Irbid" },
                { ar: "الزرقاء", en: "Zarqa" }
            ]
        },
        "فلسطين": {
            country: "Palestine",
            cities: [
                { ar: "القدس", en: "Jerusalem" },
                { ar: "غزة", en: "Gaza" },
                { ar: "رام الله", en: "Ramallah" },
                { ar: "الخليل", en: "Hebron" }
            ]
        },
        "العراق": {
            country: "Iraq",
            cities: [
                { ar: "بغداد", en: "Baghdad" },
                { ar: "البصرة", en: "Basra" },
                { ar: "الموصل", en: "Mosul" },
                { ar: "النجف", en: "Najaf" }
            ]
        },
        "الكويت": {
            country: "Kuwait",
            cities: [
                { ar: "مدينة الكويت", en: "Kuwait City" }
            ]
        },
        "قطر": {
            country: "Qatar",
            cities: [
                { ar: "الدوحة", en: "Doha" }
            ]
        },
        "المغرب": {
            country: "Morocco",
            cities: [
                { ar: "الرباط", en: "Rabat" },
                { ar: "الدار البيضاء", en: "Casablanca" },
                { ar: "مراكش", en: "Marrakesh" },
                { ar: "فاس", en: "Fes" }
            ]
        },
        "الجزائر": {
            country: "Algeria",
            cities: [
                { ar: "الجزائر العاصمة", en: "Algiers" },
                { ar: "وهران", en: "Oran" },
                { ar: "قسنطينة", en: "Constantine" }
            ]
        },
        "تونس": {
            country: "Tunisia",
            cities: [
                { ar: "تونس العاصمة", en: "Tunis" },
                { ar: "صفاقس", en: "Sfax" }
            ]
        },
        "ليبيا": {
            country: "Libya",
            cities: [
                { ar: "طرابلس", en: "Tripoli" },
                { ar: "بنغازي", en: "Benghazi" }
            ]
        },
        "سوريا": {
            country: "Syria",
            cities: [
                { ar: "دمشق", en: "Damascus" },
                { ar: "حلب", en: "Aleppo" }
            ]
        },
        "لبنان": {
            country: "Lebanon",
            cities: [
                { ar: "بيروت", en: "Beirut" },
                { ar: "طرابلس", en: "Tripoli" }
            ]
        },
        "السودان": {
            country: "Sudan",
            cities: [
                { ar: "الخرطوم", en: "Khartoum" }
            ]
        },
        "اليمن": {
            country: "Yemen",
            cities: [
                { ar: "صنعاء", en: "Sanaa" },
                { ar: "عدن", en: "Aden" }
            ]
        },
        "تركيا": {
            country: "Turkey",
            cities: [
                { ar: "إسطنبول", en: "Istanbul" },
                { ar: "أنقرة", en: "Ankara" }
            ]
        },
        "ماليزيا": {
            country: "Malaysia",
            cities: [
                { ar: "كوالالمبور", en: "Kuala Lumpur" }
            ]
        },
        "إندونيسيا": {
            country: "Indonesia",
            cities: [
                { ar: "جاكرتا", en: "Jakarta" }
            ]
        },
        "باكستان": {
            country: "Pakistan",
            cities: [
                { ar: "إسلام آباد", en: "Islamabad" },
                { ar: "كراتشي", en: "Karachi" }
            ]
        },
        "المملكة المتحدة": {
            country: "United Kingdom",
            cities: [
                { ar: "لندن", en: "London" }
            ]
        },
        "الولايات المتحدة": {
            country: "United States",
            cities: [
                { ar: "نيويورك", en: "New York" }
            ]
        }
    };

    const PRAYER_LABELS = {
        Fajr: "الفجر",
        Sunrise: "الشروق",
        Dhuhr: "الظهر",
        Asr: "العصر",
        Maghrib: "المغرب",
        Isha: "العشاء"
    };

    const PRAYER_ICONS = {
        Fajr: `<path d="M17 18a5 5 0 0 0-10 0"></path><line x1="12" y1="2" x2="12" y2="9"></line><line x1="4.22" y1="10.22" x2="5.64" y2="11.64"></line><line x1="1" y1="18" x2="3" y2="18"></line><line x1="21" y1="18" x2="23" y2="18"></line><line x1="18.36" y1="11.64" x2="19.78" y2="10.22"></line><line x1="23" y1="22" x2="1" y2="22"></line><polyline points="8 6 12 2 16 6"></polyline>`,
        Sunrise: `<path d="M17 18a5 5 0 0 0-10 0"></path><line x1="12" y1="2" x2="12" y2="9"></line><line x1="4.22" y1="10.22" x2="5.64" y2="11.64"></line><line x1="1" y1="18" x2="3" y2="18"></line><line x1="21" y1="18" x2="23" y2="18"></line><line x1="18.36" y1="11.64" x2="19.78" y2="10.22"></line><line x1="23" y1="22" x2="1" y2="22"></line><polyline points="8 10 12 6 16 10"></polyline>`,
        Dhuhr: `<circle cx="12" cy="12" r="4"></circle><line x1="12" y1="2" x2="12" y2="4"></line><line x1="12" y1="20" x2="12" y2="22"></line><line x1="4.93" y1="4.93" x2="6.34" y2="6.34"></line><line x1="17.66" y1="17.66" x2="19.07" y2="19.07"></line><line x1="2" y1="12" x2="4" y2="12"></line><line x1="20" y1="12" x2="22" y2="12"></line><line x1="4.93" y1="19.07" x2="6.34" y2="17.66"></line><line x1="17.66" y1="6.34" x2="19.07" y2="4.93"></line>`,
        Asr: `<circle cx="12" cy="14" r="4"></circle><line x1="12" y1="2" x2="12" y2="6"></line><line x1="4.93" y1="7.93" x2="6.7" y2="9.7"></line><line x1="19.07" y1="7.93" x2="17.3" y2="9.7"></line><line x1="2" y1="14" x2="4.5" y2="14"></line><line x1="19.5" y1="14" x2="22" y2="14"></line>`,
        Maghrib: `<path d="M17 18a5 5 0 0 0-10 0"></path><line x1="12" y1="9" x2="12" y2="2"></line><line x1="4.22" y1="10.22" x2="5.64" y2="11.64"></line><line x1="1" y1="18" x2="3" y2="18"></line><line x1="21" y1="18" x2="23" y2="18"></line><line x1="18.36" y1="11.64" x2="19.78" y2="10.22"></line><line x1="23" y1="22" x2="1" y2="22"></line><polyline points="16 5 12 9 8 5"></polyline>`,
        Isha: `<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>`
    };

    const countrySelect = document.getElementById("countrySelect");
    const citySelect = document.getElementById("citySelect");
    const updateBtn = document.getElementById("updateTimingsBtn");
    const statusBox = document.getElementById("mawaqitStatus");
    const summaryBox = document.getElementById("mawaqitSummary");
    const cityNameEl = document.getElementById("mawaqitCityName");
    const hijriTextEl = document.getElementById("mawaqitHijriText");
    const nextPrayerEl = document.getElementById("mawaqitNextPrayer");
    const gridEl = document.getElementById("mawaqitGrid");

    // Populate country dropdown
    Object.keys(LOCATIONS).forEach((countryAr) => {
        const opt = document.createElement("option");
        opt.value = countryAr;
        opt.textContent = countryAr;
        countrySelect.appendChild(opt);
    });

    function populateCities(countryAr) {
        citySelect.innerHTML = "";
        const cities = LOCATIONS[countryAr]?.cities || [];
        cities.forEach((city) => {
            const opt = document.createElement("option");
            opt.value = city.en;
            opt.textContent = city.ar;
            citySelect.appendChild(opt);
        });
    }

    countrySelect.addEventListener("change", () => {
        populateCities(countrySelect.value);
    });

    // Restore last saved selection, defaulting to Makkah, Saudi Arabia
    const savedLocation = JSON.parse(localStorage.getItem("noor-mawaqit-location") || "null");
    const defaultCountry = (savedLocation && LOCATIONS[savedLocation.countryAr]) ? savedLocation.countryAr : "السعودية";
    countrySelect.value = defaultCountry;
    populateCities(defaultCountry);
    if (savedLocation && savedLocation.cityEn) {
        citySelect.value = savedLocation.cityEn;
    }

    function showStatus(message, isError) {
        statusBox.style.display = "block";
        statusBox.textContent = message;
        statusBox.classList.toggle("mawaqit-status-error", !!isError);
    }
    function hideStatus() {
        statusBox.style.display = "none";
        statusBox.classList.remove("mawaqit-status-error");
    }

    // Convert "HH:MM" (24h) to Arabic 12h format with ص/م
    function formatTime12h(hhmm) {
        const [hStr, mStr] = hhmm.split(":");
        let h = parseInt(hStr, 10);
        const m = mStr;
        const period = h >= 12 ? "م" : "ص";
        h = h % 12;
        if (h === 0) h = 12;
        return `${h}:${m} ${period}`;
    }

    function minutesSinceMidnight(hhmm) {
        const [h, m] = hhmm.split(":").map((v) => parseInt(v, 10));
        return h * 60 + m;
    }

    function renderTimings(data, cityAr, countryAr) {
        const timings = data.timings;
        const order = ["Fajr", "Sunrise", "Dhuhr", "Asr", "Maghrib", "Isha"];

        // Determine current/next prayer
        const now = new Date();
        const nowMinutes = now.getHours() * 60 + now.getMinutes();
        let nextKey = null;
        for (const key of order) {
            const raw = timings[key].split(" ")[0];
            if (minutesSinceMidnight(raw) > nowMinutes) {
                nextKey = key;
                break;
            }
        }
        if (!nextKey) nextKey = order[0]; // after Isha, next is tomorrow's Fajr

        gridEl.innerHTML = order.map((key) => {
            const raw = timings[key].split(" ")[0];
            const isNext = key === nextKey;
            return `
                <div class="mawaqit-card${isNext ? " active" : ""}">
                    <svg class="mawaqit-card-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        ${PRAYER_ICONS[key]}
                    </svg>
                    <span class="mawaqit-card-name">${PRAYER_LABELS[key]}</span>
                    <span class="mawaqit-card-time">${formatTime12h(raw)}</span>
                </div>
            `;
        }).join("");

        cityNameEl.textContent = `${cityAr}، ${countryAr}`;
        const hijri = data.date.hijri;
        hijriTextEl.textContent = `${hijri.day} ${hijri.month.ar} ${hijri.year} هـ`;
        nextPrayerEl.innerHTML = `الصلاة القادمة: <strong>${PRAYER_LABELS[nextKey]}</strong> — ${formatTime12h(timings[nextKey].split(" ")[0])}`;

        summaryBox.style.display = "flex";
    }

    async function fetchTimings() {
        const countryAr = countrySelect.value;
        const cityEn = citySelect.value;
        const cityOption = LOCATIONS[countryAr]?.cities.find((c) => c.en === cityEn);
        const cityAr = cityOption ? cityOption.ar : cityEn;
        const countryEn = LOCATIONS[countryAr]?.country;

        if (!countryEn || !cityEn) return;

        updateBtn.disabled = true;
        showStatus("جارٍ تحديث مواقيت الصلاة...", false);
        summaryBox.style.display = "none";
        gridEl.innerHTML = "";

        try {
            const url = `${API_BASE}/timingsByCity?city=${encodeURIComponent(cityEn)}&country=${encodeURIComponent(countryEn)}&method=5`;
            const res = await fetch(url);
            if (!res.ok) throw new Error("HTTP error " + res.status);
            const json = await res.json();
            if (json.code !== 200 || !json.data) throw new Error("Invalid response");

            renderTimings(json.data, cityAr, countryAr);
            hideStatus();

            // Save selection for next visit
            localStorage.setItem("noor-mawaqit-location", JSON.stringify({ countryAr, countryEn, cityAr, cityEn }));
        } catch (err) {
            console.error("Prayer timings fetch failed:", err);
            showStatus("تعذّر جلب مواقيت الصلاة. الرجاء التأكد من الاتصال بالإنترنت والمحاولة مرة أخرى.", true);
        } finally {
            updateBtn.disabled = false;
        }
    }

    updateBtn.addEventListener("click", fetchTimings);

    // Auto-load on first visit
    fetchTimings();
});