/* ==========================================================================
   Noor Al-Hidayah — الإعدادات Page Logic
   ========================================================================== */

document.addEventListener("DOMContentLoaded", () => {
    // ---- Shared shell elements (theme + master sidebar toggle) ----
    const themeSwitch = document.getElementById("themeSwitch");
    const savedTheme = localStorage.getItem("noor-theme");
    const isDark = savedTheme === "dark";
    document.body.classList.toggle("light-theme", !isDark);
    themeSwitch.checked = isDark;

    // ---- Master sidebar toggle: collapses the right nav to icons (no left column on this page) ----
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

    function applyTheme(dark) {
        document.body.classList.toggle("light-theme", !dark);
        localStorage.setItem("noor-theme", dark ? "dark" : "light");
        themeSwitch.checked = dark;
        settingsThemeSwitch.checked = dark;
    }

    themeSwitch.addEventListener("change", (e) => applyTheme(e.target.checked));

    // ==================== Settings page controls ====================

    // ---- Appearance: dark/light mode ----
    const settingsThemeSwitch = document.getElementById("settingsThemeSwitch");
    settingsThemeSwitch.checked = isDark;
    settingsThemeSwitch.addEventListener("change", (e) => applyTheme(e.target.checked));

    // ---- Sidebar: show/hide the left widgets column only ----
    const settingsLeftbarSwitch = document.getElementById("settingsLeftbarSwitch");
    const leftHiddenSaved = localStorage.getItem("noor-leftbar-hidden") === "true";
    settingsLeftbarSwitch.checked = !leftHiddenSaved;

    settingsLeftbarSwitch.addEventListener("change", (e) => {
        const visible = e.target.checked;
        localStorage.setItem("noor-leftbar-hidden", visible ? "false" : "true");
    });

    // ---- Location: country / city picker (shared with مواقيت الصلاة + the sidebar widget) ----
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
            cities: [{ ar: "مدينة الكويت", en: "Kuwait City" }]
        },
        "قطر": {
            country: "Qatar",
            cities: [{ ar: "الدوحة", en: "Doha" }]
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
            cities: [{ ar: "الخرطوم", en: "Khartoum" }]
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
            cities: [{ ar: "كوالالمبور", en: "Kuala Lumpur" }]
        },
        "إندونيسيا": {
            country: "Indonesia",
            cities: [{ ar: "جاكرتا", en: "Jakarta" }]
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
            cities: [{ ar: "لندن", en: "London" }]
        },
        "الولايات المتحدة": {
            country: "United States",
            cities: [{ ar: "نيويورك", en: "New York" }]
        }
    };

    const countrySelect = document.getElementById("settingsCountrySelect");
    const citySelect = document.getElementById("settingsCitySelect");
    const saveLocationBtn = document.getElementById("settingsSaveLocationBtn");
    const currentLocationEl = document.getElementById("settingsCurrentLocation");
    const saveNote = document.getElementById("settingsSaveNote");

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

    countrySelect.addEventListener("change", () => populateCities(countrySelect.value));

    const savedLocation = JSON.parse(localStorage.getItem("noor-mawaqit-location") || "null");
    const defaultCountry = (savedLocation && LOCATIONS[savedLocation.countryAr]) ? savedLocation.countryAr : "السعودية";
    countrySelect.value = defaultCountry;
    populateCities(defaultCountry);
    if (savedLocation && savedLocation.cityEn) {
        citySelect.value = savedLocation.cityEn;
    }

    function renderCurrentLocation() {
        const loc = JSON.parse(localStorage.getItem("noor-mawaqit-location") || "null");
        if (loc && loc.cityAr && loc.countryAr) {
            currentLocationEl.textContent = `الموقع الحالي: ${loc.cityAr}، ${loc.countryAr}`;
        } else {
            currentLocationEl.textContent = "لم يتم تحديد موقع بعد (المستخدم حاليًا: مكة المكرمة، السعودية)";
        }
    }
    renderCurrentLocation();

    saveLocationBtn.addEventListener("click", () => {
        const countryAr = countrySelect.value;
        const cityEn = citySelect.value;
        const cityOption = LOCATIONS[countryAr]?.cities.find((c) => c.en === cityEn);
        const cityAr = cityOption ? cityOption.ar : cityEn;
        const countryEn = LOCATIONS[countryAr]?.country;

        localStorage.setItem("noor-mawaqit-location", JSON.stringify({ countryAr, countryEn, cityAr, cityEn }));
        renderCurrentLocation();

        saveNote.style.display = "block";
        setTimeout(() => { saveNote.style.display = "none"; }, 2500);
    });
});
