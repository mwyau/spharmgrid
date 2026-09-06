(() => {
  const storageKey = "spharmgrid-docs-theme";
  const media = window.matchMedia("(prefers-color-scheme: dark)");
  const modes = ["auto", "light", "dark"];
  let preference = localStorage.getItem(storageKey);
  let button = null;

  if (!modes.includes(preference)) {
    preference = "auto";
  }

  const currentTheme = () => {
    if (preference === "auto") {
      return media.matches ? "dark" : "light";
    }
    return preference;
  };

  const applyTheme = () => {
    const theme = currentTheme();
    document.documentElement.classList.toggle("theme-dark", theme === "dark");
    document.documentElement.style.colorScheme = theme;

    if (button) {
      const label = preference[0].toUpperCase() + preference.slice(1);
      button.textContent = `Theme: ${label}`;
      button.setAttribute("aria-label", `Color theme: ${label}`);
      button.title = "Switch color theme: Auto → Light → Dark";
    }
  };

  const addToggle = () => {
    const container = document.querySelector(".wy-side-nav-search");
    if (!container || container.querySelector(".theme-toggle")) {
      return;
    }

    button = document.createElement("button");
    button.type = "button";
    button.className = "theme-toggle";
    button.addEventListener("click", () => {
      const next = (modes.indexOf(preference) + 1) % modes.length;
      preference = modes[next];
      localStorage.setItem(storageKey, preference);
      applyTheme();
    });

    container.appendChild(button);
    applyTheme();
  };

  applyTheme();

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", addToggle, { once: true });
  } else {
    addToggle();
  }

  media.addEventListener("change", () => {
    if (preference === "auto") {
      applyTheme();
    }
  });
})();
