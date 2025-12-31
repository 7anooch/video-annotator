/**
 * Dark mode toggle functionality
 * Persists preference in localStorage and applies across all pages
 */

(function() {
  'use strict';

  const DARK_MODE_KEY = 'darkModeEnabled';
  const toggleBtn = document.getElementById('dark-mode-toggle');

  // Check localStorage for saved preference
  function isDarkModeEnabled() {
    const saved = localStorage.getItem(DARK_MODE_KEY);
    if (saved !== null) {
      return saved === 'true';
    }
    // Default to light mode if no preference saved
    return false;
  }

  // Apply dark mode class to body
  function applyDarkMode(enabled) {
    if (enabled) {
      document.body.classList.add('dark-mode');
      if (toggleBtn) {
        toggleBtn.textContent = '☀️ Light';
        toggleBtn.title = 'Toggle light mode';
      }
    } else {
      document.body.classList.remove('dark-mode');
      if (toggleBtn) {
        toggleBtn.textContent = '🌙 Dark';
        toggleBtn.title = 'Toggle dark mode';
      }
    }
  }

  // Initialize on page load
  function initDarkMode() {
    const enabled = isDarkModeEnabled();
    applyDarkMode(enabled);
  }

  // Toggle dark mode
  function toggleDarkMode() {
    const current = document.body.classList.contains('dark-mode');
    const newState = !current;
    applyDarkMode(newState);
    localStorage.setItem(DARK_MODE_KEY, String(newState));
  }

  // Set up event listener
  if (toggleBtn) {
    toggleBtn.addEventListener('click', toggleDarkMode);
  }

  // Initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initDarkMode);
  } else {
    initDarkMode();
  }
})();

