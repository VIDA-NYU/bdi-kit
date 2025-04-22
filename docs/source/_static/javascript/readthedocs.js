
document.addEventListener(
"readthedocs-addons-data-ready",
function (event) {
  const config = event.detail.data();

  // Update "stable" to merge with the most recent version
  const stableEntry = config.versions.active[0];
  const mostRecentEntry = config.versions.active[1];
  stableEntry.slug = `${mostRecentEntry.slug} (stable)`;
  stableEntry.urls.documentation = stableEntry.urls.documentation;

  // Update current if it's "stable" or the most recent version
  if (
    config.versions.current.slug === "stable" ||
    config.versions.current.slug === mostRecentEntry.slug
  ) {
    config.versions.current.slug = stableEntry.slug;
    config.versions.current.urls.documentation = stableEntry.urls.documentation;
  }

  // Rebuild the active list with "devel", the updated "stable" entry and  the other versions
  const filteredVersions = config.versions.active.slice(2, -1);
  const develEntry = config.versions.active.at(-1);
  config.versions.active = [
    develEntry,
    stableEntry,
    ...filteredVersions,
  ];

  // Create the version selector HTML
  const versionSelector = `
      <select onchange="window.location.href=this.value">
        ${config.versions.active
          .map(version => `
            <option value="${version.urls.documentation}" 
              ${version.slug === config.versions.current.slug ? "selected" : ""}>
              ${version.slug}
            </option>
          `).join('')}
      </select>
  `;

  // Insert the version selector into the page
  document.querySelector('.version-switch')
    .insertAdjacentHTML('afterbegin', versionSelector);
}
);