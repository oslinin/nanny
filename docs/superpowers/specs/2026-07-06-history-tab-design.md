# History tab — design

## Goal

Give the parent a way to see every logged activity across days, not just
today's rhythm strip. Introduce tab navigation to the frontend and add a
History view that lists all activities grouped by day.

Frontend-only change to `web/index.html`. No backend changes: the
`/api/history` endpoint already returns every activity for the client.

## Navigation

Add a tab bar directly under the masthead with two pills:

- **Today** — active by default.
- **History**.

Clicking a pill toggles which view is visible (client-side only, no routing).
The active pill is styled with the accent token.

## Today view

The existing page contents, unchanged: the "Today's rhythm" strip, Quick tap,
Chat, and the References panel. These simply become the body of the Today tab.

## History view

A new panel, hidden until the History tab is selected. Lists every activity
from `/api/history`:

- Grouped by day with a header per day: **Today**, **Yesterday**, then a
  formatted date (e.g. `Mon, Jul 6`) for older days.
- Days newest-first; within a day, activities newest-first.
- Each row: activity glyph (reuse `RHYTHM_GLYPHS`), a human label
  (e.g. `+4oz bottle`, `poop`), the time (`3:10p`), and `notes` when present.
- Empty state: "Nothing logged yet."
- Read-only. No edit/delete controls — there is no backend mutation endpoint
  for activities, and adding one is out of scope.

## Data flow

Reuses the existing `authHeaders()` + `fetch(API_BASE + '/api/history')` call
already in the file. History is fetched when the History tab is first opened,
and re-fetched after any new activity is logged (quick-tap or chat), so
switching to the tab shows current data.

Refactor the day-labeling (Today / Yesterday / date) into a small shared
helper so the rhythm strip's "today" filter and the history grouping agree on
what counts as today.

## Styling

Reuse existing CSS tokens (`--surface`, `--border`, `--bubble`, `--muted`,
`--accent`, `--radius`, `--shadow`). Tab pills mirror the existing pill/button
idioms. Stays a single self-contained `index.html` with no new dependencies.

## Out of scope (YAGNI)

Search, filters, date-range picker, pagination, editing, and deleting
activities. The grouped read-only list only.
