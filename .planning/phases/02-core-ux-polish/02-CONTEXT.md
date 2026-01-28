# Phase 2: Core UX Polish - Context

**Gathered:** 2026-01-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Efficient audio classification editing through keyboard-driven workflow and visual feedback. Users can navigate, edit, and manage classifications via keyboard shortcuts, with clear progress feedback and error handling. Focus is on improving the existing editing experience, not adding new capabilities.

</domain>

<decisions>
## Implementation Decisions

### Keyboard Shortcuts

- **Scope and behavior:** Claude's discretion on focus model (global vs table-focused) and shortcut mapping
- **Number keys (1-5):** Claude designs intuitive cycling/assignment behavior - user has no specific vision
- **Visual feedback:** Claude decides whether to show button highlights/ripples when shortcuts are used
- **Discoverability:** Claude decides on tooltips, help panel, or combination approach

### Undo & Redo

- **Scope:** All track edits are undoable (classification changes, time edits, segment splits/merges)
- **History depth:** Limited history (10-20 steps) - balances memory with useful undo capability
- **Save behavior:** Undo history persists across explicit saves (Ctrl+S) - can undo even after saving
- **Redo function:** Yes, include redo (Ctrl+Y or Ctrl+Shift+Z) - standard editing pattern

### Progress Feedback & Cancellation

- **Detail level:** Show state with percentage (e.g., "Analyzing... 45%")
- **Cancellation:** User should be able to cancel operations without causing problems (Claude decides which operations are cancellable)
- **Location:** Inline where action started - progress shows in the button/control that triggered it
- **Partial results:** Discard partial results on cancellation - rollback to state before operation

### Error Presentation & Recovery

- **Display method:** Toast notifications in corner - non-blocking
- **Visibility duration:** User dismissal required (no auto-dismiss) - ensures errors don't disappear if user is away from desk
- **Stacking behavior:** Don't stack to infinity - Claude designs smart queueing/replacement
- **Error content:** Claude decides on message detail level (user-friendly vs technical, error codes, etc.)
- **Retry capability:** Include retry where it makes sense (network errors, transient failures, etc.)

### Claude's Discretion

- Keyboard shortcut focus model and conflict resolution
- Number key mapping/cycling behavior for classifications
- Visual feedback design for shortcuts
- Help/tooltip approach for discoverability
- Error message detail level and formatting
- Retry button inclusion logic (which error types)
- Toast queueing/replacement strategy

</decisions>

<specifics>
## Specific Ideas

- **Error visibility:** User had bad experience with auto-dismissing errors - was doing other things in browser and missed error content. Critical that errors stay visible until acknowledged.

- **Cancellation philosophy:** Operations should be cancellable without generating problems - prefer clean abort over partial state issues.

</specifics>

<deferred>
## Deferred Ideas

None - discussion stayed within phase scope

</deferred>

---

*Phase: 02-core-ux-polish*
*Context gathered: 2026-01-28*
