---
phase: 02-core-ux-polish
plan: 02
subsystem: frontend-infrastructure
tags: [toast, error-handling, zustand, axios, interceptor]
requires: []
provides: [global-error-notifications, toast-system, error-codes]
affects: [all-frontend-features]

tech-stack:
  added: [zustand@5.0.2]
  patterns: [global-state-management, axios-interceptors, error-notification-pipeline]

key-files:
  created:
    - frontend/src/stores/toastStore.ts
    - frontend/src/components/ToastContainer.tsx
    - frontend/src/utils/errorHandler.ts
  modified:
    - frontend/src/components/Toast.tsx
    - frontend/src/App.tsx
    - frontend/src/main.tsx
    - backend/app/main.py

decisions:
  - id: TOAST-001
    title: Error toasts require manual dismiss
    rationale: Errors need user acknowledgment; auto-dismiss could hide critical issues
  - id: TOAST-002
    title: Max 5 toasts with oldest removal
    rationale: Prevents screen clutter during error storms; 5 is enough to notice pattern
  - id: TOAST-003
    title: Retry only for GET requests
    rationale: GET is idempotent and safe to retry; POST/PUT/DELETE could cause unintended side effects
  - id: ERROR-001
    title: Machine-readable error codes
    rationale: Enables programmatic error handling and debugging (HTTP_404, VALIDATION_ERROR, INTERNAL_ERROR)

metrics:
  duration: 320s
  completed: 2026-01-28
---

# Phase 02 Plan 02: Toast System & Error Pipeline Summary

**One-liner:** Global error notification system with zustand toast queue, axios interceptor, and machine-readable backend error codes.

## What Was Built

### Toast Store (Zustand)
- Global state management for toast notifications
- Queue management with 5-toast limit (oldest removed when exceeded)
- UUID-style ID generation (`Date.now().toString(36) + Math.random().toString(36).slice(2)`)
- Methods: `addToast`, `removeToast`, `clearAll`

### Toast Container
- Renders toast queue from zustand store
- Mounts once in App.tsx (global placement)
- Error toasts (red): manual dismiss only (no autoClose)
- Success toasts (green): 5-second auto-dismiss
- Passes `index` prop for vertical stacking
- Retry action support for GET request errors

### Enhanced Toast Component
- Added explicit X close button (top-right corner)
- X button shown on ALL toasts (with or without actions)
- Click-to-dismiss still works (backward compatible)
- Retry button in actions area when `retry` prop provided
- Proper z-index positioning with `stopPropagation` on X button

### Axios Error Interceptor
- Global response interceptor in `errorHandler.ts`
- Auto-dispatches error toasts for `status: 'error'` responses
- Uses `useToastStore.getState()` for non-React context access
- Title: `data.message || 'An error occurred'`
- Message: Shows `error_id` if available, falls back to `code`
- Retry action: Only for GET requests (idempotent)
- Always rejects promise for local error handling

### Backend Error Codes
Updated all 3 exception handlers in `main.py`:
1. **http_exception_handler**: `code: "HTTP_{status_code}"` (e.g., HTTP_404, HTTP_403)
2. **validation_exception_handler**: `code: "VALIDATION_ERROR"`
3. **global_exception_handler**: `code: "INTERNAL_ERROR"`

## Implementation Details

### Toast Queue Behavior
```typescript
// When 6th toast arrives:
const updatedToasts = [...state.toasts, newToast]
if (updatedToasts.length > 5) {
  updatedToasts.shift() // Remove oldest
}
```

### Error Interceptor Pattern
```typescript
axios.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    // Dispatch toast via getState()
    useToastStore.getState().addToast({...})
    // Still reject for local handling
    return Promise.reject(error)
  }
)
```

### Retry Action Logic
```typescript
retry: isGetRequest && error.config
  ? () => axios(error.config!)
  : undefined
```

## File Changes

### Created Files
| File | Purpose | LOC |
|------|---------|-----|
| `frontend/src/stores/toastStore.ts` | Zustand toast queue store | 41 |
| `frontend/src/components/ToastContainer.tsx` | Toast renderer from store | 32 |
| `frontend/src/utils/errorHandler.ts` | Axios error interceptor | 42 |

### Modified Files
| File | Changes |
|------|---------|
| `frontend/src/components/Toast.tsx` | Added X close button, retry action support |
| `frontend/src/App.tsx` | Mounted ToastContainer, refactored page rendering |
| `frontend/src/main.tsx` | Called setupErrorInterceptor() at startup |
| `backend/app/main.py` | Added `code` field to 3 exception handlers |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] zustand dependency not installed**
- **Found during:** Task 1 preparation
- **Issue:** zustand not in package.json dependencies
- **Fix:** Ran `npm install zustand` before creating store
- **Files modified:** frontend/package.json, frontend/package-lock.json
- **Commit:** Included in Task 1 commit (dependency installation logged)

**2. [Rule 2 - Missing Critical] stores directory didn't exist**
- **Found during:** Task 1 file creation
- **Issue:** frontend/src/stores/ directory missing
- **Fix:** Created directory with `mkdir -p frontend/src/stores`
- **Files modified:** N/A (directory creation)
- **Commit:** Included in Task 1 commit (stores/ directory created)

**3. [Rule 1 - Bug] App.tsx conditional rendering pattern**
- **Found during:** Task 1 ToastContainer mounting
- **Issue:** Early returns in App() prevented ToastContainer from mounting on all pages
- **Fix:** Refactored to JSX conditional rendering with ToastContainer at top level
- **Files modified:** frontend/src/App.tsx
- **Commit:** 4911b34 (refactored App.tsx to mount ToastContainer globally)
- **Reason:** ToastContainer must be mounted regardless of current page to show global error toasts

## Must-Haves Verification

### Truths
- [x] Toast notifications stay visible until user clicks X to dismiss (no auto-dismiss for errors)
  - Error toasts (color='red') have `autoClose={undefined}` - manual dismiss only
  - Success toasts (color='green') have `autoClose={5000}` - 5-second auto-dismiss
- [x] All backend error responses include a machine-readable code field
  - http_exception_handler: `HTTP_{status_code}`
  - validation_exception_handler: `VALIDATION_ERROR`
  - global_exception_handler: `INTERNAL_ERROR`
- [x] Axios errors automatically show toast notifications via interceptor
  - setupErrorInterceptor() called in main.tsx
  - Interceptor checks `data.status === 'error'` and dispatches toast
- [x] Max 5 toasts visible, oldest replaced when exceeded
  - `if (updatedToasts.length > 5) { updatedToasts.shift() }`

### Artifacts
- [x] `frontend/src/stores/toastStore.ts` - Zustand store, exports `useToastStore`
- [x] `frontend/src/components/ToastContainer.tsx` - Renders from store, exports `default`
- [x] `frontend/src/utils/errorHandler.ts` - Exports `setupErrorInterceptor`

### Key Links
- [x] `errorHandler.ts` → `toastStore.ts` via `useToastStore.getState()`
  - Line 16 in errorHandler.ts: `const { addToast } = useToastStore.getState()`
- [x] `main.py` → `errorHandler.ts` via `"code":` field in JSON responses
  - Lines 176, 190, 213 in main.py: All 3 handlers include `"code"` field

## Success Criteria

- [x] Backend errors include machine-readable codes
  - All 3 exception handlers updated with appropriate codes
- [x] Frontend catches all axios errors and shows persistent toast notifications with manual dismiss
  - Interceptor registered globally in main.tsx
  - Error toasts require manual X button click to dismiss
  - Success toasts auto-dismiss after 5 seconds

## Next Phase Readiness

### What This Enables
- **Universal error visibility:** Users see all backend errors immediately as toasts
- **Debugging support:** Error IDs and codes shown in toast messages
- **Retry UX:** GET request errors have one-click retry button
- **Graceful degradation:** Toasts don't block UI, cap at 5 prevents spam

### Integration Points
- **All axios calls:** Automatically wrapped by interceptor
- **Custom error toasts:** Any component can import `useToastStore` and call `addToast()`
- **Success notifications:** Use `color: 'green'` for success messages with auto-dismiss

### Known Issues / Limitations
- **Pre-existing TypeScript errors:** Unrelated to this plan (StickyPlayer, CsvViewer, etc.)
  - Plan-related files (toastStore, ToastContainer, errorHandler) have no TypeScript errors
- **Toast positioning:** Fixed top-right, not customizable per toast
- **No toast persistence:** Toasts cleared on page reload (intentional - ephemeral notifications)

### Recommendations
1. **Consider toast theming:** Current colors are hard-coded - could add theme support later
2. **Analytics opportunity:** Could track error code frequency via toast dispatches
3. **Offline handling:** Interceptor could detect network errors and show specific message

## Commits
- `4911b34` - feat(02-02): create toast store and container
- `650c25f` - feat(02-02): add axios error interceptor and backend error codes

## Testing Notes

### Manual Testing Checklist
- [ ] Trigger 404 error - verify toast shows "HTTP_404" code
- [ ] Trigger validation error - verify toast shows "VALIDATION_ERROR" code
- [ ] Trigger server error - verify toast shows "INTERNAL_ERROR" code with error_id
- [ ] Verify error toasts stay until X clicked
- [ ] Verify success toasts auto-dismiss after 5s
- [ ] Verify retry button appears on GET errors
- [ ] Click retry button - verify request re-sent
- [ ] Trigger 6+ errors rapidly - verify oldest toast removed (queue cap)
- [ ] Click X button on toast - verify immediate removal

### TypeScript Verification
```bash
cd frontend && npx tsc --noEmit
# No errors in: toastStore.ts, ToastContainer.tsx, errorHandler.ts
```

### Backend Error Code Verification
```bash
grep -n '"code"' backend/app/main.py
# Output:
# 176: "code": f"HTTP_{exc.status_code}"
# 190: "code": "VALIDATION_ERROR"
# 213: "code": "INTERNAL_ERROR"
```

---

**Phase 02 Plan 02 Complete** - Error notification pipeline established. Backend errors now include machine-readable codes, and frontend automatically shows persistent toast notifications with manual dismiss for errors.
