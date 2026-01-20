# Feature Landscape: Production-Quality Audio Analysis Tools

**Domain:** Audio annotation / ML-assisted classification tools
**Researched:** 2026-01-20
**Mode:** Features dimension for polish/stability milestone

## Table Stakes

Features users expect from any production-quality ML annotation tool. Missing these = frustration and lost work.

### Error Handling & Recovery

| Feature | Why Expected | Current State | Priority |
|---------|--------------|---------------|----------|
| **Specific error messages** | Users need to know what went wrong and how to fix it | Generic "Error" messages, bare except blocks | HIGH |
| **Actionable feedback** | "File too large (max 2GB)" not just "Upload failed" | Missing context in most errors | HIGH |
| **Graceful degradation** | Backend hiccup shouldn't lose user's work | Autosave exists but errors can disrupt workflow | MEDIUM |
| **Network error recovery** | Retry with backoff, resume interrupted operations | No retry logic, user must manually retry | HIGH |
| **Visual error states** | Red indicators, error icons, disabled states | Some toasts exist but inconsistent | MEDIUM |

### Progress Feedback

| Feature | Why Expected | Current State | Priority |
|---------|--------------|---------------|----------|
| **Accurate progress bars** | Show real completion, not fake animation | Analysis shows file progress but overall % can be jumpy | MEDIUM |
| **Stage indicators** | "Loading... Analyzing... Saving..." | Generic "Loading..." only | MEDIUM |
| **Time estimates** | "~3 min remaining" for long operations | No time estimates | LOW |
| **Cancellation option** | User can abort long-running operations | Batch cancel exists, single analysis cancel missing | MEDIUM |
| **Background operation indicator** | Show when something is processing without blocking UI | Analysis Monitor page exists but requires navigation | LOW |

### Data Safety

| Feature | Why Expected | Current State | Priority |
|---------|--------------|---------------|----------|
| **Autosave** | Never lose more than a few seconds of work | EXISTS - immediate autosave on change | LOW (done) |
| **Undo/redo** | Reverse mistakes without full discard | MISSING - only "Discard all changes" | HIGH |
| **Confirmation dialogs** | Prevent accidental destructive actions | Some exist (delete CSV) but inconsistent | MEDIUM |
| **Backup before overwrite** | Keep original when saving edits | Autosave is separate file, main save overwrites | LOW |

### Performance & Responsiveness

| Feature | Why Expected | Current State | Priority |
|---------|--------------|---------------|----------|
| **Sub-second UI response** | Clicks feel instant | Most interactions fast, waveform loading slow | MEDIUM |
| **Streaming for large files** | Don't freeze loading 40min audio | Basic streaming exists, waveform loads fully | HIGH |
| **Caching** | Don't recompute same data repeatedly | No waveform cache, regenerated each load | HIGH |
| **Optimistic UI updates** | Show changes immediately, sync in background | Direct state updates exist | LOW (done) |

### Keyboard Navigation

| Feature | Why Expected | Current State | Priority |
|---------|--------------|---------------|----------|
| **Spacebar = play/pause** | Universal audio convention | NOT IMPLEMENTED - spacebar does nothing | HIGH |
| **Basic navigation shortcuts** | Arrow keys, Enter, Escape | Only Enter/Escape for delete confirm modal | HIGH |
| **Common editing shortcuts** | Ctrl+S save, Ctrl+Z undo | NOT IMPLEMENTED | HIGH |
| **Focus management** | Tab through form fields logically | Browser default only | MEDIUM |

## Differentiators

Features that would make Filharmonia feel polished and professional. Not expected but valued.

### Workflow Efficiency

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **One-click workflow** | "Sort + Analyze + Open" in single action | Medium | Current: Sort, wait, navigate to Monitor, wait, navigate to Editor |
| **Smart defaults** | Pre-select likely actions based on state | Low | e.g., auto-select "ready to move" files already done |
| **Batch operations with preview** | See what will happen before executing | Medium | Current batch analyze shows nothing until started |
| **Quick segment class cycling** | Tab/number keys to change class quickly | Low | Currently requires dropdown selection |
| **Jump to uncertain segments** | Navigate directly to low-confidence predictions | Low | Uncertainty Review exists but separate page |

### Audio Editing UX

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Waveform click-to-seek** | Click anywhere to jump to that time | Low | EXISTS but could be more responsive |
| **Drag boundary handles** | Adjust segment boundaries by dragging | Medium | Currently text input only |
| **Split segment at playhead** | Press key to split current segment | Low | Would speed up correction workflow |
| **Zoom to selection** | Double-click segment to zoom | Low | Zoom exists but manual only |
| **Minimap/overview** | See full recording while zoomed | Medium | Standard in DAWs like Audacity |

### Visual Polish

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Consistent color coding** | Same colors everywhere (list, waveform, legend) | Low | Colors exist but legend placement varies |
| **Segment labels on waveform** | See class names without hovering | Low | Currently color-only |
| **Current segment highlight** | Playing segment visually distinct | Low | Row highlight exists in table |
| **Animation for state changes** | Smooth transitions, not jarring jumps | Low | Some transitions exist |

### Export & Integration

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Copy tracklist to clipboard** | Quick export for email/document | Low | EXISTS |
| **Export formats selection** | CSV, JSON, plain text | Low | Single format currently |
| **Export preview** | See exactly what will be copied | Low | Currently just count |

## Anti-Features

Features to explicitly NOT build. Common over-engineering traps for this type of tool.

### Multi-User Collaboration

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Real-time collaboration | Single-user tool, adds massive complexity | Keep simple file-based workflow |
| User accounts/auth | Local trusted network, no threat model | Document intended deployment |
| Role-based permissions | One person uses this | Single user assumptions OK |
| Audit trails | No compliance requirements | Simple autosave is sufficient |

### Premature Infrastructure

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Database for job status | File-based works fine for single user | Clean up /tmp files instead |
| WebSocket real-time updates | Polling works, adds deployment complexity | Optimize polling intervals |
| Microservices architecture | Single backend handles load easily | Keep monolith, refactor modules |
| Container orchestration | Overkill for 1-3 machines | Docker compose at most |

### Feature Creep

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| In-app audio editing (trim, effects) | Scope explosion, Audacity exists | Focus on annotation only |
| Automatic music identification | Complex, unreliable, separate concern | Keep as future milestone |
| Report generation | Simple copy-paste works for ZAIKS | Leave to next milestone |
| Spectrogram view | Waveform sufficient for this use case | Avoid unless specifically needed |

### Over-Engineering the UI

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Customizable themes | Maintenance burden, low value | One good theme |
| Configurable keyboard shortcuts | Complexity for power users only | Fixed sensible defaults |
| Drag-and-drop everything | More fragile than direct interaction | Keep forms and buttons |
| Infinite undo history | Memory bloat, complex state management | 10-20 step buffer is plenty |

## Feature Dependencies

```
Core Stability (Phase 1)
    |
    +-- Error handling improvements
    +-- Network retry logic
    +-- Specific error messages
    |
    v
Basic Polish (Phase 2)
    |
    +-- Keyboard shortcuts (spacebar play/pause first)
    +-- Undo/redo (single step initially)
    +-- Progress improvements
    |
    v
Workflow Efficiency (Phase 3)
    |
    +-- One-click workflows
    +-- Batch operation previews
    +-- Quick class cycling
```

## MVP Recommendation for Polish Milestone

**Prioritize these first (table stakes gaps):**

1. **Keyboard shortcuts** - Spacebar play/pause, Ctrl+S save, Ctrl+Z undo
   - Reason: Audio tools without spacebar feel broken
   - Effort: Low (event handlers only)

2. **Specific error messages** - Replace "Error" with actionable text
   - Reason: Users currently guess what went wrong
   - Effort: Low (string changes, add error context)

3. **Undo/redo for segment edits** - At least single-step undo
   - Reason: Users fear making changes that can't be reversed
   - Effort: Medium (state history management)

4. **Waveform caching** - Don't regenerate on every load
   - Reason: 30-40 min files = slow loads currently
   - Effort: Medium (cache layer, invalidation logic)

**Defer these (nice-to-have):**

- Time estimates for operations
- Drag boundary handles
- Minimap/overview waveform
- Export format selection
- Custom keyboard shortcuts

## Sources

### Audio Annotation Best Practices
- [Annotationbox - Audio Annotation Tools and Best Practices](https://annotationbox.com/audio-annotation-tools-and-best-practices/)
- [Label Studio Blog - Labeling Audio Data](https://labelstud.io/blog/labeling-audio-data-with-label-studio/)
- [Encord - Annotate Audio Files](https://encord.com/blog/annotate-audio/)

### Error Handling & UX
- [Google PAIR - Errors + Graceful Failure](https://pair.withgoogle.com/chapter/errors-failing/)
- [Pencil & Paper - Error Message UX](https://www.pencilandpaper.io/articles/ux-pattern-analysis-error-feedback)
- [MoldStud - Impact of Error Handling on UX](https://moldstud.com/articles/p-the-impact-of-error-handling-on-user-experience)

### Progress & File Upload
- [Uploadcare - File Uploader UX Best Practices](https://uploadcare.com/blog/file-uploader-ux-best-practices/)
- [CLIMB - 10 File Upload UX Best Practices](https://climbtheladder.com/10-file-upload-ux-best-practices/)
- [BricxLabs - 9 Progress Bar UX Examples](https://bricxlabs.com/blogs/progress-bar-ux-examples)

### Keyboard Shortcuts & Workflow
- [Noble Desktop - Audio Workflows with Keyboard Shortcuts](https://www.nobledesktop.com/learn/adobe-audition/improving-your-audio-workflows-with-keyboard-shortcuts-in-adobe-audition)
- [Splice Blog - Pro Tools Shortcuts Guide](https://splice.com/blog/pro-tools-shortcuts-guide/)
- [Prodigy - Audio and Video Annotation](https://prodi.gy/docs/audio-video)

### Undo/Redo Patterns
- [esveo - Undo, Redo, and the Command Pattern](https://www.esveo.com/en/blog/undo-redo-and-the-command-pattern/)
- [DEV Community - You Don't Know Undo/Redo](https://dev.to/isaachagoel/you-dont-know-undoredo-4hol)

### Batch Processing
- [iZotope RX - Batch Processor](https://www.izotope.com/en/products/rx/features/batch-processing.html)
- [NUGEN Audio - AMB Scalable Audio Management](https://nugenaudio.com/amb/)

### Anti-Patterns
- [Baeldung - What Is an Anti-pattern](https://www.baeldung.com/cs/anti-patterns)
- [DEV Community - Anti-patterns Every Developer Should Know](https://dev.to/yogini16/anti-patterns-that-every-developer-should-know-4nph)
- [GeeksforGeeks - Types of Anti Patterns to Avoid](https://www.geeksforgeeks.org/blogs/types-of-anti-patterns-to-avoid-in-software-development/)
