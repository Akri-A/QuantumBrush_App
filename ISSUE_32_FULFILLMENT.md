# Issue #32 Fulfillment — [Refactoring] Improve UX/UI of the App with Code Implementation

> Reference: https://github.com/moth-quantum/QuantumBrush/issues/32

This document outlines how the Electron + React rewrite of Quantum Brush addresses every requirement from Issue #32.

---

## 1. (10 pts) Improve Window Arrangement & UI/UX Workflow

**Problem:** The original Java app scattered its interface across three separate windows (Control Panel, Canvas, Stroke Manager), making the workflow confusing and cluttered.

**Solution:** The rewrite consolidates everything into a **single, responsive Electron window** (default 1400x900, min 900x600) with a clean layout:

| Region | Position | Width | Purpose |
|---|---|---|---|
| Sidebar | Left | 56px fixed | Tool selection, import/export, project actions |
| Canvas | Center | Flexible (`flex-1`) | Fabric.js drawing surface with zoom/pan |
| Control Panel | Right | 320px fixed | Brush settings, effect selector, dynamic parameters |
| Stroke Manager | Bottom overlay | Full width, 192px | Collapsible stroke history drawer |
| Title Bar | Top | Full width, 32px | Project name and file path display |

**Key improvements over the original:**
- All panels live within one window — no scattered floating dialogs
- The canvas fills all available space and is fully responsive to any screen size
- The Stroke Manager sits as a collapsible bottom drawer, accessible but out of the way
- Flexbox-based layout adapts cleanly across Windows, macOS, and Linux
- Space+drag panning and scroll-wheel zoom provide smooth canvas navigation
- Keyboard shortcuts (V/B/E/D for tools, Ctrl+Z/Y for undo/redo, Ctrl+S to save) streamline the workflow

**Relevant files:**
- `src/App.tsx` — Root layout composition
- `src/components/CanvasArea.tsx` — Responsive canvas with zoom/pan
- `src/components/Sidebar.tsx` — Left toolbar
- `src/components/ControlPanel.tsx` — Right panel
- `src/components/StrokeManager.tsx` — Bottom drawer

---

## 2. (10 pts) Improve the JSON Communication System

**Problem:** The original app used a cumbersome JSON document exchange between Java and Python, making graphical execution slow and inconvenient.

**Solution:** The rewrite implements a structured IPC pipeline with SVG support:

```
React UI → IPC (contextBridge) → Electron Main Process → Python subprocess → PNG + JSON files
```

**How it works:**
1. The canvas state is serialized as a **PNG data URL** and saved to disk
2. An **instruction JSON** file is written with all parameters:
   ```json
   {
     "effect_id": "heisenbrush",
     "stroke_id": "stroke_1234",
     "project_id": "proj_5678",
     "user_input": { "Radius": 5, "Strength": 0.5, "Color": "#FF0000" },
     "stroke_input": {
       "path": [[x1, y1], [x2, y2]],
       "clicks": [[cx, cy]]
     }
   }
   ```
3. Python is spawned as a subprocess with a 3-minute timeout
4. Python reads the instruction, runs the effect, produces an output PNG (diff mask — only changed pixels)
5. The output is returned to the renderer as a base64 data URL and composited on the canvas

**SVG support:**
- Stroke paths are archived as SVG files per-stroke via `saveStrokeSvg()`
- Full SVG import/export is supported through native file dialogs (`importSvg` / `exportSvg`)
- SVG data is generated from Fabric.js path objects, providing a clean vector representation of user strokes

**Relevant files:**
- `electron/main.ts` — IPC handler registration
- `electron/preload.ts` — Typed contextBridge API
- `electron/python.ts` — Python subprocess spawner, JSON/PNG/SVG I/O
- `python/apply_effect.py` — Python entry point for effect execution

---

## 3. (20 pts) New Design Language

**Problem:** The original app used Java Swing's default look, making it feel like software from the early 2010s.

**Solution:** A complete modern design system built with **Tailwind CSS v4**, documented in `design/DESIGN_SPEC.md`:

### Color Palette
| Layer | Tailwind Class | Hex |
|---|---|---|
| App background | `bg-gray-950` | `#0a0a0f` |
| Panel background | `bg-gray-900/80` | `rgba(17,24,39,0.8)` |
| Surface/Input | `bg-gray-800/80` | `rgba(31,41,55,0.8)` |
| Borders | `border-white/10` | `rgba(255,255,255,0.1)` |
| Primary accent | `bg-blue-600` | `#2563eb` |
| Active glow | `shadow-blue-600/30` | `rgba(37,99,235,0.3)` |

### Design Principles
- **Dark mode only** — optimized for creative work and reduced eye strain
- **Glassmorphism** — panels use `backdrop-blur-xl` with semi-transparent backgrounds for depth
- **Inter font family** — clean, modern typography with system fallbacks
- **Lucide React icons** — consistent, lightweight icon set throughout the UI
- **Thin scrollbars** — minimal visual noise
- **Toast notifications** — slide-in animations for feedback (success/error/info)
- **Cross-platform consistency** — looks great on Windows, macOS, and Linux

**Relevant files:**
- `design/DESIGN_SPEC.md` — Full design specification
- `src/index.css` — Global styles, font, color scheme, animations
- `tailwind.config.js` — Tailwind configuration

---

## 4. (60 pts) Full Desktop Application Implementation

**Problem:** The original Java + Processing stack was limiting for modern graphical software development.

**Solution:** A complete rewrite using a modern tech stack:

### Tech Stack
| Technology | Role |
|---|---|
| **Electron** | Cross-platform desktop shell |
| **React 18** | UI framework |
| **TypeScript** | Type-safe development |
| **Vite** | Fast bundler with HMR |
| **Fabric.js v7** | HTML5 canvas engine |
| **Tailwind CSS v4** | Utility-first styling |
| **Zustand** | Lightweight state management |
| **electron-builder** | Packaging for macOS (.dmg), Windows (NSIS), Linux (AppImage) |

### Basic Features Implemented

#### Creating a New Project
- `ProjectDialog.tsx` provides a modal with project name, canvas width, and height inputs
- Projects are stored internally with metadata at `project/{id}/metadata.json`
- Each project gets its own `stroke/` subdirectory for effect data

#### Importing an Image
- Native file dialog via `ipcRenderer.importImage()` (PNG, JPEG, GIF, BMP, WebP)
- Drag-and-drop support via `react-dropzone` (accepts `image/*` and `.svg`)
- Images are scaled to fit the canvas and inserted at layer index 0 (behind strokes)
- SVG import also supported via `ipcRenderer.importSvg()`

#### Drawing on the Canvas
| Tool | Description |
|---|---|
| **Brush** | Free-drawing with `PencilBrush`, configurable color/width/opacity |
| **Eraser** | Manual hit-test removal of objects under cursor |
| **Dot** | Single-click circle placement for point-based effects |
| **Select** | Native Fabric.js selection with delete/backspace support |

Additional canvas features:
- Scroll-wheel zoom (0.1x to 20x)
- Space+drag panning
- Full undo/redo stack (20 states)

#### Saving for Execution
- **Internal save:** Canvas state serialized via `canvas.toJSON()` and stored in project directory
- **File save:** `.qbrush` file format (versioned JSON wrapper around canvas state)
- **Export:** PNG and SVG export via native Save dialogs
- **Effect execution:** One-click "Apply Effect" button runs the Python backend with all stroke/parameter data

#### Effect Plugin System
Effects are dynamically discovered from `python/{effectId}/` directories, each containing:
- `{effectId}_requirements.json` — schema defining name, description, parameters, and input types
- `{effectId}.py` — Python implementation with `run(req) -> np.ndarray`

Available effects include: `acrylic`, `chemical`, `clone`, `damping`, `GoL`, `heisenbrush`, `heisenbrush2`, `pointillism`, `qdrop`, `steerable`.

The Control Panel dynamically generates parameter controls (sliders, color pickers, toggles) based on each effect's requirements JSON.

### Cross-Platform Distribution
Configured via `electron-builder.json5`:
- **macOS:** `.dmg` installer
- **Windows:** NSIS installer (x64)
- **Linux:** AppImage

**Relevant files:**
- `electron/main.ts` — Electron window, menus, IPC handlers
- `electron/preload.ts` — Secure contextBridge API
- `electron/python.ts` — Python subprocess management
- `src/App.tsx` — Root application layout
- `src/store.ts` — Zustand global state
- `src/types.ts` — TypeScript type definitions
- `src/components/CanvasArea.tsx` — Full canvas implementation
- `src/components/Sidebar.tsx` — Toolbar and actions
- `src/components/ControlPanel.tsx` — Effect parameters and brush settings
- `src/components/StrokeManager.tsx` — Stroke history
- `src/components/ProjectDialog.tsx` — Project creation/open dialog
- `src/components/Notification.tsx` — Toast notification system
- `python/apply_effect.py` — Effect execution engine
- `vite.config.ts` — Build configuration
- `electron-builder.json5` — Packaging configuration

---

## Summary

| Requirement | Points | Status |
|---|---|---|
| Improve window arrangement and UI/UX workflow | 10 | Fulfilled — Single responsive window with dedicated panel layout |
| Improve JSON communication system (SVG support) | 10 | Fulfilled — Structured IPC pipeline with SVG archival and import/export |
| Design new design language | 20 | Fulfilled — Modern dark glassmorphism design with full spec document |
| Build desktop app with modern framework | 60 | Fulfilled — Complete Electron + React implementation with all basic features |
| **Total** | **100** | **All requirements addressed** |
