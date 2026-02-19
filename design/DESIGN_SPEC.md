# Quantum Brush — Design Specification

> This document describes the design language, layout decisions, and component specifications for the modernized Quantum Brush desktop application. It serves as the design deliverable for the 20-pt NO CODE criterion of Issue #32.

## Design Philosophy

The design moves away from Java SwingX's dated look (separate floating windows, system-default widgets, no visual hierarchy) toward a **modern, dark-mode-first, single-window workspace** inspired by professional creative tools (Figma, Blender, VS Code).

Key principles:
- **Unified workspace** — everything in one responsive window, no scattered dialogs
- **Glassmorphism** — translucent panels with backdrop blur for depth
- **Information density** — compact but readable, no wasted space
- **Progressive disclosure** — advanced features (Stroke Manager, effect descriptions) are toggleable

---

## Color Palette

| Token | Hex | Usage |
|---|---|---|
| Background | `#0a0a0f` | App body, canvas surround |
| Panel BG | `rgba(17,24,39, 0.8)` | Sidebar, ControlPanel, StrokeManager (gray-900/80) |
| Surface | `rgba(31,41,55, 0.8)` | Input fields, cards (gray-800/80) |
| Border | `rgba(255,255,255, 0.1)` | Panel edges (white/10) |
| Primary | `#2563eb` | Active tool, buttons, accents (blue-600) |
| Primary Hover | `#1d4ed8` | Button hover states (blue-700) |
| Primary Glow | `rgba(37,99,235, 0.3)` | Active tool shadow (blue-600/30) |
| Text Primary | `#ffffff` | Headings, active labels |
| Text Secondary | `#9ca3af` | Descriptions, inactive labels (gray-400) |
| Text Muted | `#6b7280` | Hints, timestamps (gray-500) |
| Success | `#4ade80` | Completed strokes (green-400) |
| Error | `#f87171` | Failed strokes, errors (red-400) |
| Info | `#60a5fa` | Notifications (blue-400) |

## Typography

- **Font family**: `Inter, system-ui, Avenir, Helvetica, Arial, sans-serif`
- **Headings**: 14px semibold (panel titles)
- **Labels**: 11px medium uppercase tracking-wider (section labels)
- **Body**: 13px regular (parameters, descriptions)
- **Mono**: System monospace (color hex values, numeric displays)

---

## Layout Structure

```
+--------+---------------------------------------------+----------+
| Sidebar|              Title Bar                       | Control  |
| (56px) |              (32px)                           | Panel    |
|        +---------------------------------------------+| (320px)  |
|  Tools |                                              ||          |
|        |              Canvas Area                     || Effect   |
| Import |              (Fabric.js)                     || Selector |
| Export |                                              ||          |
|        |              flex-1                          || Dynamic  |
| Stroke |                                              || Params   |
| Mgr    |                                              ||          |
|        +----------------------------------------------+| Apply    |
|  ---   |     Stroke Manager (collapsible, 192px)      || Button   |
| NewProj|     Bottom drawer, toggleable                ||          |
| OpenPrj|                                              ||          |
| Save   |                                              ||          |
+--------+----------------------------------------------+----------+
```

### Sidebar (56px wide)
- **Top section**: Tool buttons in a vertical stack with 4px gap
  - Each button: 36x36px, 8px border-radius
  - Active state: blue-600 background with drop shadow
  - Inactive state: gray-400 icon, hover → white icon + white/10 background
- **Middle section**: Action buttons (Import, Export, Stroke Manager)
  - Separated by 1px horizontal dividers (white/10)
- **Bottom section**: Project actions (New, Open, Save)
  - Save only visible when a project is open

### Canvas Area (flex-1)
- **Title bar**: 32px height, shows project name or "No project"
- **Canvas**: White (#ffffff) Fabric.js canvas, fills available space
- **Zoom indicator**: Bottom-right corner hint text (gray-500 on gray-900/60)
- **Drop zone overlay**: Blue-600/20 with dashed blue-400 border when dragging files

### Control Panel (320px wide)
- **Header**: "Control Panel" with Sliders icon (blue-400)
- **Brush section**: Width slider (1-50px) + color picker
- **Effect section**: Dropdown populated from `*_requirements.json` files
  - Shows effect name + author
  - Toggleable description text
- **Parameters section**: Generated dynamically per effect type
  - `int`/`float` → range slider with min/max/value display
  - `color` → native color picker + hex display
  - `bool` → checkbox toggle
- **Stroke info**: Path count and point count for current stroke
- **Apply button**: Full-width, blue-600, shows spinner during execution

### Stroke Manager (192px tall, bottom drawer)
- Toggleable via Sidebar button
- Header with stroke count badge
- Scrollable list of strokes, newest first
- Each stroke row shows:
  - Status icon (Clock/Spinner/Check/X)
  - Effect name + timestamp
  - Color-coded left border by status
  - Error message if failed

---

## Component States

### Tool Buttons
| State | Background | Text Color | Shadow |
|---|---|---|---|
| Default | transparent | gray-400 | none |
| Hover | white/10 | white | none |
| Active | blue-600 | white | blue-600/30 |

### Buttons (Apply, Save, etc.)
| State | Background | Text Color |
|---|---|---|
| Default | blue-600 | white |
| Hover | blue-700 | white |
| Disabled/Loading | blue-600/50 | white |

### Input Fields
| State | Border | Background |
|---|---|---|
| Default | white/10 | gray-800/80 |
| Focus | blue-500 | gray-800/80 |

### Notifications (Toasts)
| Type | Border | Background | Icon |
|---|---|---|---|
| Success | green-500/20 | green-900/20 | CheckCircle (green-400) |
| Error | red-500/20 | red-900/20 | XCircle (red-400) |
| Info | blue-500/20 | blue-900/20 | Info (blue-400) |

---

## Dialog Designs

### New Project Dialog
- Modal overlay: black/60 + backdrop-blur
- Card: gray-900, white/10 border, 12px radius, max-width 448px
- Fields: Project name (text), Width (number), Height (number)
- Actions: Cancel (text button), Create (blue-600 button)

### Open Project Dialog
- Same modal overlay
- Scrollable list of projects with hover highlight
- Each row: folder icon, project name, dimensions, last updated date
- Delete button appears on hover (gray → red-400)

---

## Interaction Patterns

### Canvas Tools
| Tool | Cursor | Canvas Mode | Action |
|---|---|---|---|
| Select (V) | default | selection on | Click to select, drag to move |
| Brush (B) | crosshair | drawing mode | Click+drag to draw paths |
| Eraser (E) | pointer | selection off | Click on object to remove |
| Dot (D) | crosshair | selection off | Click to place a point |

### Canvas Navigation
| Input | Action |
|---|---|
| Scroll wheel | Zoom in/out (0.1x–20x) |
| Ctrl + / Ctrl - | Zoom in/out |
| Ctrl 0 | Reset zoom to 100% |
| Space + drag | Pan canvas |
| Ctrl+Z | Undo (20-state history) |
| Ctrl+Shift+Z / Ctrl+Y | Redo |
| Ctrl+S | Save project |
| Delete / Backspace | Remove selected objects |

### Effect Application Flow
1. Draw strokes (brush) or place points (dot) on the canvas
2. Select an effect from the Control Panel dropdown
3. Adjust parameters using dynamically generated controls
4. Click "Apply Effect" → loading spinner shown
5. Python backend processes → output overlay composited onto canvas
6. Stroke appears in Stroke Manager with completion status

---

## Responsive Behavior

- **Minimum window**: 900x600px (enforced by Electron `minWidth`/`minHeight`)
- **Canvas**: Resizes to fill available space between sidebar and control panel
- **Stroke Manager**: Fixed 192px height when open, 0px when closed
- **Control Panel**: Fixed 320px width, scrollable content area
- **Sidebar**: Fixed 56px width

---

## Cross-Platform Considerations

- Dark mode only (avoids OS-level light/dark mode inconsistencies)
- System font stack with Inter as primary (widely available, excellent readability)
- Native file dialogs for import/export (via Electron `dialog`)
- Standard keyboard shortcuts (Ctrl on Windows/Linux, Cmd on macOS handled via `metaKey`)
- Thin scrollbars via CSS `scrollbar-width: thin`

---

## Accessibility

- All interactive elements have `title` attributes for tooltips
- Keyboard shortcuts for all tools (V/B/E/D)
- Focus-visible outlines on form inputs (blue-500 border)
- Sufficient contrast ratios (white text on dark backgrounds)
