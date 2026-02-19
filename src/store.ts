import { create } from 'zustand';
import type { Tool, EffectDefinition, StrokeRecord, ProjectMeta } from './types';

const MAX_UNDO = 20;

interface AppState {
  // Tools
  currentTool: Tool;
  brushWidth: number;
  brushColor: string;
  brushOpacity: number;

  // Effects
  availableEffects: EffectDefinition[];
  currentEffect: EffectDefinition | null;
  effectParams: Record<string, unknown>;

  // Project
  currentProject: ProjectMeta | null;
  projectFilePath: string | null;

  // Strokes
  strokes: StrokeRecord[];

  // Canvas (non-reactive — stored outside React render cycle)
  _canvasInstance: unknown | null;

  // Collected path/click data for the current stroke
  currentPaths: number[][][];
  currentClicks: number[][];

  // Undo / Redo (JSON snapshots of canvas state)
  undoStack: string[];
  redoStack: string[];

  // Zoom
  zoomLevel: number;

  // UI toggles
  strokeManagerOpen: boolean;
  projectDialogOpen: boolean;
  projectDialogMode: 'new' | 'open' | null;
  notification: { message: string; type: 'info' | 'error' | 'success' } | null;

  // Actions
  setTool: (tool: Tool) => void;
  setBrushWidth: (width: number) => void;
  setBrushColor: (color: string) => void;
  setBrushOpacity: (opacity: number) => void;
  setCurrentEffect: (effect: EffectDefinition | null) => void;
  setEffectParam: (key: string, value: unknown) => void;
  loadEffects: (effects: EffectDefinition[]) => void;
  addStroke: (stroke: StrokeRecord) => void;
  updateStroke: (id: string, updates: Partial<StrokeRecord>) => void;
  setProject: (project: ProjectMeta | null) => void;
  setProjectFilePath: (filePath: string | null) => void;
  setCanvasInstance: (canvas: unknown) => void;
  addPath: (path: number[][]) => void;
  addClick: (point: number[]) => void;
  clearCurrentStrokeData: () => void;
  pushUndoState: (json: string) => void;
  undo: () => string | null;
  redo: () => string | null;
  setZoomLevel: (level: number) => void;
  toggleStrokeManager: () => void;
  openProjectDialog: (mode: 'new' | 'open') => void;
  closeProjectDialog: () => void;
  notify: (message: string, type: 'info' | 'error' | 'success') => void;
  clearNotification: () => void;
}

export const useStore = create<AppState>((set, get) => ({
  // Initial state
  currentTool: 'brush',
  brushWidth: 5,
  brushColor: '#000000',
  brushOpacity: 100,

  availableEffects: [],
  currentEffect: null,
  effectParams: {},

  currentProject: null,
  projectFilePath: null,

  strokes: [],

  _canvasInstance: null,
  currentPaths: [],
  currentClicks: [],

  undoStack: [],
  redoStack: [],

  zoomLevel: 1,

  strokeManagerOpen: false,
  projectDialogOpen: false,
  projectDialogMode: null,
  notification: null,

  // Actions
  setTool: (tool) => set({ currentTool: tool }),
  setBrushWidth: (width) => set({ brushWidth: width }),
  setBrushColor: (color) => set({ brushColor: color }),
  setBrushOpacity: (opacity) => set({ brushOpacity: opacity }),

  setCurrentEffect: (effect) => {
    if (effect) {
      const params: Record<string, unknown> = {};
      for (const [key, def] of Object.entries(effect.user_input)) {
        params[key] = def.default;
      }
      set({ currentEffect: effect, effectParams: params });
    } else {
      set({ currentEffect: null, effectParams: {} });
    }
  },

  setEffectParam: (key, value) =>
    set((state) => ({ effectParams: { ...state.effectParams, [key]: value } })),

  loadEffects: (effects) => set({ availableEffects: effects }),

  addStroke: (stroke) => set((state) => ({ strokes: [...state.strokes, stroke] })),

  updateStroke: (id, updates) =>
    set((state) => ({
      strokes: state.strokes.map((s) => (s.id === id ? { ...s, ...updates } : s)),
    })),

  setProject: (project) => set({ currentProject: project }),
  setProjectFilePath: (filePath) => set({ projectFilePath: filePath }),
  setCanvasInstance: (canvas) => set({ _canvasInstance: canvas }),

  addPath: (path) => set((state) => ({ currentPaths: [...state.currentPaths, path] })),
  addClick: (point) => set((state) => ({ currentClicks: [...state.currentClicks, point] })),
  clearCurrentStrokeData: () => set({ currentPaths: [], currentClicks: [] }),

  pushUndoState: (json) =>
    set((state) => ({
      undoStack: [...state.undoStack.slice(-(MAX_UNDO - 1)), json],
      redoStack: [],
    })),

  undo: () => {
    const { undoStack } = get();
    if (undoStack.length < 2) return null;
    const current = undoStack[undoStack.length - 1];
    const previous = undoStack[undoStack.length - 2];
    set((state) => ({
      undoStack: state.undoStack.slice(0, -1),
      redoStack: [...state.redoStack, current],
    }));
    return previous;
  },

  redo: () => {
    const { redoStack } = get();
    if (redoStack.length === 0) return null;
    const next = redoStack[redoStack.length - 1];
    set((state) => ({
      redoStack: state.redoStack.slice(0, -1),
      undoStack: [...state.undoStack, next],
    }));
    return next;
  },

  setZoomLevel: (level) => set({ zoomLevel: level }),
  toggleStrokeManager: () => set((state) => ({ strokeManagerOpen: !state.strokeManagerOpen })),
  openProjectDialog: (mode) => set({ projectDialogOpen: true, projectDialogMode: mode }),
  closeProjectDialog: () => set({ projectDialogOpen: false, projectDialogMode: null }),

  notify: (message, type) => set({ notification: { message, type } }),
  clearNotification: () => set({ notification: null }),
}));
