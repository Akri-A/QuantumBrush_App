import { useState } from 'react';
import { X, FileUp } from 'lucide-react';
import { useStore } from '../store';
import type { ProjectMeta } from '../types';

const ProjectDialog = () => {
  const isOpen = useStore((s) => s.projectDialogOpen);
  const mode = useStore((s) => s.projectDialogMode);
  const closeDialog = useStore((s) => s.closeProjectDialog);
  const setProject = useStore((s) => s.setProject);
  const notify = useStore((s) => s.notify);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="bg-gray-900 border border-white/10 rounded-xl shadow-2xl w-full max-w-md mx-4">
        {mode === 'new' ? (
          <NewProjectForm
            onCreated={(meta) => {
              setProject(meta);
              closeDialog();
              notify(`Project "${meta.name}" created`, 'success');
            }}
            onClose={closeDialog}
          />
        ) : (
          <OpenProjectForm
            onOpened={(meta) => {
              setProject(meta);
              closeDialog();
              notify(`Opened "${meta.name}"`, 'success');
            }}
            onClose={closeDialog}
          />
        )}
      </div>
    </div>
  );
};

function NewProjectForm({
  onCreated,
  onClose,
}: {
  onCreated: (meta: ProjectMeta) => void;
  onClose: () => void;
}) {
  const [name, setName] = useState('');
  const [width, setWidth] = useState(1200);
  const [height, setHeight] = useState(800);
  const [loading, setLoading] = useState(false);

  const handleCreate = async () => {
    if (!name.trim()) return;
    setLoading(true);
    const result = await window.ipcRenderer.createProject(name.trim(), width, height);
    setLoading(false);
    if (result.success && result.data) {
      onCreated(result.data);
    } else {
      useStore.getState().notify('Failed to create project: ' + (result.error || 'Unknown error'), 'error');
    }
  };

  return (
    <>
      <div className="flex items-center justify-between px-5 py-4 border-b border-white/10">
        <h3 className="text-white font-semibold">New Project</h3>
        <button onClick={onClose} className="text-gray-400 hover:text-white transition-colors">
          <X size={18} />
        </button>
      </div>
      <div className="p-5 space-y-4">
        <div className="space-y-1.5">
          <label className="text-xs text-gray-400 font-medium">Project Name</label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="My Quantum Art"
            autoFocus
            className="w-full bg-gray-800 border border-white/10 text-white rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-blue-500 transition-colors"
            onKeyDown={(e) => e.key === 'Enter' && handleCreate()}
          />
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-1.5">
            <label className="text-xs text-gray-400 font-medium">Width (px)</label>
            <input
              type="number"
              value={width}
              onChange={(e) => setWidth(Number(e.target.value))}
              min={100}
              max={4000}
              className="w-full bg-gray-800 border border-white/10 text-white rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-blue-500"
            />
          </div>
          <div className="space-y-1.5">
            <label className="text-xs text-gray-400 font-medium">Height (px)</label>
            <input
              type="number"
              value={height}
              onChange={(e) => setHeight(Number(e.target.value))}
              min={100}
              max={4000}
              className="w-full bg-gray-800 border border-white/10 text-white rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-blue-500"
            />
          </div>
        </div>
      </div>
      <div className="px-5 py-4 border-t border-white/10 flex justify-end gap-2">
        <button
          onClick={onClose}
          className="px-4 py-2 text-sm text-gray-300 hover:text-white transition-colors rounded-lg"
        >
          Cancel
        </button>
        <button
          onClick={handleCreate}
          disabled={loading || !name.trim()}
          className="px-4 py-2 text-sm bg-blue-600 hover:bg-blue-700 disabled:bg-blue-600/50 text-white rounded-lg font-medium transition-colors"
        >
          {loading ? 'Creating...' : 'Create'}
        </button>
      </div>
    </>
  );
}

function OpenProjectForm({
  onOpened,
  onClose,
}: {
  onOpened: (meta: ProjectMeta) => void;
  onClose: () => void;
}) {
  const notify = useStore((s) => s.notify);

  const handleOpenFromFile = async () => {
    const result = await window.ipcRenderer.openFromFile();
    if (result.success && result.data) {
      const canvas = useStore.getState()._canvasInstance as any;
      if (canvas && result.data.canvasJson) {
        await canvas.loadFromJSON(JSON.parse(result.data.canvasJson));
        canvas.renderAll();
      }
      useStore.getState().setProjectFilePath(result.data.filePath);
      onOpened(result.data.meta);
    } else if (result.error && result.error !== 'cancelled') {
      notify('Failed to open file: ' + result.error, 'error');
    }
  };

  return (
    <>
      <div className="flex items-center justify-between px-5 py-4 border-b border-white/10">
        <h3 className="text-white font-semibold">Open Project</h3>
        <button onClick={onClose} className="text-gray-400 hover:text-white transition-colors">
          <X size={18} />
        </button>
      </div>

      <div className="p-5">
        <button
          onClick={handleOpenFromFile}
          className="w-full flex items-center gap-2 px-3 py-2.5 rounded-lg border border-dashed border-white/10 hover:border-blue-500/40 hover:bg-blue-500/5 text-gray-400 hover:text-blue-400 transition-all"
        >
          <FileUp size={16} />
          <span className="text-sm">Open .qbrush file from disk...</span>
        </button>
      </div>

      <div className="px-5 py-3 border-t border-white/10 flex justify-end">
        <button
          onClick={onClose}
          className="px-4 py-2 text-sm text-gray-300 hover:text-white transition-colors rounded-lg"
        >
          Close
        </button>
      </div>
    </>
  );
}

export default ProjectDialog;
