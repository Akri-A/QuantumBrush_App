import { useEffect } from 'react';
import { useStore } from './store';
import Sidebar from './components/Sidebar';
import CanvasArea from './components/CanvasArea';
import ControlPanel from './components/ControlPanel';
import StrokeManager from './components/StrokeManager';
import ProjectDialog from './components/ProjectDialog';
import Notification from './components/Notification';

function App() {
  const currentProject = useStore((s) => s.currentProject);
  const projectFilePath = useStore((s) => s.projectFilePath);
  const notify = useStore((s) => s.notify);

  // Check Python environment on startup
  useEffect(() => {
    window.ipcRenderer.checkPython().then((result) => {
      if (!result.success) return;
      const data = result.data!;
      if (!data.available) {
        notify('python3 not found. Install Python 3 to use effects.', 'error');
      } else if (data.missing && data.missing.length > 0) {
        notify(`Missing Python packages: ${data.missing.join(', ')}`, 'error');
      }
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="flex h-screen w-screen bg-gray-950 text-white overflow-hidden">
      {/* Left Sidebar — Tools & Actions */}
      <Sidebar />

      {/* Main area */}
      <div className="flex-1 flex flex-col relative">
        {/* Project title bar */}
        <div className="h-8 bg-gray-900/50 border-b border-white/5 flex items-center px-4 shrink-0 select-none">
          <span className="text-xs text-gray-500 truncate">
            {currentProject
              ? `${currentProject.name}${projectFilePath ? ` — ${projectFilePath}` : ''}`
              : 'Quantum Brush — No project'}
          </span>
        </div>

        {/* Canvas + Stroke Manager */}
        <div className="flex-1 relative">
          <CanvasArea />
          <StrokeManager />
        </div>
      </div>

      {/* Right Control Panel */}
      <ControlPanel />

      {/* Overlays */}
      <ProjectDialog />
      <Notification />
    </div>
  );
}

export default App;
