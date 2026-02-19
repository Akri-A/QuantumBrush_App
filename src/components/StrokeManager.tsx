import { Layers, CheckCircle2, XCircle, Loader2, Clock } from 'lucide-react';
import { useStore } from '../store';

const statusIcons = {
  pending: <Clock size={14} className="text-gray-400" />,
  running: <Loader2 size={14} className="text-blue-400 animate-spin" />,
  completed: <CheckCircle2 size={14} className="text-green-400" />,
  failed: <XCircle size={14} className="text-red-400" />,
};

const statusColors = {
  pending: 'border-gray-700',
  running: 'border-blue-500/30',
  completed: 'border-green-500/20',
  failed: 'border-red-500/20',
};

const StrokeManager = () => {
  const strokes = useStore((s) => s.strokes);
  const isOpen = useStore((s) => s.strokeManagerOpen);
  const toggleStrokeManager = useStore((s) => s.toggleStrokeManager);

  if (!isOpen) return null;

  return (
    <div className="absolute bottom-0 left-14 right-80 h-48 bg-gray-900/90 backdrop-blur-xl border-t border-white/10 flex flex-col z-40">
      {/* Header */}
      <div className="px-4 py-2 border-b border-white/10 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-2">
          <Layers size={14} className="text-blue-400" />
          <span className="text-xs font-semibold text-white">Stroke Manager</span>
          <span className="text-xs text-gray-500">({strokes.length})</span>
        </div>
        <button
          onClick={toggleStrokeManager}
          className="text-gray-400 hover:text-white text-xs transition-colors"
        >
          Close
        </button>
      </div>

      {/* Stroke list */}
      <div className="flex-1 overflow-y-auto px-2 py-2 space-y-1">
        {strokes.length === 0 ? (
          <p className="text-xs text-gray-500 text-center py-6">
            No strokes yet. Draw on the canvas and apply an effect.
          </p>
        ) : (
          [...strokes].reverse().map((stroke) => (
            <div
              key={stroke.id}
              className={`flex items-center gap-3 px-3 py-2 rounded-lg border ${statusColors[stroke.status]} bg-gray-800/40 transition-colors`}
            >
              {statusIcons[stroke.status]}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-xs text-white font-medium truncate">
                    {stroke.effectName}
                  </span>
                  <span className="text-xs text-gray-500">
                    {new Date(stroke.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                {stroke.error && (
                  <p className="text-xs text-red-400 truncate mt-0.5">{stroke.error}</p>
                )}
              </div>
              <span className="text-xs text-gray-500 capitalize">{stroke.status}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default StrokeManager;
