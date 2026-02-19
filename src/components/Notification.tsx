import { useEffect } from 'react';
import { CheckCircle2, XCircle, Info } from 'lucide-react';
import { useStore } from '../store';

const icons = {
  success: <CheckCircle2 size={16} className="text-green-400" />,
  error: <XCircle size={16} className="text-red-400" />,
  info: <Info size={16} className="text-blue-400" />,
};

const bgColors = {
  success: 'border-green-500/20 bg-green-900/20',
  error: 'border-red-500/20 bg-red-900/20',
  info: 'border-blue-500/20 bg-blue-900/20',
};

const Notification = () => {
  const notification = useStore((s) => s.notification);
  const clearNotification = useStore((s) => s.clearNotification);

  useEffect(() => {
    if (!notification) return;
    const timer = setTimeout(clearNotification, 3000);
    return () => clearTimeout(timer);
  }, [notification, clearNotification]);

  if (!notification) return null;

  return (
    <div className="fixed top-4 right-4 z-[100] animate-in fade-in slide-in-from-top-2">
      <div
        className={`flex items-center gap-2 px-4 py-2.5 rounded-lg border backdrop-blur-xl shadow-lg ${bgColors[notification.type]}`}
      >
        {icons[notification.type]}
        <span className="text-sm text-white">{notification.message}</span>
      </div>
    </div>
  );
};

export default Notification;
