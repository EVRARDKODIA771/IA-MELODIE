import Recorder from "./Recorder.jsx";
import { useEffect } from "react";

// ======================
// URL du backend depuis .env
// ======================
const backendUrl = import.meta.env.VITE_BACKEND_URL;

function App() {
  // ======================
  // Anti-sommeil / Ping backend
  // ======================
  useEffect(() => {
    const pingBackend = async () => {
      try {
        await fetch(`${backendUrl}/ping`);
      } catch (err) {
        console.error("Ping backend failed", err);
      }
    };

    // Ping immédiat au chargement
    pingBackend();

    // Ping toutes les 5 minutes
    const interval = setInterval(pingBackend, 5 * 60 * 1000);

    // Cleanup à la destruction du composant
    return () => clearInterval(interval);
  }, []);

  return (
    <div>
      <Recorder />
    </div>
  );
}

export default App;
