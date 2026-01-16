import Recorder from "./Recorder.jsx";
import { useEffect } from "react";

useEffect(() => {
  const pingBackend = async () => {
    try {
      await fetch("https://ia-melodie.onrender.com/ping");
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


function App() {
  return (
    <div>
      <Recorder />
    </div>
  );
}

export default App;
