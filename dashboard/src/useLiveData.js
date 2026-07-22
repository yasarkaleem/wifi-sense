import { useEffect, useRef, useState } from 'react';

import { WS_URL, fetchHistory, fetchStatus } from './api.js';

const HISTORY_MINUTES = 10;
const HISTORY_WINDOW_US = HISTORY_MINUTES * 60 * 1_000_000;
const RECONNECT_DELAY_MS = 2000;

const EMPTY_STATE = {
  timestamp: null,
  presence: null,
  motion_intensity: null,
  count: null,
  confidence: null,
  zones: null,
};

function trimToWindow(pointsByTimestamp) {
  if (pointsByTimestamp.size === 0) return pointsByTimestamp;
  const newestUs = Math.max(...pointsByTimestamp.keys());
  const cutoffUs = newestUs - HISTORY_WINDOW_US;
  for (const ts of pointsByTimestamp.keys()) {
    if (ts < cutoffUs) pointsByTimestamp.delete(ts);
  }
  return pointsByTimestamp;
}

/**
 * Live dashboard data: current merged state (from GET /status + WebSocket
 * pushes) and a rolling `HISTORY_MINUTES`-minute history (backfilled from
 * GET /history, then extended live from the same WebSocket stream).
 * Reconnects automatically if the WebSocket drops (e.g. api restarting).
 */
export function useLiveData() {
  const [current, setCurrent] = useState(EMPTY_STATE);
  const [history, setHistory] = useState([]);
  const [connectionStatus, setConnectionStatus] = useState('connecting');
  const pointsRef = useRef(new Map()); // timestamp_us -> snapshot, sorted on read

  useEffect(() => {
    let cancelled = false;

    fetchStatus()
      .then((status) => {
        if (!cancelled) setCurrent(status);
      })
      .catch((err) => console.warn('failed to fetch /status:', err));

    fetchHistory(HISTORY_MINUTES)
      .then(({ snapshots }) => {
        if (cancelled) return;
        for (const snap of snapshots) {
          pointsRef.current.set(snap.timestamp, snap);
        }
        setHistory([...trimToWindow(pointsRef.current).values()].sort((a, b) => a.timestamp - b.timestamp));
      })
      .catch((err) => console.warn('failed to fetch /history:', err));

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let socket;
    let reconnectTimer;
    let closedByEffect = false;

    function connect() {
      setConnectionStatus('connecting');
      socket = new WebSocket(WS_URL);

      socket.onopen = () => setConnectionStatus('open');

      socket.onmessage = (event) => {
        const snapshot = JSON.parse(event.data);
        setCurrent(snapshot);
        if (snapshot.timestamp != null) {
          pointsRef.current.set(snapshot.timestamp, snapshot);
          setHistory([...trimToWindow(pointsRef.current).values()].sort((a, b) => a.timestamp - b.timestamp));
        }
      };

      socket.onclose = () => {
        setConnectionStatus('closed');
        if (!closedByEffect) {
          reconnectTimer = setTimeout(connect, RECONNECT_DELAY_MS);
        }
      };

      socket.onerror = () => socket.close();
    }

    connect();

    return () => {
      closedByEffect = true;
      clearTimeout(reconnectTimer);
      socket?.close();
    };
  }, []);

  return { current, history, connectionStatus };
}
