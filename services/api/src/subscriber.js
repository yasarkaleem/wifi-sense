// ZeroMQ SUB client for services/pipeline's PUB socket. Mirrors the
// Python services' zmq usage (see e.g. services/pipeline/src/pipeline/service.py):
// connects (auto-reconnects if pipeline isn't up yet), subscribes to every
// topic pipeline might publish, and forwards parsed events to a callback.

import { Subscriber } from 'zeromq';

const TOPICS = ['presence', 'count', 'zones'];

/**
 * Connects to tcp://host:port and calls onEvent(topic, event) for every
 * message received. Runs until `signal` aborts. Malformed JSON payloads
 * are logged and skipped rather than crashing the process.
 */
export async function subscribeToPipeline({ host, port, onEvent, signal }) {
  const sock = new Subscriber();
  sock.connect(`tcp://${host}:${port}`);
  for (const topic of TOPICS) {
    sock.subscribe(topic);
  }

  console.log(`api: subscribing to pipeline at tcp://${host}:${port} (topics: ${TOPICS.join(', ')})`);

  try {
    for await (const [topicBuf, payloadBuf] of sock) {
      if (signal?.aborted) break;

      const topic = topicBuf.toString('utf-8');
      let event;
      try {
        event = JSON.parse(payloadBuf.toString('utf-8'));
      } catch (err) {
        console.warn(`api: dropping malformed ${topic} payload: ${err.message}`);
        continue;
      }
      onEvent(topic, event);
    }
  } finally {
    sock.close();
  }
}
