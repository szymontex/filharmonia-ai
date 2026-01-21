import { useState, useEffect, useRef, useCallback } from 'react';

interface UseExponentialPollingOptions {
  initialInterval: number;  // Starting interval in ms (e.g., 1000)
  maxInterval: number;      // Maximum interval in ms (e.g., 30000)
  multiplier: number;       // Growth factor (e.g., 1.5)
  resetOnChange: boolean;   // Reset interval when data changes
}

interface UseExponentialPollingResult<T> {
  data: T | null;
  error: Error | null;
  isPolling: boolean;
  startPolling: () => void;
  stopPolling: () => void;
  currentInterval: number;
}

export function useExponentialPolling<T>(
  fetchFn: () => Promise<T>,
  options: UseExponentialPollingOptions = {
    initialInterval: 1000,
    maxInterval: 30000,
    multiplier: 1.5,
    resetOnChange: true
  }
): UseExponentialPollingResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [isPolling, setIsPolling] = useState(false);
  const [currentInterval, setCurrentInterval] = useState(options.initialInterval);

  const intervalRef = useRef(options.initialInterval);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const previousDataRef = useRef<string>('');
  const mountedRef = useRef(true);
  const isPollingRef = useRef(false);

  const poll = useCallback(async () => {
    if (!mountedRef.current || !isPollingRef.current) return;

    try {
      const result = await fetchFn();
      if (!mountedRef.current) return;

      setData(result);
      setError(null);

      const dataStr = JSON.stringify(result);

      // Check if data changed
      if (options.resetOnChange && dataStr !== previousDataRef.current) {
        // Reset to fast polling on change
        intervalRef.current = options.initialInterval;
      } else {
        // Exponential backoff when stable
        intervalRef.current = Math.min(
          intervalRef.current * options.multiplier,
          options.maxInterval
        );
      }

      setCurrentInterval(intervalRef.current);
      previousDataRef.current = dataStr;

    } catch (err) {
      if (!mountedRef.current) return;
      setError(err as Error);
      // On error, still backoff to avoid hammering server
      intervalRef.current = Math.min(
        intervalRef.current * options.multiplier,
        options.maxInterval
      );
      setCurrentInterval(intervalRef.current);
    }

    // Schedule next poll if still polling
    if (mountedRef.current && isPollingRef.current) {
      timeoutRef.current = setTimeout(poll, intervalRef.current);
    }
  }, [fetchFn, options]);

  const startPolling = useCallback(() => {
    intervalRef.current = options.initialInterval;
    setCurrentInterval(options.initialInterval);
    setIsPolling(true);
    isPollingRef.current = true;
  }, [options.initialInterval]);

  const stopPolling = useCallback(() => {
    setIsPolling(false);
    isPollingRef.current = false;
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
  }, []);

  // Start polling when isPolling becomes true
  useEffect(() => {
    if (isPolling) {
      poll();
    }
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
    };
  }, [isPolling, poll]);

  // Cleanup on unmount
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      isPollingRef.current = false;
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  return { data, error, isPolling, startPolling, stopPolling, currentInterval };
}

export default useExponentialPolling;
