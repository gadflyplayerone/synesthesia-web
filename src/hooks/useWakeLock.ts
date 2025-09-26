// src/hooks/useWakeLock.ts
import { useEffect, useRef } from "react";

/**
 * Keep the screen awake while `enabled` is true and the tab is visible.
 * Uses the Screen Wake Lock API with automatic re-acquisition on visibility changes.
 */
export function useWakeLock(enabled: boolean) {
  const sentinelRef = useRef<any | null>(null);
  const supported = typeof navigator !== "undefined" && "wakeLock" in navigator;

  useEffect(() => {
    if (!enabled || !supported) return;

    let cancelled = false;

    const request = async () => {
      try {
        // Only request when visible
        if (document.visibilityState !== "visible") return;
        // Avoid duplicate requests
        if (sentinelRef.current) return;
        // @ts-ignore
        const sentinel = await navigator.wakeLock.request("screen");
        if (cancelled) {
          try { await sentinel.release(); } catch {}
          return;
        }
        sentinelRef.current = sentinel;

        sentinel.addEventListener?.("release", () => {
          // If it releases while we still want it and we're visible, try again
          if (!cancelled && enabled && document.visibilityState === "visible") {
            sentinelRef.current = null;
            request();
          }
        });
      } catch {
        // Some platforms need a user gesture first or deny silently.
        // We fail silently to avoid console noise.
      }
    };

    const onVisibility = () => {
      if (document.visibilityState === "visible") {
        request();
      } else {
        release();
      }
    };

    const onPageShow = () => request();
    const onPageHide = () => release();

    const release = async () => {
      const s = sentinelRef.current;
      sentinelRef.current = null;
      if (s) {
        try { await s.release(); } catch {}
      }
    };

    // Prime it once
    request();

    document.addEventListener("visibilitychange", onVisibility);
    window.addEventListener("pageshow", onPageShow);
    window.addEventListener("pagehide", onPageHide);

    return () => {
      cancelled = true;
      document.removeEventListener("visibilitychange", onVisibility);
      window.removeEventListener("pageshow", onPageShow);
      window.removeEventListener("pagehide", onPageHide);
      release();
    };
  }, [enabled, supported]);
}
