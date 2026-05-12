/* ═══════════════════════════════════════════════════════════════════════════
   KOSPI ORACLE V9 — SERVICE WORKER
   ───────────────────────────────────────────────────────────────────────────
   ▸ index.html은 항상 캐시
   ▸ data.json은 네트워크 우선 (실시간성 중요)
   ▸ 외부 라이브러리(Chart.js)는 캐시 우선
══════════════════════════════════════════════════════════════════════════ */

const CACHE_VERSION = 'kospi-oracle-v9-001';
const CACHE_NAME = `oracle-${CACHE_VERSION}`;

// 정적 자원만 사전 캐시 (data.json 등 동적 파일은 제외)
const STATIC_ASSETS = [
  './',
  './index.html',
  './manifest.json',
];

// ─── INSTALL: 정적 자원 사전 캐시 ──────────────────────
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('[SW] 정적 자원 캐싱:', STATIC_ASSETS);
      return cache.addAll(STATIC_ASSETS);
    })
  );
  self.skipWaiting();
});

// ─── ACTIVATE: 옛 캐시 정리 ─────────────────────────────
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys.filter((k) => k.startsWith('oracle-') && k !== CACHE_NAME)
            .map((k) => { console.log('[SW] 옛 캐시 삭제:', k); return caches.delete(k); })
      )
    )
  );
  self.clients.claim();
});

// ─── FETCH: 요청별 전략 ────────────────────────────────
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // POST/PUT 등은 패스
  if (event.request.method !== 'GET') return;

  // ─ data.json / history.json / prediction_log.json: 네트워크 우선 ─
  if (url.pathname.endsWith('data.json') ||
      url.pathname.endsWith('history.json') ||
      url.pathname.endsWith('prediction_log.json')) {
    event.respondWith(networkFirst(event.request));
    return;
  }

  // ─ 외부 CDN (Chart.js, Pretendard): 캐시 우선 ─
  if (url.origin !== self.location.origin) {
    event.respondWith(cacheFirst(event.request));
    return;
  }

  // ─ 그 외 (index.html 등): 캐시 우선 ─
  event.respondWith(cacheFirst(event.request));
});

// ─── 전략 1: 네트워크 우선 (실시간 데이터용) ─────────────
async function networkFirst(request) {
  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, response.clone());
    }
    return response;
  } catch (err) {
    const cached = await caches.match(request);
    if (cached) return cached;
    return new Response(JSON.stringify({ error: 'offline' }), {
      status: 503,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}

// ─── 전략 2: 캐시 우선 (정적 자원용) ───────────────────
async function cacheFirst(request) {
  const cached = await caches.match(request);
  if (cached) {
    // 백그라운드에서 캐시 갱신
    fetch(request).then((response) => {
      if (response.ok) {
        caches.open(CACHE_NAME).then((cache) => cache.put(request, response));
      }
    }).catch(() => {});
    return cached;
  }
  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, response.clone());
    }
    return response;
  } catch (err) {
    return new Response('Offline', { status: 503 });
  }
}
