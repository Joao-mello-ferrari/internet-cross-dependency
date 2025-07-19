import { popCountryMap, popCountryIso2Map } from './popMap.js';

// ==============================
// Helper Functions
// ==============================
export async function fetchHeaders(urls) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2000);

    const promises = urls.map(url => fetchFunc(url, controller));
    const results = await Promise.allSettled(promises);

    clearTimeout(timeoutId);

    return urls.map((url, idx) => {
        if (results[idx].status === "fulfilled") {
            const response = results[idx].value;
            const headersArray = [];
            response.headers.forEach((value, name) => {
                headersArray.push({ name, value });
            });
            return { url, headers: headersArray };
        } else {
            return { url, headers: [] };
        }
    });
}

export function formatEdgeInfo(edgeInfo) {
    const { pop, provider, location, cacheStatus } = edgeInfo;

    return omitEmpty({
        pop,
        provider,
        location,
        cacheStatus,
        contentLocality: getLocality(pop),
    });
}

export function chunkArray(array, size = 20) {
    let result = [];
    for (let i = 0; i < array.length; i += size) {
        result.push(array.slice(i, i + size));
    }
    return result;
}

function getLocality(pop) {
    if (!(pop && pop.length && popCountryIso2Map[pop[0]])) {
        return null;
    }
    return popCountryIso2Map[pop[0]];
}

function omitEmpty(obj) {
    return Object.fromEntries(
        Object.entries(obj).filter(([_, v]) =>
            v !== null && v !== undefined &&
            (!(Array.isArray(v) || typeof v === 'string') || v.length > 0)
        )
    );
}

async function fetchFunc(url, controller) {
    return fetch(url, {
        method: 'HEAD',
        headers: {
            'User-Agent': 'Mozilla/5.0 (compatible)'
        },
        signal: controller.signal
    });
}
