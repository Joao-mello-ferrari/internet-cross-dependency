import { popCountryIso2Map } from './popMap.js';

// ==============================
// Helper Functions
// ==============================
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


function getLocality(pop) {
    if (!pop || !Array.isArray(pop) || pop.length === 0) {
        return null;
    }
    for (const iata of pop) {
        if (popCountryIso2Map[iata]) {
            return popCountryIso2Map[iata];
        }
    }
    return null;
}

function omitEmpty(obj) {
    return Object.fromEntries(
        Object.entries(obj).filter(([_, v]) =>
            v !== null && v !== undefined &&
            (!(Array.isArray(v) || typeof v === 'string') || v.length > 0)
        )
    );
}
