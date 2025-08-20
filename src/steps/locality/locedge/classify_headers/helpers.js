import { popCountryMap, popCountryIso2Map } from './popMap.js';

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
