import fs from 'fs';
import parse from './index.js';
import { popCountryMap, popCountryIso2Map } from './popMap.js';
const countryCode = process.argv[2]; // Get the country code from CLI args
const countryName = process.argv[3]; // Get the country name from CLI args
const vpn = process.argv[4]; // Get the country name from CLI args
if (!countryCode || !countryName) {
    console.error("Usage: node script.js <country_code>");
    process.exit(1);
}


// Load websites from JSON
const jsonFile = `./results/${countryCode}/output.json`; // Update this path if needed

try {
    const data = fs.readFileSync(jsonFile, 'utf-8');
    const websitesData = JSON.parse(data);
    const outputData = {}; // Store results here
    let counter = 0;
    for (const sites of chunkArray(websitesData)) {
        const result = await fetchHeaders(sites);
        result.forEach(({ headers }, idx) => {
            const harData = { log: { entries: [{ response: { headers } }] } };
            const parsed = parse(harData);
            parsed.log.entries.forEach((host) => {
                const key = sites[idx].replace(/^https?:\/\//, '');
                outputData[key] = formatEdgeInfo(host._edgeInfo); // Store in object
            });
            counter++;
        });
        process.stdout.write(`Processed ${counter * 100 / websitesData.length}% of ${countryName}\r`);
    }

    // Write results to `output.json`
    try {
        fs.writeFile(`results/${countryCode}/locality/${vpn}/locedge.json`, JSON.stringify(outputData, null, 2), {}, (err) => { });
        //console.log("Results saved to output.json");
    } catch (error) {
        console.error("Error writing to file:", error);
    }

} catch (error) {
    console.error("Error reading websites JSON:", error);
    process.exit(1);
}

async function fetchHeaders(urls) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2000);

    const promises = urls.map(url => fetchFunc(url, controller));

    // Use Promise.allSettled to handle both fulfilled & rejected promises
    const results = await Promise.allSettled(promises);

    clearTimeout(timeoutId);
    
    let finalResults = urls.map((url, idx) => {
        if (results[idx].status === "fulfilled") {
            const response = results[idx].value;
            const headersArray = [];
            response.headers.forEach((value, name) => {
                headersArray.push({ name, value });
            });
            return { url, headers: headersArray };
        } else {
            return { url, headers: [] }; // Handle failed requests gracefully
        }
    });

    return finalResults;
}

function formatEdgeInfo(edgeInfo) {
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
    //if (!(pop && pop.length && popCountryMap[pop[0]])) {
    //    return null;
    //}

    //const country = popCountryMap[pop[0]];
    //return country === countryName ? "Local" : "External";
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

function chunkArray(array, size = 20) {
    let result = [];
    for (let i = 0; i < array.length; i += size) {
        result.push(array.slice(i, i + size));
    }
    return result;
}

async function fetchFunc(url, controller) {
    return fetch(url, {
        method: 'HEAD',
        headers: {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        },
        signal: controller.signal
    });
}   