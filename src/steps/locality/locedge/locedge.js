import fs from 'fs';
import cliProgress from 'cli-progress';

import parse from './index.js';
import { chunkArray, fetchHeaders, formatEdgeInfo } from './helpers.js';

// ==============================
// Args
// ==============================
const args = process.argv.slice(2);

function getArg(flag) {
    const index = args.indexOf(flag);
    if (index !== -1 && index + 1 < args.length) {
        return args[index + 1];
    }
    return null;
}

const countryName = getArg('--country');
const countryCode = getArg('--code');
const vpn = getArg('--vpn');

if (!countryCode || !countryName || !vpn) {
    console.error('Usage: node script.js --country <code> --name <name> --vpn <vpn>');
    process.exit(1);
}

// ==============================
// Load Websites
// ==============================
const jsonFile = `./results/${countryCode}/output.json`;

let websitesData = [];
try {
    const data = fs.readFileSync(jsonFile, 'utf-8');
    websitesData = JSON.parse(data);
} catch (error) {
    console.error("❌ Error reading websites JSON:", error);
    process.exit(1);
}

const outputData = {};

// ==============================
// Progress Bar Init
// ==============================
const bar = new cliProgress.SingleBar({
    format: `Progress | {bar} | {percentage}% | {value}/{total} Websites`,
    barCompleteChar: '\u2588',
    barIncompleteChar: '\u2591',
    hideCursor: true
});
bar.start(websitesData.length, 0);

// ==============================
// Processing
// ==============================
let counter = 0;
for (const sites of chunkArray(websitesData)) {
    const result = await fetchHeaders(sites);
    result.forEach(({ headers }, idx) => {
        const harData = { log: { entries: [{ response: { headers } }] } };
        const parsed = parse(harData);
        parsed.log.entries.forEach((host) => {
            const key = sites[idx].replace(/^https?:\/\//, '');
            outputData[key] = formatEdgeInfo(host._edgeInfo);
        });
        counter++;
        bar.increment();
    });
}

bar.stop();

// ==============================
// Write Results
// ==============================
const outputPath = `results/${countryCode}/locality/${vpn}/locedge.json`;
try {
    fs.mkdirSync(`results/${countryCode}/locality/${vpn}`, { recursive: true });
    fs.writeFileSync(outputPath, JSON.stringify(outputData, null, 2));
    console.log(`✅ Results saved to ${outputPath}`);
} catch (error) {
    console.error("❌ Error writing to file:", error);
}