export const popCountryMap = {
  // North America
  "iad": "United States",  // Washington D.C. (AWS, Cloudflare, Akamai)
  "atl": "United States",  // Atlanta
  "dfw": "United States",  // Dallas-Fort Worth
  "mia": "United States",  // Miami (Key for Latin America traffic)
  "lax": "United States",  // Los Angeles
  "sjc": "United States",  // San Jose (Silicon Valley)
  "yyz": "Canada",         // Toronto (Cloudflare, AWS, Fastly)
  "yul": "Canada",         // Montreal
  "yvr": "Canada",         // Vancouver
  "mex": "Mexico",         // Mexico City (Cloudflare, Akamai, AWS)

  // Central America & Caribbean
  "gye": "Ecuador",        // Guayaquil (Cloudflare, Akamai)
  "pty": "Panama",         // Panama City (Cloudflare, Akamai)
  "sjo": "Costa Rica",     // San José (Cloudflare, Akamai)
  "hav": "Cuba",           // Havana (limited connectivity)
  "sju": "Puerto Rico",    // San Juan (Cloudflare, AWS)

  // South America
  "gru": "Brazil",         // São Paulo (Cloudflare, Akamai, AWS)
  "gig": "Brazil",         // Rio de Janeiro (Cloudflare, AWS)
  "eze": "Argentina",      // Buenos Aires (Cloudflare, AWS, Akamai)
  "scl": "Chile",          // Santiago (Cloudflare, AWS, Akamai)
  "lim": "Peru",           // Lima (Cloudflare, AWS, Akamai)
  "bog": "Colombia",       // Bogotá (Cloudflare, AWS, Akamai)
  "uio": "Ecuador",        // Quito (Cloudflare, Akamai)
  "mvd": "Uruguay",        // Montevideo (Cloudflare, AWS, Akamai)
  "asu": "Paraguay",       // Asunción (Cloudflare, Akamai)
  "lpb": "Bolivia",        // La Paz (Cloudflare, Akamai)

  // Other territories
  "bgi": "Barbados",       // Bridgetown (Cloudflare)
  "aua": "Aruba",          // Oranjestad (Cloudflare)
  "cay": "French Guiana",  // Cayenne (Cloudflare)
  "pos": "Trinidad and Tobago",  // Port of Spain (Cloudflare)
}

// Mapper from code to ISO 3166-1 alpha-2 country codes (uppercase)
export const popCountryIso2Map = {
  "iad": "us",
  "atl": "us",
  "dfw": "us",
  "mia": "us",
  "lax": "us",
  "sjc": "us",
  "yyz": "ca",
  "yul": "ca",
  "yvr": "ca",
  "mex": "mx",

  "gye": "ec",
  "pty": "pa",
  "sjo": "cr",
  "hav": "cu",
  "sju": "pr",

  "gru": "br",
  "gig": "br",
  "eze": "ar",
  "scl": "cl",
  "lim": "pe",
  "bog": "co",
  "uio": "ec",
  "mvd": "uy",
  "asu": "py",
  "lpb": "bo",

  "bgi": "bb",
  "aua": "aw",
  "cay": "gf",
  "pos": "tt",
}
