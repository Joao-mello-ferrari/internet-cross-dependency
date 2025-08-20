import argparse
import json
import asyncio
import aiohttp
import time
import random
from pathlib import Path

from tqdm.asyncio import tqdm


# ===================
# Helper Functions
# ===================
async def fetch_headers(session, url, semaphore, pbar, request_timeout=10.0, max_retries=5):
    """Fetch headers for a single URL with timeout and error handling."""
    original_url = url
    attempt = 0
    
    async with semaphore:
        while attempt <= max_retries:
            try:
                # Ensure URL has protocol
                if not url.startswith(('http://', 'https://')):
                    url = f'https://{url}'
                
                timeout = aiohttp.ClientTimeout(total=request_timeout)
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                async with session.head(url, timeout=timeout, headers=headers, allow_redirects=True) as response:
                    # Convert headers to dict format
                    headers_dict = {}
                    for name, value in response.headers.items():
                        headers_dict[name.lower()] = value
                    
                    result = {
                        'url': str(response.url),
                        'status': response.status,
                        'headers': headers_dict,
                        'success': True,
                        'attempts': attempt + 1
                    }
                    
                    pbar.update(1)
                    return result
                    
            except asyncio.TimeoutError:
                error_msg = 'TIMEOUT'
                should_retry = True
            except aiohttp.ClientError as e:
                error_msg = f'CLIENT_ERROR: {str(e)}'
                # Don't retry on certain client errors
                should_retry = not any(x in str(e).lower() for x in [
                    'header value is too long',
                    'ssl certificate',
                    'certificate verify failed',
                    'hostname mismatch'
                ])
            except Exception as e:
                error_msg = f'UNKNOWN_ERROR: {str(e)}'
                should_retry = True
            
            attempt += 1
            
            # If we should retry and haven't exceeded max retries
            if should_retry and attempt <= max_retries:
                # Exponential backoff with jitter
                delay = (2 ** (attempt - 1)) + random.uniform(0, 1)
                await asyncio.sleep(delay)
                continue
            else:
                # Final failure
                result = {
                    'url': original_url,
                    'success': False,
                    'error': error_msg,
                    'headers': {},
                    'attempts': attempt
                }
                
                pbar.update(1)
                return result


async def fetch_all_headers(websites, max_concurrent=50, request_timeout=10.0, max_retries=3):
    """Fetch headers for all websites with controlled concurrency."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Configure connector with larger header limits
    connector = aiohttp.TCPConnector(
        limit=100,
        limit_per_host=10,
        ttl_dns_cache=300,
        use_dns_cache=True,
        keepalive_timeout=30,
        enable_cleanup_closed=True
    )
    
    # Configure client session with larger header limits
    async with aiohttp.ClientSession(
        connector=connector,
        connector_owner=True,
        read_bufsize=64 * 1024,  # 64KB buffer
        max_line_size=8190 * 8,  
        max_field_size=8190 * 8,
        headers={'Connection': 'keep-alive'}
    ) as session:
        # Create progress bar
        with tqdm(total=len(websites), desc="Fetching headers", unit="site") as pbar:
            tasks = [
                fetch_headers(session, website, semaphore, pbar, request_timeout, max_retries) 
                for website in websites
            ]
            
            # Process in batches to avoid overwhelming the system
            batch_size = 1000
            results = {}
            
            for i in range(0, len(tasks), batch_size):
                batch_tasks = tasks[i:i + batch_size]
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        continue
                        
                    # Use original URL as key, store result as value
                    original_url = result['url']
                    results[original_url] = result
                
                # Small delay between batches to be respectful
                await asyncio.sleep(0.1)
    
    return results


def load_websites(input_file):
    """Load websites from the JSON output file."""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract unique websites from the data structure
        websites = set()
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict) and 'website' in value:
                    websites.add(value['website'])
                elif isinstance(value, str):
                    # Handle direct website entries
                    if '.' in value and ' ' not in value:
                        websites.add(value)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and 'website' in item:
                    websites.add(item['website'])
                elif isinstance(item, str):
                    websites.add(item)
        
        # Clean and validate websites
        clean_websites = []
        for website in websites:
            if website and isinstance(website, str):
                # Clean the URL
                website = website.strip()
                if website and '.' in website:
                    clean_websites.append(website)
        
        return list(set(clean_websites))  # Remove duplicates
        
    except Exception as e:
        print(f"[❌] Error loading websites from {input_file}: {e}")
        return []


def save_json(data, filepath):
    """Save results to JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def print_statistics(results):
    """Print statistics about the fetch results."""
    total = len(results)
    successful = sum(1 for r in results.values() if r.get('success', False))
    failed = total - successful
    
    # Calculate retry statistics
    total_attempts = sum(r.get('attempts', 1) for r in results.values())
    retried_requests = sum(1 for r in results.values() if r.get('attempts', 1) > 1)
    
    print("\n" + "="*50)
    print("FETCH STATISTICS")
    print("="*50)
    print(f"Total websites: {total}")
    print(f"Successful: {successful} ({(successful/total*100):.1f}%)")
    print(f"Failed: {failed} ({(failed/total*100):.1f}%)")
    print(f"Total attempts: {total_attempts}")
    print(f"Requests retried: {retried_requests} ({(retried_requests/total*100):.1f}%)")
    print(f"Average attempts per request: {total_attempts/total:.2f}")
    
    # Count error types
    error_types = {}
    for result in results.values():
        if not result.get('success', True):
            error = result.get('error', 'UNKNOWN')
            error_type = error.split(':')[0] if ':' in error else error
            error_types[error_type] = error_types.get(error_type, 0) + 1
    
    if error_types:
        print("\nError breakdown:")
        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {error_type}: {count}")
    
    # Show retry success statistics
    successful_after_retry = sum(
        1 for r in results.values() 
        if r.get('success', False) and r.get('attempts', 1) > 1
    )
    if successful_after_retry > 0:
        print(f"\nSuccessful after retry: {successful_after_retry}")


# ===================
# Main Logic
# ===================
async def main():
    parser = argparse.ArgumentParser(description="Fetch HTTP headers for websites from JSON output.")
    parser.add_argument("--country", type=str.lower, required=True, help="Country code (label)")
    parser.add_argument("--code", type=str.lower, required=True, help="Country code (folder)")
    parser.add_argument("--vpn", type=str.lower, required=True, help="VPN country code (locality folder)")
    parser.add_argument("--concurrent", "-c", type=int, default=100, help="Maximum concurrent requests (default: 100)")
    parser.add_argument("--timeout", "-t", type=float, default=10.0, help="Request timeout in seconds (default: 10.0)")
    parser.add_argument("--retries", "-r", type=int, default=5, help="Maximum retry attempts (default: 5)")
    
    args = parser.parse_args()
    
    # Paths
    base_path = Path(f"results/{args.code}")
    input_file = base_path / "output.json"
    output_file = base_path / "locality" / args.vpn / "edgeHeaders.json"
    
    print(f"Loading websites from: {input_file}")
    
    # Load websites
    websites = load_websites(input_file)
    if not websites:
        print("[❌] No websites found in input file!")
        return
    
    print(f"Found {len(websites)} unique websites")
    print(f"Starting header fetch with {args.concurrent} concurrent requests...")
    start_time = time.time()
    
    # Fetch headers with progress bar and retry logic
    results = await fetch_all_headers(websites, args.concurrent, args.timeout, args.retries)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\nCompleted in {duration:.2f} seconds")
    print(f"Average: {len(websites)/duration:.1f} requests/second")
    
    # Save results
    save_json(results, output_file)
    print(f"\n✅ Headers data saved to {output_file}")
    
    # Print statistics
    print_statistics(results)


if __name__ == "__main__":
    asyncio.run(main())
