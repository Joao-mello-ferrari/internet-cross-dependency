import time
import subprocess
import threading


def run_command(command):
    """Execute a shell command and return its output."""
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None

def run_script(args):
    start_time = time.time()
    print("Starting script", args[1])
    thread = threading.Thread(
        target=subprocess.run,
        args=(args,),
        kwargs={"check": True}
    )
    thread.start()
    thread.join()
    print(f"Time taken for {args[1]}: {time.time() - start_time} seconds \n")


# def latency_match_geolocation(lat1, lon1, lat2, lon2, latency_ms, medium="fiber"):
#     """
#     Check if the given latency is physically possible for the distance.
# 
#     Parameters:
#     lat1, lon1 - Coordinates of point 1 (degrees)
#     lat2, lon2 - Coordinates of point 2 (degrees)
#     latency_ms - Measured latency (milliseconds, round-trip)
#     medium - "fiber" (200,000 km/s) or "air" (300,000 km/s)
# 
#     Returns:
#     True if the latency is realistic, False otherwise
#     """
#     speed = 200_000 if medium == "fiber" else 300_000  # km/s
#     distance = haversine(lat1, lon1, lat2, lon2)
# 
#     max_possible_distance = (latency_ms / 1000) * (speed / 2)  # One-way distance
#     lowered_distance = max_possible_distance * 0.8  # 20% less distance for latency
# 
#     return 0.5 < distance / lowered_distance < 2
