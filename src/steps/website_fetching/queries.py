def country_global_intersection(country_code, offset, limit=100):
    return f"""
    WITH c AS (
      SELECT DISTINCT origin, experimental.popularity.rank AS rank, country_code
      FROM `chrome-ux-report.experimental.country`
      WHERE yyyymm = 202501
      AND country_code IN ('{country_code}')
      ORDER BY experimental.popularity.rank
      LIMIT 1000
    ),
    g AS (
      SELECT DISTINCT origin, experimental.popularity.rank
      FROM `chrome-ux-report.experimental.global`
      WHERE yyyymm = 202501
      ORDER BY experimental.popularity.rank
      LIMIT 1000
    )
    SELECT c.origin, c.rank 
    FROM c 
    JOIN g ON c.origin = g.origin
    ORDER BY origin
    LIMIT {limit}
    OFFSET {offset} 
    """

def country(country_code, offset, limit=100, semester="202501"):
    #SELECT DISTINCT origin, experimental.popularity.rank AS rank, country_code
    return f"""
    SELECT DISTINCT origin
    FROM `chrome-ux-report.experimental.country`
    WHERE yyyymm = {semester}
    AND country_code IN ('{country_code}')
    AND experimental.popularity.rank in (1000)
    ORDER BY origin
    LIMIT {limit}
    OFFSET {offset}
    """

def get_global(country_code, offset, limit=100, semester="202501"):
    return f"""
    SELECT DISTINCT origin
    FROM `chrome-ux-report.experimental.global`
    WHERE yyyymm = {semester}
    AND experimental.popularity.rank in (1000)
    ORDER BY origin
    LIMIT {limit}
    OFFSET {offset}
    """
