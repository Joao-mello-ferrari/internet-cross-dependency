
sort-ip:
	(head -n 1 db/csv_ipv4.csv && tail -n +2 db/csv_ipv4.csv | sort -t, -k12,12n) > db/sorted_csv_ipv4.csv

sort-anycast:
	(head -n 1 db/anycast_ipv4.csv && tail -n +2 db/anycast_ipv4.csv | sort -t, -k4,4n) > db/sorted_anycast_csv_ipv4.csv