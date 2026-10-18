# Major Incident Log 2026

## INC-2041 — Halden Logistics picking outage
On 2026-05-14 the AX-200 arms at Halden Logistics froze during the night shift. Picking was down for 6 hours before arms were restarted manually. Root cause: the memory leak in the grip controller introduced in firmware 4.2. FleetOS rolled firmware 4.2.1 out to the site on 2026-05-22 and the issue has not recurred. Support lead Arjun Rao ran the post-incident review.

## INC-2057 — Sunpeak Grocers battery warnings
Between 2026-06-03 and 2026-06-09 several Courier C3 robots at Sunpeak Grocers raised battery thermal warnings and parked themselves. There was no picking outage; throughput dropped by about 15% for two days. Root cause: a faulty batch of battery cells from the external cell supplier. Firmware was not involved. The affected packs were replaced under warranty.
