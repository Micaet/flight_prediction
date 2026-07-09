"""Pobieranie surowych danych (BTS On-Time Performance, OpenSky, Open-Meteo).

Środowisko ma inspekcję TLS (firmowy/AV root CA). `truststore.inject_into_ssl()`
przełącza weryfikację certyfikatów w Pythonie na systemowy magazyn Windows,
który ten root CA zna — dzięki temu `requests` nie rzuca CERTIFICATE_VERIFY_FAILED.
Wołane raz przy imporcie pakietu ingestion (bezpieczniejsze niż verify=False).
"""

import truststore

truststore.inject_into_ssl()
