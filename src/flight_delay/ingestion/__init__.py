"""Raw data download.

My machine does TLS inspection with its own root CA, so `requests` fails with
CERTIFICATE_VERIFY_FAILED. truststore makes Python use the Windows cert store,
which knows that CA.
"""

import truststore

truststore.inject_into_ssl()
