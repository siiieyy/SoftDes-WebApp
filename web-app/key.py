import os

# Generate a random 32-byte secret key
secret_key = os.urandom(32).hex()
print(secret_key)
