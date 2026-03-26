from desilofhe import Engine

engine = Engine()
secret_key = engine.create_secret_key()
public_key = engine.create_public_key(secret_key)
relinearization_key = engine.create_relinearization_key(secret_key)

message = [1, 2, 3]
ciphertext = engine.encrypt(message, public_key)

squared = engine.square(ciphertext, relinearization_key)
print(engine.decrypt(squared, secret_key))

