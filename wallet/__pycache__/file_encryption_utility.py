import os
import base64
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

class SecureFileEncryptor:
    def __init__(self, salt_length=16, iterations=100000):
        self.salt_length = salt_length
        self.iterations = iterations
        self.backend = default_backend()

    def derive_key(self, password, salt):
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self.iterations,
            backend=self.backend
        )
        return kdf.derive(password.encode())

    def encrypt_file(self, input_path, output_path, password):
        salt = os.urandom(self.salt_length)
        key = self.derive_key(password, salt)
        
        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=self.backend)
        encryptor = cipher.encryptor()
        
        with open(input_path, 'rb') as infile:
            plaintext = infile.read()
        
        padding_length = 16 - (len(plaintext) % 16)
        plaintext += bytes([padding_length]) * padding_length
        
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        
        with open(output_path, 'wb') as outfile:
            outfile.write(salt)
            outfile.write(iv)
            outfile.write(ciphertext)

    def decrypt_file(self, input_path, output_path, password):
        with open(input_path, 'rb') as infile:
            salt = infile.read(self.salt_length)
            iv = infile.read(16)
            ciphertext = infile.read()
        
        key = self.derive_key(password, salt)
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=self.backend)
        decryptor = cipher.decryptor()
        
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        padding_length = plaintext[-1]
        plaintext = plaintext[:-padding_length]
        
        with open(output_path, 'wb') as outfile:
            outfile.write(plaintext)

def generate_secure_password(length=32):
    return base64.urlsafe_b64encode(os.urandom(length)).decode()[:length]

if __name__ == "__main__":
    encryptor = SecureFileEncryptor()
    
    test_data = b"This is a secret message that needs encryption."
    with open('test_plain.txt', 'wb') as f:
        f.write(test_data)
    
    password = generate_secure_password()
    print(f"Generated password: {password}")
    
    encryptor.encrypt_file('test_plain.txt', 'test_encrypted.bin', password)
    encryptor.decrypt_file('test_encrypted.bin', 'test_decrypted.txt', password)
    
    with open('test_decrypted.txt', 'rb') as f:
        decrypted = f.read()
    
    print(f"Decryption successful: {decrypted == test_data}")
    
    for file in ['test_plain.txt', 'test_encrypted.bin', 'test_decrypted.txt']:
        if os.path.exists(file):
            os.remove(file)