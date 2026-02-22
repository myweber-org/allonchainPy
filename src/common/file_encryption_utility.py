
import os
import base64
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

class FileEncryptor:
    def __init__(self, password):
        self.password = password.encode()
        self.salt = os.urandom(16)
        self.backend = default_backend()
    
    def _derive_key(self, salt):
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=self.backend
        )
        return kdf.derive(self.password)
    
    def encrypt_file(self, input_path, output_path):
        with open(input_path, 'rb') as f:
            plaintext = f.read()
        
        key = self._derive_key(self.salt)
        iv = os.urandom(16)
        
        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(iv),
            backend=self.backend
        )
        encryptor = cipher.encryptor()
        
        pad_length = 16 - (len(plaintext) % 16)
        padded_data = plaintext + bytes([pad_length] * pad_length)
        
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()
        
        with open(output_path, 'wb') as f:
            f.write(self.salt + iv + ciphertext)
        
        return True
    
    def decrypt_file(self, input_path, output_path):
        with open(input_path, 'rb') as f:
            data = f.read()
        
        salt = data[:16]
        iv = data[16:32]
        ciphertext = data[32:]
        
        key = self._derive_key(salt)
        
        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(iv),
            backend=self.backend
        )
        decryptor = cipher.decryptor()
        
        padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        
        pad_length = padded_plaintext[-1]
        plaintext = padded_plaintext[:-pad_length]
        
        with open(output_path, 'wb') as f:
            f.write(plaintext)
        
        return True

def main():
    encryptor = FileEncryptor("secure_password_123")
    
    test_data = b"This is a secret message that needs encryption."
    with open("test_plain.txt", "wb") as f:
        f.write(test_data)
    
    encryptor.encrypt_file("test_plain.txt", "test_encrypted.bin")
    encryptor.decrypt_file("test_encrypted.bin", "test_decrypted.txt")
    
    with open("test_decrypted.txt", "rb") as f:
        decrypted = f.read()
    
    print("Original:", test_data)
    print("Decrypted:", decrypted)
    print("Match:", test_data == decrypted)
    
    os.remove("test_plain.txt")
    os.remove("test_encrypted.bin")
    os.remove("test_decrypted.txt")

if __name__ == "__main__":
    main()